"""Local rehearsal lookup: no model loading, generation, networking, or downloads.

Unreviewed results are structural/lexical candidates, not semantic proof. Cached
verdicts are reused only for identical target context and material contents.
"""
from collections import Counter, defaultdict
from functools import lru_cache
import json
import math
import re

import numpy as np

from .rehearsal_retrieval import _current, _quote_in, _text

_STOP = set('a an the of to in on at by for from with and or as is are was were be been being '
            'do does did can could would should may might will have has had this that these those '
            'it its their they them we you our your i us'.split())
_NEGATIVE = {'not', 'no', 'never', 'without', 'cannot'}
_OPPOSITES = [('increase', 'decrease'), ('protect', 'harm'), ('improve', 'worsen'),
              ('enable', 'prevent'), ('safe', 'unsafe'), ('necessary', 'unnecessary')]


def _normalize(text):
    return ' '.join(text.casefold().split())


@lru_cache(maxsize=16384)
def _terms(text):
    words = re.findall(r"[a-z0-9]+", text.casefold().replace("n't", ' not'))
    # A small inflection normalizer; these terms do not constitute an NLI model.
    result = []
    for word in words:
        if word in _STOP:
            continue
        if word.endswith('ing') and len(word) > 5:
            word = word[:-3]
        elif word.endswith('s') and not word.endswith(('ss', 'is')) and len(word) > 4:
            word = word[:-1]
        result.append(word)
    return tuple(result)


def polarity_conflict(target, anchor):
    """Reject obvious opposing anchors; absence of a conflict is not entailment."""
    left, right = set(_terms(target)), set(_terms(anchor))
    a, b = left-_NEGATIVE, right-_NEGATIVE
    overlap = len(a & b)/max(1, len(a | b))
    if overlap >= .8 and bool(left & _NEGATIVE) != bool(right & _NEGATIVE):
        return True
    if not left & _NEGATIVE and not right & _NEGATIVE:
        for positive, negative in _OPPOSITES:
            if ((positive in left and negative in right) or (negative in left and positive in right)):
                a, b = left-{positive, negative}, right-{positive, negative}
                if len(a & b)/max(1,len(a | b)) >= .5:
                    return True
    return False


def verdict_key(motion, action, candidate, material):
    return (motion, action['action'], action['target_claim'], _text(action.get('target_argument', '')),
            json.dumps(action.get('target_context', {}), sort_keys=True, ensure_ascii=False),
            candidate['claim'], candidate.get('argument', ''), candidate.get('parent_claim', ''),
            material['claim'], material['argument'])


def remember_verdicts(cache, motion, action, candidates, decisions):
    """Cache explicit verdicts; omitted or malformed assessments remain unknown."""
    if not isinstance(decisions, list):
        return
    allowed = {'challenges', 'answers'} if action['action'] in {'attack', 'rebut'} else {'supports'}
    counts = Counter(d.get('id') for d in decisions if isinstance(d, dict) and type(d.get('id')) is int)
    by_id = {c['id']: c for c in candidates}
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        cid = decision.get('id')
        if type(cid) is not int or counts[cid] != 1 or cid not in by_id:
            continue
        candidate = by_id[cid]
        rows = decision.get('materials')
        if not isinstance(rows, list):
            continue
        mids = Counter(d.get('id') for d in rows if isinstance(d,dict) and type(d.get('id')) is int)
        materials = {m['id']: m for m in candidate['materials']}
        for row in rows:
            if not isinstance(row, dict):
                continue
            mid = row.get('id')
            if type(mid) is not int or mids[mid] != 1 or mid not in materials:
                continue
            material = materials[mid]
            if row.get('relation') not in {'supports','challenges','answers','related','unrelated','uncertain'}:
                continue
            # An uncertain verdict must not be promoted by a local similarity score.
            usable = (row['relation'] in allowed and row.get('scope') == 'compatible'
                      and row.get('target_part') in {'claim','premise','objection'}
                      and isinstance(row.get('reason'),str) and bool(row['reason'].strip())
                      and _quote_in(row.get('target_quote'), [action['target_claim'], _text(action.get('target_argument',''))])
                      and _quote_in(row.get('material_quote'), [material['claim'],material['argument']]))
            cache[verdict_key(motion, action, candidate, material)] = bool(usable)


class LocalRehearsalRetriever:
    def __init__(self):
        self.signature = None
        self.records = []
        self.idf = {}
        self.postings = {}
        self.vector_signature = None
        self.vector_matrix = None
        self.vector_rows = []
        self.stats = {}

    def _collect(self, own, opponent, side, oppo_side, attack):
        records, seen = [], set()
        target_side = oppo_side if attack else side
        for source, trees in [('Prepared-Tree-Retrieval', own or []),
                              ('Prepared-Opponent-Tree-Retrieval', opponent or [])]:
            for tree in trees:
                for anchor in tree.get_node_by_side(target_side):
                    if not anchor.claim.strip() or not _current(anchor):
                        continue
                    candidates = anchor.children if attack else [anchor]
                    for material in candidates:
                        if material.side != side or not _current(material) or not material.claim.strip():
                            continue
                        argument = _text(material.argument).strip()
                        if not attack and not argument:
                            continue
                        parent = getattr(anchor, 'parent', None)
                        candidate = {'claim':anchor.claim, 'argument':_text(anchor.argument),
                                     'parent_claim':parent.claim if parent else ''}
                        content = {'claim':material.claim, 'argument':argument}
                        key = (source, candidate['claim'],candidate['argument'],candidate['parent_claim'],
                               content['claim'],content['argument'])
                        if key in seen:
                            continue
                        seen.add(key)
                        records.append({'key':key,'source':source,'candidate':candidate,
                                        'material':content,'node':material})
        return records

    def _index(self, records):
        # Refresh node refs even when identical text moves to a new tree instance.
        signature = tuple(r['key'] for r in records)
        self.records = records
        if signature == self.signature:
            return False
        self.signature = signature
        self.vector_signature = None
        documents = []
        for record in records:
            a = record['candidate']['claim']
            m = record['material']
            # Parent anchor retains extra weight; child explanations enable premise matches.
            documents.append(Counter(_terms(a)+_terms(a)+_terms(m['claim'])+_terms(m['argument'])))
        frequency = Counter(term for doc in documents for term in doc)
        self.idf = {t:math.log(1+len(documents)/(1+n)) for t,n in frequency.items()}
        postings = defaultdict(list)
        for i,doc in enumerate(documents):
            values = {t:(1+math.log(n))*self.idf[t] for t,n in doc.items()}
            norm = math.sqrt(sum(x*x for x in values.values())) or 1
            for term,value in values.items():
                postings[term].append((i,value/norm))
        self.postings = dict(postings)
        return True

    def _lexical(self, text):
        scores = np.zeros(len(self.records),dtype=np.float32)
        weights = {t:(1+math.log(n))*self.idf[t] for t,n in Counter(_terms(text)).items() if t in self.idf}
        norm = math.sqrt(sum(x*x for x in weights.values())) or 1
        for term,weight in weights.items():
            for idx,value in self.postings[term]:
                scores[idx] += weight/norm*value
        return scores

    def _cached_cosines(self, query, caches):
        def lookup(text):
            return next((c[text] for c in caches if text in c), None)
        vector = lookup(query)
        if vector is None:
            return {}
        q = np.asarray(vector,dtype=np.float32)
        if q.ndim != 1 or not np.isfinite(q).all() or not np.linalg.norm(q):
            return {}
        values = [lookup(r['candidate']['claim']) for r in self.records]
        signature = (len(q), tuple(id(v) for v in values))
        if signature != self.vector_signature:
            rows, vectors = [], []
            for i,v in enumerate(values):
                if v is None:
                    continue
                array = np.asarray(v,dtype=np.float32)
                if array.shape == q.shape and np.isfinite(array).all() and np.linalg.norm(array):
                    rows.append(i); vectors.append(array/np.linalg.norm(array))
            self.vector_rows = rows
            self.vector_matrix = np.asarray(vectors,dtype=np.float32) if vectors else None
            self.vector_signature = signature
        if self.vector_matrix is None:
            return {}
        # einsum avoids starting a large BLAS thread pool for this small matrix.
        scores = np.einsum('ij,j->i', self.vector_matrix, q/np.linalg.norm(q))
        return dict(zip(self.vector_rows, scores.tolist()))

    def _scores(self, target, argument, caches):
        lexical = self._lexical(target)
        if argument:
            lexical = .8*lexical+.2*self._lexical(argument)
        cosines = self._cached_cosines(target, caches)
        scores = lexical.copy()
        for i, cosine in cosines.items():
            scores[i] = .7*max(0, cosine)+.3*lexical[i]
        return scores, len(cosines)

    def _eligible(self, score, index, min_score):
        return score >= min_score

    def retrieve(self, motion, action, side, oppo_side, own, opponent, depth, *,
                 embedding_caches=(), verdicts=None, candidate_k=12, max_results=3, min_score=.25,
                 max_per_anchor=None):
        kind = action['action']
        if kind not in {'propose','reinforce','attack','rebut'}:
            raise ValueError(f'Invalid action: {kind}')
        if any(type(n) is not int or n < 1 for n in (candidate_k,max_results)) or not 0 <= min_score <= 1:
            raise ValueError('Invalid local retrieval limits')
        if max_per_anchor is not None and (type(max_per_anchor) is not int or max_per_anchor < 1):
            raise ValueError('max_per_anchor must be a positive integer or None')
        target = action['target_claim']
        records = self._collect(own,opponent,side,oppo_side,kind in {'attack','rebut'})
        rebuilt = self._index(records)
        argument = _text(action.get('target_argument',''))
        scores, vector_matches = self._scores(target, argument, embedding_caches)
        candidates = []
        for i,record in enumerate(records):
            key = verdict_key(motion,action,record['candidate'],record['material'])
            verified = (verdicts or {}).get(key)
            if verified is False:
                continue
            anchor = record['candidate']['claim']
            exact = _normalize(anchor) == _normalize(target)
            if verified is not True and polarity_conflict(target,anchor):
                continue
            score = float(scores[i])
            if exact:
                score = max(score,1.0)
            if verified is True or exact or self._eligible(score, i, min_score):
                candidates.append((verified is True,exact,score,i))
        candidates.sort(key=lambda x:(-x[0],-x[1],-x[2],x[3]))
        info, matches, seen = [], [], set()
        verified_count = 0
        anchor_counts = Counter()
        diversity_skipped = 0
        for verified,_,score,i in candidates[:candidate_k]:
            record = records[i]; material = record['material']
            text = '\n'.join(dict.fromkeys(s for s in (material['claim'],material['argument']) if s))
            if text in seen:
                continue
            anchor_key = _normalize(record['candidate']['claim'])
            # A grounded verdict outranks a diversity heuristic. Unverified
            # siblings must share their parent's allowance across both pools.
            if not verified and max_per_anchor is not None and anchor_counts[anchor_key] >= max_per_anchor:
                diversity_skipped += 1
                continue
            seen.add(text)
            anchor_counts[anchor_key] += 1
            text += f" (Strength: {record['node'].get_strength(max_depth=depth):.1f})\n\t"
            info.append(text)
            matches.append([record['source'],kind,target,record['candidate']['claim'],score,text])
            verified_count += verified
            if len(matches) >= max_results:
                break
        self.stats = dict(mode='local',index_rebuilt=rebuilt,eligible_materials=len(records),
                          cached_vector_matches=vector_matches,verified_materials=verified_count,
                          heuristic_materials=len(matches)-verified_count,network_calls=0,
                          max_per_anchor=max_per_anchor,distinct_anchors=len(anchor_counts),
                          diversity_skipped=diversity_skipped)
        return info,matches
