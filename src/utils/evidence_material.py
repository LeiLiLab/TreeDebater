"""Owned selection annotations and local, verbatim writing material. No API calls."""
import copy
import json
import re


class EvidenceSelection(list):
    """List-compatible native result; source dictionaries remain unchanged."""
    def __init__(self, evidence=(), analysis=None):
        super().__init__(copy.deepcopy(list(evidence)))
        ids = {str(e['id']) for e in self}
        self.analysis = {str(k): v for k, v in (analysis or {}).items()
                         if str(k) in ids and isinstance(v, str)}


_STOP = set('the and for that this with from have has are was were will would could should '
            'our their they your evidence source research study claim argument point statement '
            'revision guidance changes support supports supplied'.split())


def _terms(text):
    return set(re.findall(r'[^\W_]{3,}', text.casefold())) - _STOP


def writing_evidence(evidence, statement, feedback='', *, n_words=500):
    """Route excerpts to draft paragraphs; budget characters, not evidence count.

    Reasons are untrusted selection annotations, never source text. Excerpts are
    contiguous verbatim sentence windows (including neighbors for qualifications).
    Full documents remain in the caller's evidence pool.
    """
    claims = [p.strip() for p in re.split(r'\n\s*\n', statement) if p.strip()] or [feedback]
    terms = [_terms(p) for p in claims]
    feedback_terms = _terms(feedback)
    reasons = getattr(evidence, 'analysis', {})
    ranked = []
    for order, e in enumerate(evidence):
        content = str(e.get('content') or '')
        if not content.strip():
            continue
        reason = reasons.get(str(e.get('id')), '')
        source_terms = _terms(content + ' ' + str(e.get('title', '')))
        hint_terms = _terms(reason)
        scores = [len(t & source_terms) + .25 * len(t & hint_terms) for t in terms]
        index = max(range(len(claims)), key=lambda i: scores[i])
        relevance = scores[index] + .25 * len(feedback_terms & source_terms)
        query = terms[index] | feedback_terms
        spans = list(re.finditer(r'\S.*?(?:[.!?]+(?=\s|$)|\n|$)', content, re.DOTALL))
        if not spans:
            continue
        best = max(range(len(spans)), key=lambda i: len(_terms(spans[i].group()) & query))
        lo, hi = max(0, best-1), min(len(spans)-1, best+1)
        excerpt = content[spans[lo].start():spans[hi].end()].strip()
        # Keep full sentences; do not manufacture ellipses within a finding.
        card = dict(id=e.get('id'), supports=claims[index] if scores[index] else '',
                    title=e.get('title', ''), source=e.get('source', ''), content=excerpt)
        for key in ('url', 'date', 'year', 'author'):
            if e.get(key):
                card[key] = e[key]
        if reason:
            card['selection_reason'] = reason
        key = e.get('url') or (e.get('title'), e.get('source'), e.get('date'), e.get('year'))
        if not e.get('url') and not e.get('title'):
            key = ('id', e.get('id'))
        ranked.append((relevance, order, key, card))
    # Explicit native selections with no lexical overlap remain usable; mark
    # their association unknown instead of inventing a supporting relationship.
    has_matches = any(score > 0 for score, *_ in ranked)
    budget = max(1600, int(n_words * 16))
    result, seen = [], {}
    # Give each current passage a chance before filling the budget with several
    # sources for the same passage. There is no fixed number of evidence items.
    groups = {}
    for row in sorted(ranked, key=lambda row: (-row[0], row[1])):
        groups.setdefault(row[3]['supports'], []).append(row)
    ordered = [group[depth] for depth in range(max((len(g) for g in groups.values()), default=0))
               for group in groups.values() if depth < len(group)]
    for score, _, key, card in ordered:
        if has_matches and score <= 0:
            continue
        if key in seen:
            existing = seen[key]
            merged = copy.deepcopy(existing)
            if card['id'] != existing['id']:
                merged.setdefault('also_selected_ids', []).append(card['id'])
            if card['content'] not in existing['content']:
                merged['content'] += '\n\n' + card['content']
            if card['supports'] and card['supports'] not in existing['supports']:
                merged['supports'] += '\n\n' + card['supports']
            reason = card.get('selection_reason')
            if reason and reason not in existing.get('selection_reason', ''):
                merged['selection_reason'] = existing.get('selection_reason', '') + '\n' + reason
            extra = len(json.dumps(merged, ensure_ascii=False)) - len(json.dumps(existing, ensure_ascii=False))
            if extra <= budget:
                existing.update(merged)
                budget -= extra
            continue
        size = len(json.dumps(card, ensure_ascii=False))
        if size > budget:
            continue
        result.append(card)
        seen[key] = card
        budget -= size
    return result


EVIDENCE_USE_INSTRUCTION = (
    '\nEvidence material is grouped by the draft passage in supports; this is a local relevance hint, '
    'not proof. selection_reason is the selector\'s interpretation, not a source or established fact. '
    'Use the verbatim content excerpt to establish what the source actually says, retain qualifications, '
    'and explain which specific reasoning step it supports. You may add sourced factual support and '
    'revise the unpublished argument while preserving its stance, claim ownership and valid factual meaning. '
    'A source need not prove the entire argument. Do not substitute an analogy or a vague reference to '
    'research for available relevant findings. Attribute sources verbally using only supplied metadata; '
    'do not invent dates, credentials or findings. Missing metadata is unknown.\n'
)
