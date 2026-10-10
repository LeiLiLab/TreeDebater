"""Bounded, speculative use of TreeDebater's native evidence selector.

Local term matching routes new needs to unused candidates; it is not a semantic
approval of evidence. Final authors/reviewers retain responsibility for claims.
No worker writes player state or marks evidence used.
"""
import copy
from dataclasses import dataclass, field
import json
import re
import threading
import time

from utils.evidence_material import EvidenceSelection

_STOP = set('the a an and or of to in is are was were be been being that this these those '
            'it its for from with as on at by we our you your they their have has had '
            'will would should could can may must do does did not but than then '
            'evidence source sources study studies data provide add need needs more '
            'support explain clarify revision guidance changes selected already'.split())


def terms(text):
    return set(re.findall(r'[^\W_]{3,}', text.casefold())) - _STOP


def query(data):
    return '\n'.join(('Opponent speech to respond to (not our position): ' + data.get('heard_transcript', ''),
        'Our response framework: ' + json.dumps(data.get('framework') or {}, ensure_ascii=False),
        'Our body plan: ' + json.dumps(data.get('body_plan') or [], ensure_ascii=False)))


@dataclass
class EvidenceState:
    selected: list = field(default_factory=list)
    covered: set = field(default_factory=set)
    documents: dict = field(default_factory=dict)
    calls: int = 0
    initialized: bool = False
    events: list = field(default_factory=list)
    bindings: dict = field(default_factory=dict)
    analysis: dict = field(default_factory=dict)

    def select(self, selector, statement, guidance, candidates, *, stage,
               candidate_limit=20, initial_query=None, reserve=None):
        """Keep valid choices; native selection only sees unselected candidates.

        The first selection uses the full native pool. Subsequent selections use
        local lexical matches for novel requirements, with a bounded shortlist.
        Counts include reserved in-flight selections, even if later discarded.
        """
        pool = {e['id']: copy.deepcopy(e) for e in candidates}
        valid = [e for e in self.selected if pool.get(e['id']) == e]
        invalidated = len(valid) != len(self.selected)
        self.selected = valid
        self.analysis = {str(e['id']): self.analysis[str(e['id'])] for e in valid if str(e['id']) in self.analysis}
        if stage == 'closing':
            return []
        text = initial_query if initial_query is not None else statement + '\n' + guidance
        wanted = terms(text)
        if initial_query is not None:
            # Drop only choices whose original topic has disappeared from a
            # corrected input/plan. The final writer still checks applicability.
            valid = [e for e in valid if not self.bindings.get(e['id'])
                     or self.bindings[e['id']] & wanted]
            invalidated |= len(valid) != len(self.selected)
            self.selected = valid
        # An updated document must be eligible again, even for an old need.
        changed = {key for key, e in pool.items()
                   if key in self.documents and self.documents[key][0] != e}
        for key, e in pool.items():
            if key not in self.documents or key in changed:
                projection = {k: v for k, v in e.items() if k != 'raw_content'}
                self.documents[key] = (e, terms(json.dumps(projection, ensure_ascii=False)))
        self.documents = {key: value for key, value in self.documents.items() if key in pool}
        selected_ids = {e['id'] for e in valid}
        remaining = [e for key, e in pool.items() if key not in selected_ids]
        novel = wanted - self.covered
        if initial_query is None:
            # A final reviewer may identify an unmet need already mentioned in
            # the input. Seeing the topic earlier does not mean we selected
            # supporting evidence for it.
            novel |= terms(guidance)
        if invalidated or changed:
            novel = wanted
        if self.initialized:
            # Existing selected sources can cover new wording without another
            # selection call. Never present them again as selectable candidates.
            covered_by_sources = set().union(*(self.documents[e['id']][1] for e in valid)) if valid else set()
            novel -= covered_by_sources
            scored = [(len(novel & self.documents[e['id']][1]), i, e)
                      for i, e in enumerate(remaining)]
            remaining = [e for score, _, e in sorted(scored, key=lambda x: (-x[0], x[1]))
                         if score > 0][:candidate_limit]
        event = dict(start=time.perf_counter(), candidate_ids=[e['id'] for e in remaining],
                     reused_ids=[e['id'] for e in valid], new_ids=[])
        if not remaining:
            event['status'] = 'covered_or_no_matching_candidates'
            self.events.append(event)
            return EvidenceSelection(valid, self.analysis)
        if reserve is not None and not reserve():
            event['status'] = 'frozen'
            self.events.append(event)
            return EvidenceSelection(valid, self.analysis)
        summaries = [dict(id=e['id'], title=str(e.get('title', ''))[:120],
                          coverage=str(e.get('content', ''))[:320]) for e in valid]
        guidance += ('\nAlready selected evidence is retained; select only additions for uncovered needs. '
                     'Do not select these IDs again.\n' + json.dumps(summaries, ensure_ascii=False))
        self.calls += 1
        # Reserve before dispatch. Exceptions leave existing choices intact and
        # do not mark failed requirements as covered.
        try:
            result = selector(statement, guidance, copy.deepcopy(remaining), stage=stage)
            reasons = getattr(result, 'analysis', {})
            offered = {e['id']: e for e in remaining}
            additions = []
            for e in result:
                key = e['id']
                if key in offered and key not in selected_ids:
                    additions.append(copy.deepcopy(offered[key]))
                    selected_ids.add(key)
                    self.bindings[key] = wanted & self.documents[key][1]
                    if str(key) in reasons:
                        self.analysis[str(key)] = reasons[str(key)]
            self.selected.extend(additions)
            self.covered.update(wanted)
            self.initialized = True
            event.update(status='selected', new_ids=[e['id'] for e in additions])
            return EvidenceSelection(self.selected, self.analysis)
        except Exception as exc:
            event.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            event['end'] = time.perf_counter()
            self.events.append(event)


class ListeningEvidence:
    """Single coalescing worker; freeze never waits for an obsolete API call."""
    def __init__(self, selector, config):
        self.selector, self.config = selector, config
        self._lock = threading.Lock()
        self._state = EvidenceState()
        self._pending = self._thread = None
        self._stopped = False
        self._requests = 0
        self._available = {}
        self.scope = None

    def reserve(self, *, endpoint=False):
        with self._lock:
            if self._stopped and not endpoint:
                return False
            self._requests += 1
            return True

    def snapshot(self):
        with self._lock:
            value = copy.deepcopy(self._state)
            value.calls = self._requests
            value.selected = EvidenceSelection([e for e in value.selected if self._available.get(e['id']) == e], value.analysis)
            return value

    def offer(self, data, candidates):
        if data['stage'] == 'closing':
            return
        with self._lock:
            if self._stopped:
                return
            scope = (data['stage'], data.get('turn'))
            if self.scope is not None and scope != self.scope:
                raise ValueError('Evidence preparation belongs to one stage and listening turn')
            self.scope = scope
            self._available = {e['id']: copy.deepcopy(e) for e in candidates}
            self._pending = (copy.deepcopy(data), copy.deepcopy(candidates))
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name='listening-evidence', daemon=True)
                self._thread.start()

    def _run(self):
        while True:
            with self._lock:
                if self._stopped or self._pending is None:
                    self._thread = None
                    return
                (data, candidates), self._pending = self._pending, None
                state = copy.deepcopy(self._state)
            try:
                state.select(self.selector, query(data), 'Prepare evidence for the current response plan.',
                    candidates, stage=data['stage'],
                    candidate_limit=self.config.listening_evidence_candidates, initial_query=query(data),
                    reserve=self.reserve)
            except Exception:
                pass  # Native endpoint fallback remains available; failure is traced.
            with self._lock:
                if not self._stopped:
                    self._state = state

    def freeze(self):
        with self._lock:
            self._stopped = True
            self._pending = None
        return self.snapshot()

    def close(self):
        self.freeze()
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join()
