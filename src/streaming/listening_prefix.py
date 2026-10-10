"""Prepare a stable overview and provisional body while listening.

Speculative workers receive value snapshots and never mutate the debater. The
first paragraph and each replacement receive a model review before TTS. Reviews
are cached by exact text and review context. The existing whole-speech feedback reconciles the remaining body once complete input is available.
Framework readiness comes from the existing incremental planner. Details update
body drafts without rewriting the overview. Final whole-speech work reconciles
completed preparation with all input, keeping the published first paragraph immutable.
"""
import copy
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import threading
import time

from .constraint_review import current_checklist, opponent_sources, supplied_evidence
from .body_task import BodyTask
from .body_plan import resolve_body_plan
from .clash_records import exchange_records
from .config import SpeechBudgets, from_mapping
from .flat_speaking import SegmentRejected, _json_object
from .full_speech import remaining_text
from .overview_planning import parse_overview, framework_stamp
from .overview_review import review_payload, framework_stance_conflict, review, review_stamp
from utils.prompts.speech_generation import build_draft
from utils.speech_context import authoring_history
from utils.prompts.authoring import DEFAULT_DEBATER_SYSTEM, authoring_options, debater_system
from utils import speech_length
from utils.time_estimator import LengthEstimator
from utils.tool import logger


def next_stage(side, stage, first_side='for'):
    """The next speaker follows the configured order, including reversed debates."""
    stages = ('opening', 'rebuttal', 'closing')
    if stage not in stages:
        return None
    index = stages.index(stage) + (side != first_side)
    return stages[index] if index < len(stages) else None


def material(player, stage, history=None, *, include_rehearsal=True):
    context = player._planning_context()
    conditions = current_checklist(player)
    history = history or []
    transcript = history[-1]['content'] if history else ' '.join(player.planner.chunks)
    our_tree, opponent_tree = player._generation_tree_context()
    budgets = from_mapping(SpeechBudgets, getattr(player, 'speech_budgets', None))
    data = dict(motion=player.motion, our_side=player.side, stage=stage,
        max_time=getattr(budgets, stage),
        our_tree=our_tree, opponent_tree=opponent_tree,
        our_definition=getattr(player, 'definition', ''),
        our_main_claims=getattr(player, 'main_claims_content', []),
        our_previous_speeches=[m['content'] for m in player.conversation if m['role'] == 'assistant'],
        private_claim_options=[group[0].get('claim', '') for group in getattr(player, 'claim_pool', [])[:8]
                               if isinstance(group, list) and group and isinstance(group[0], dict)],
        private_claim_candidates=context.get('private_claim_candidates', []),
        debate_history=history or context['prior_debate'],
        framework=player.planner.overview or player.planner.state.get('overview'),
        current_targets=context['tree_targets'], condition_candidates=conditions,
        selected_target_ids=[c['node_id'] for c in player.planner.state.get('claims', [])],
        clash_records=context.get('clash_records', []),
        body_plan=[] if context.get('listening_source_selection') else player.planner.state.get('body_plan', []),
        current_plan=player.planner.plan, opponent_sources=opponent_sources(player, history),
        supplied_evidence=supplied_evidence(player), final_transcript=transcript,
        feedback_context=player._feedback_context(stage),
        heard_transcript=' '.join(player.planner.chunks), turn=player.planner.turn)
    if include_rehearsal and getattr(player, 'use_rehearsal_tree', False):
        data['prepared_rehearsal_materials'] = player._listening_rehearsal_materials(data)
    return copy.deepcopy(data)


def source_stamp(data, *, include_rehearsal=True):
    # Private plan formatting must not invalidate unchanged input.
    sources = {k: data[k] for k in ('current_targets', 'opponent_sources',
                                   'heard_transcript', 'final_transcript', 'turn')}
    sources['conditions'] = [{k: v for k, v in c.items() if k != 'planned_target'}
                             for c in data['condition_candidates']]
    sources['clash_records'] = data.get('clash_records', [])
    sources['authoring_material'] = {key: data.get(key) for key in (
        'our_definition', 'our_main_claims', 'our_tree', 'opponent_tree',
        'supplied_evidence', 'prepared_rehearsal_materials', 'max_time')}
    if not include_rehearsal:
        sources['authoring_material'].pop('prepared_rehearsal_materials')
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def opponent_input_stamp(data):
    """Distinguish newly heard content from private planning/tree changes."""
    return hashlib.sha256(json.dumps(
        [data['opponent_sources'], data['heard_transcript']], ensure_ascii=False).encode()).hexdigest()


def prefix_limits(config, max_time=None):
    seconds = min(config.first_chunk_seconds, max_time) if max_time else config.first_chunk_seconds
    target = max(config.listening_prefix_min_words,
                 config.listening_prefix_target_words or speech_length.draft_word_budget(seconds, rounding='floor'))
    return target, max(target, math.ceil(target * 1.5))


def preparation_time(data):
    return data.get('max_time', getattr(SpeechBudgets(), data['stage']))


def body_word_budget(data, prefix):
    """Reserve the actual overview length within the whole stage's draft budget."""
    total = speech_length.draft_word_budget(preparation_time(data))
    return max(1, math.floor(total - speech_length.draft_word_count(prefix)))


def _prefix_format_errors(candidate, config, max_time=None):
    text = candidate.get('text')
    if not isinstance(text, str) or not text.strip():
        return ['text must be a nonempty first-paragraph string']
    errors = []
    if text != text.strip() or '\n' in text:
        errors.append('text must be one paragraph without surrounding whitespace or newlines')
    if text[-1:] not in '.?!':
        errors.append('text must end with a complete sentence (. ? or !)')
    if any(mark in text for mark in ('**', '```', '**Reference**')):
        errors.append('text must be spoken prose without Markdown headings or code fences')
    if len(text) > config.max_chunk_chars:
        errors.append(f'text exceeds {config.max_chunk_chars} characters')
    if len(text.split()) < config.listening_prefix_min_words:
        errors.append(f'text must contain at least {config.listening_prefix_min_words} words; develop the existing point without filler')
    maximum = prefix_limits(config, max_time)[1]
    if speech_length.draft_word_count(text) > maximum:
        errors.append(f'text exceeds {maximum} word equivalents')
    return errors


def _valid_candidate(candidate, config, max_time=None):
    return not _prefix_format_errors(candidate, config, max_time)


class PrefixFormatRejected(SegmentRejected):
    """No structurally usable candidate; does not establish a framing conflict."""


class InitialPrefixReview:
    """Version-bound reviews, with up to three format attempts per review."""
    def __init__(self):
        self._lock = threading.Lock()
        self._result = None
        self._text = None
        self._results = {}
        self._invalidated_texts = set()

    def check(self, candidate, data, helper, *, endpoint=False):
        key = review_stamp(candidate, data)
        with self._lock:
            first = key not in self._results
            if first:
                self._results[key] = (candidate['text'], Future())
                if self._result is None:
                    self._text, self._result = self._results[key]
            result = self._results[key][1]
        if not first:
            return copy.deepcopy(result.result())
        try:
            # Retry a malformed reviewer response on the unchanged candidate.
            # This allowance is independent of prepare()'s draft repair loop;
            # valid semantic rejections are returned immediately by review().
            verdict = review(candidate, data, helper, endpoint=endpoint, max_attempts=3)
        except BaseException as exc:
            result.set_exception(exc)
            raise
        with self._lock:
            if verdict['accepted']:
                self._invalidated_texts.discard(candidate['text'])
            elif verdict.get('review_format_valid'):
                self._invalidated_texts.add(candidate['text'])
        result.set_result(verdict)
        return copy.deepcopy(verdict)

    def invalidated(self, candidate):
        with self._lock:
            return candidate['text'] in self._invalidated_texts

    def snapshot(self):
        with self._lock:
            result, text = self._result, self._text
            versions = list(self._results.items())
        state = dict(attempted=result is not None, checked_text=text,
                     completed=result is not None and result.done())
        if result is not None and result.done():
            try:
                state['verdict'] = result.result()
            except BaseException as exc:
                state['error'] = f'{type(exc).__name__}: {exc}'
        state['versions'] = []
        for key, (checked_text, future) in versions:
            row = dict(review_stamp=key, checked_text=checked_text, completed=future.done())
            if future.done():
                try:
                    row['verdict'] = copy.deepcopy(future.result())
                except BaseException as exc:
                    row['error'] = f'{type(exc).__name__}: {exc}'
            state['versions'].append(row)
        return state


def prepare(data, helper, config, *, endpoint=False, max_time=None,
            system_prompt=DEFAULT_DEBATER_SYSTEM, writing_options=None, author=None, prompt_builder=None,
            initial_review=None):
    initial_review = initial_review if initial_review is not None else InitialPrefixReview()
    from utils.speech_context import authoring_sources
    max_time = preparation_time(data) if max_time is None else max_time
    data = dict(data, endpoint=endpoint, max_time=max_time)
    if config.listening_prefix_overlap_final_update:
        data = dict(data, prefix_handoff=True)
    target, maximum = prefix_limits(config, max_time)
    total_words = speech_length.draft_word_budget(max_time)
    context = authoring_sources(data)
    if data.get('previous_speech'):
        context['previous_speech'] = data['previous_speech']
        context['prefix_review_issues'] = data.get('prefix_review_issues', [])
    framework_schema = ('{"ready":true,"position":"our assigned side",'
        '"core_dispute":"central issue","response_axes":[],"prefix_action":"keep",'
        '"reason":"brief explanation"}')
    schema = ('{"text":"first spoken paragraph","draft":"all remaining spoken paragraphs",'
              '"framework":' + framework_schema + '}')
    prompt = ('LISTENING PREFIX DRAFT: Write the opening and body together as one complete speech. '
        + build_draft(data, '', total_words, prompt_builder=prompt_builder,
            previous={'draft': data['previous_speech']} if data.get('previous_speech') else None,
            include_sources=False, output_contract=False)
        + f'\nThe first paragraph should be about {target} words, maximum {maximum}; '
        'the remaining paragraphs develop the same argument under the total word budget. '
        'Return the first paragraph in text and every subsequent paragraph in draft, '
        'without repeating the first paragraph. Both fields belong to ONE speech by the '
        'assigned speaker. Return ONLY JSON with this schema: ' + schema
        + '\n' + json.dumps(dict(context=context, endpoint=endpoint), ensure_ascii=False))
    writing_options = dict(writing_options or {})
    history_messages = authoring_history(data)
    author = author or helper
    raw = author(prompt=prompt, sys=system_prompt, history_messages=history_messages,
                 json_mode=True, **writing_options)[0]
    audits = []
    for attempt in range(2):
        parse_issue = None
        try:
            candidate = _json_object(raw)
        except SegmentRejected as exc:
            candidate = {}
            parse_issue = str(exc) + '. Regenerate one complete JSON object; no reasoning or repeated commentary.'
        # Target selection belongs to the planner. Legacy writer metadata is
        # neither a publication condition nor an authority to choose tree nodes.
        candidate.pop('target_ids', None)
        speech_errors = _prefix_format_errors(candidate, config, max_time)
        if not isinstance(candidate.get('draft'), str) or not candidate['draft'].strip():
            speech_errors.append('draft must contain the nonempty remaining spoken paragraphs')
        if not speech_errors:
            try:
                candidate['draft'] = remaining_text(candidate['draft'], candidate['text'])
                if not candidate['draft']:
                    speech_errors.append('draft must contain speech beyond the first paragraph')
            except ValueError as exc:
                speech_errors.append('draft: ' + str(exc))
        # This explanation is diagnostic metadata, not a speech or stance claim.
        # Preserve the authored speech when only its explanation was omitted.
        normalizations = []
        framework = candidate.get('framework')
        if isinstance(framework, dict) and 'reason' not in framework:
            framework['reason'] = 'Author did not provide a framework explanation.'
            normalizations.append(dict(field='framework.reason', action='record_missing_explanation'))
        framework_errors = []
        try:
            if not parse_overview(framework)['ready']:
                framework_errors.append('framework.ready must be true for a publishable speech')
        except ValueError as exc:
            framework_errors.append(str(exc))
        valid = not (parse_issue or speech_errors or framework_errors)
        conflict = framework_stance_conflict(review_payload(candidate, data)) if valid else None
        valid = valid and conflict is None
        audit = dict(accepted=valid, kind='local_format_check', semantic_review=False,
                     issues=([parse_issue] if parse_issue else speech_errors + framework_errors)
                            + ([conflict['reason']] if conflict else []))
        if valid and config.listening_prefix_review_enabled:
            audit = dict(review(candidate, data, helper, endpoint=endpoint),
                         kind='prefix_review', semantic_review=True)
            valid = audit['accepted']
        elif valid and config.listening_prefix_initial_review_enabled:
            verdict = initial_review.check(candidate, data, helper, endpoint=endpoint)
            if verdict is not None:
                audit = dict(verdict, kind='initial_prefix_review', semantic_review=True)
                valid = audit['accepted']
        if normalizations:
            audit['normalizations'] = normalizations
        audits.append(audit)
        if valid:
            body = dict(draft=candidate.pop('draft'), feedback=None,
                feedback_status='deferred_to_final_input', needs_fit=False,
                body_plan=[], prefix_text=candidate['text'], clash_records=[],
                framework=copy.deepcopy(candidate['framework']), source_stamp=source_stamp(data),
                heard_transcript=data['heard_transcript'], ready_monotonic=time.perf_counter(),
                stage=data['stage'], turn=data['turn'], source='shared_speech_draft')
            return dict(candidate, body_preparation=body, stage=data['stage'], turn=data['turn'],
                source_stamp=source_stamp(data), audits=audits, review_stamp=review_stamp(candidate, data),
                opponent_input_stamp=opponent_input_stamp(data),
                handoff_prechecked=bool(data.get('prefix_handoff')),
                reviewed_context=copy.deepcopy(data) if data.get('prefix_handoff') else None)
        if not attempt:
            if not parse_issue and not speech_errors and framework_errors:
                raw = helper(prompt='LISTENING FRAMEWORK REPAIR: Correct only the framework metadata '
                    'for this unchanged speech. Do not rewrite text or draft. Resolve each listed field error. '
                    'The speech must still pass its separate stance and publication review. '
                    'Return ONLY JSON with this schema: {"framework":' + framework_schema + '}'
                    + '\n' + json.dumps(dict(context=context, text=candidate['text'], draft=candidate['draft'],
                        framework=candidate.get('framework'), issues=framework_errors), ensure_ascii=False),
                    sys=system_prompt, history_messages=history_messages, json_mode=True, **writing_options)[0]
                try:
                    patched = _json_object(raw)
                except SegmentRejected:
                    pass  # The final attempt records the malformed repair without another call.
                else:
                    raw = json.dumps(dict(candidate, framework=patched.get('framework')), ensure_ascii=False)
                continue
            raw = author(prompt='LISTENING PREFIX REPAIR: Repair the complete unpublished speech together. '
                + build_draft(data, '', total_words, prompt_builder=prompt_builder,
                              include_sources=False, output_contract=False)
                + f'\nKeep text to one first paragraph: target {target}, hard maximum {maximum} word equivalents. '
                'Return the remaining coherent speech in draft. Do not change just the side label. '
                'Return ONLY the repaired output object with this exact schema: ' + schema
                + '\n' + json.dumps(dict(context=context, draft=candidate,
                    issues=audit['issues'] + audit.get('format_errors', []),
                    target_words=target, maximum_word_equivalents=maximum), ensure_ascii=False),
                sys=system_prompt, history_messages=history_messages, json_mode=True, **writing_options)[0]
    if audits[-1]['semantic_review']:
        raise SegmentRejected('Opening failed publication review after one repair')
    raise PrefixFormatRejected('Opening failed local format checks after one repair')


class PreparationStopped(RuntimeError):
    """No further speculative request fits the lifecycle/call budget."""


def synthesize_prefix(text, config):
    """Prepare matching audio bytes without publishing them."""
    import tts_streaming
    client = tts_streaming.OpenAI()
    try:
        return tts_streaming._query_time_profiled(client, text, voice=config.voice, model=config.model)
    finally:
        client.close()


class PrefixPreparation:
    """One initial speech draft, then coalescing body updates on source snapshots."""
    def __init__(self, turn, helper, config, audio_preparer=None, *, system_prompt=DEFAULT_DEBATER_SYSTEM,
                 writing_options=None, author=None, prompt_builder=None):
        self.turn, self.helper, self.config = turn, helper, copy.deepcopy(config)
        self.author, self.prompt_builder = author or helper, prompt_builder
        self.system_prompt = system_prompt
        self.writing_options = dict(writing_options or {})
        self._lock = threading.Lock()
        self._pending = self._latest = self._thread = None
        self._body_pending = self._body_latest = self._body_thread = None
        self._body_future = self._body_binding = None
        self._audio_preparer, self._audio_latest = audio_preparer, None
        self._audio_attempted = set()
        self._audio_futures = {}
        self._audio_threads = []
        self._candidates = []
        self._context_latest = None
        self._scope = None
        self._closed = False
        self._stopped = False
        self._calls = self._rewrites = 0
        self._has_prepared_prefix = False
        self.initial_review = InitialPrefixReview()
        self._failed_frameworks = set()
        # Retry a format-only failure once after new speech, within the shared call cap.
        self._format_failures = {}
        self._last_body_stamp = None
        self._body_started_input = None
        self.events = []
        self.evidence = None

    def _complete(self, *, body=False, authoring=False, **kwargs):
        with self._lock:
            # Leave capacity for a meaningful late overview correction. Ordinary
            # body updates cannot consume this reserve. Endpoint work is separate.
            cap = max(0, self.config.listening_prefix_max_calls - 8) if body else self.config.listening_prefix_max_calls
            if self._stopped or self._calls >= cap:
                raise PreparationStopped('Speculative call budget exhausted or turn frozen')
            self._calls += 1
        return (self.author if authoring else self.helper)(**kwargs)

    def _author_complete(self, **kwargs):
        return self._complete(authoring=True, **kwargs)

    def peek(self):
        with self._lock:
            return copy.deepcopy(self._latest)

    def prepared_audio(self, text, config):
        with self._lock:
            value = self._audio_latest
            if value and (value['text'], value['voice'], value['model']) == (text, config.voice, config.model):
                return copy.deepcopy(value)
        return None

    @staticmethod
    def _binding(candidate):
        return (candidate['stage'], candidate['turn'], candidate['text'],
                json.dumps(candidate['framework'], sort_keys=True))

    def pending_body(self, candidate):
        """Transfer only an immutable result channel bound to this exact opening."""
        with self._lock:
            if candidate is not None and self._body_binding == self._binding(candidate):
                return self._body_future
        return None

    def body_context(self, stage, turn):
        """Latest completed observer snapshot; never wait for pending analysis."""
        with self._lock:
            if self._scope == (stage, turn):
                return copy.deepcopy(self._context_latest)
        return None

    def offer(self, data):
        with self._lock:
            scope = (data['stage'], data['turn'])
            if self._closed or data['turn'] != self.turn or (self._scope is not None and scope != self._scope):
                return
            self._scope = scope
            # Freeze stops speculative requests, not completed observer snapshots.
            self._context_latest = copy.deepcopy(data)
            if self._stopped or self._calls >= self.config.listening_prefix_max_calls:
                return
            self._pending = copy.deepcopy(data)
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name='listening-overview', daemon=True)
                self._thread.start()

    def _run(self):
        while True:
            with self._lock:
                if self._stopped or self._pending is None:
                    self._thread = None
                    return
                data, self._pending = self._pending, None
                candidate = copy.deepcopy(self._latest)
            if self.config.listening_prefix_overlap_final_update:
                data = dict(data, prefix_handoff=True)
            event = dict(kind='overview', start=time.perf_counter(), source_stamp=source_stamp(data))
            try:
                framework = parse_overview(data.get('framework'))
                if not framework['ready']:
                    event['status'] = 'waiting_for_framework'
                else:
                    stamp = framework_stamp(framework)
                    needs_draft = candidate is None
                    if candidate is not None and framework['prefix_action'] == 'replace':
                        # A new central opponent point can justify a bounded
                        # update even when the earlier broad framing was valid.
                        # The existing planner already nominates a changed opening.
                        # Reuse that decision rather than asking a second model to judge it.
                        needs_draft = (stamp != framework_stamp(candidate['framework'])
                            and candidate.get('opponent_input_stamp', opponent_input_stamp(data))
                            != opponent_input_stamp(data))
                        event['grounded_update'] = needs_draft
                    if needs_draft:
                        format_failure = self._format_failures.get(stamp)
                        if stamp in self._failed_frameworks:
                            event['status'] = 'same_failed_framework'
                        elif format_failure and format_failure['attempts'] >= 2:
                            event['status'] = 'format_retry_exhausted'
                        elif format_failure and format_failure['heard_transcript'] == data['heard_transcript']:
                            event['status'] = 'same_failed_format_input'
                        elif self._has_prepared_prefix and self._rewrites >= self.config.listening_prefix_max_rewrites:
                            event['status'] = 'rewrite_budget_wait'
                        else:
                            if self._has_prepared_prefix:
                                self._rewrites += 1
                            try:
                                draft_data = (dict(data, previous_speech=candidate['text'] + '\n\n'
                                    + candidate.get('body_preparation', {}).get('draft', ''))
                                              if candidate is not None else data)
                                prepared = prepare(draft_data, self._complete, self.config,
                                                   system_prompt=self.system_prompt,
                                                   writing_options=self.writing_options,
                                                   author=self._author_complete, prompt_builder=self.prompt_builder,
                                                   initial_review=self.initial_review)
                            except PrefixFormatRejected:
                                self._format_failures[stamp] = dict(heard_transcript=data['heard_transcript'],
                                    attempts=(format_failure['attempts'] if format_failure else 0) + 1)
                                raise
                            except Exception:
                                self._failed_frameworks.add(stamp)
                                raise
                            prepared['ready_monotonic'] = time.perf_counter()
                            with self._lock:
                                if not self._stopped:
                                    self._latest = prepared
                                    self._candidates.append(copy.deepcopy(prepared))
                                    self._has_prepared_prefix = True
                                    candidate = copy.deepcopy(prepared)
                            event['status'] = 'reviewed'
                    else:
                        event['status'] = 'kept'
            except (ValueError, TypeError):
                event['status'] = 'waiting_for_framework'
            except Exception as exc:
                event.update(status='rejected', error=f'{type(exc).__name__}: {exc}')
            with self._lock:
                candidate = copy.deepcopy(self._latest)
            if candidate is not None:
                self._start_audio(candidate)
                self._offer_body(data, candidate)
            event['end'] = time.perf_counter()
            with self._lock:
                self.events.append(event)

    def _offer_body(self, data, candidate):
        key = (source_stamp(data), candidate['text'])
        words = tuple(data['heard_transcript'].split())
        with self._lock:
            if self._stopped or key == self._last_body_stamp:
                return
            self._last_body_stamp = key
            previous = self._body_started_input
            if (previous is not None and previous[1] == candidate['text']
                    and (words != previous[0] or key[0] == previous[2])):
                old_words = previous[0]
                # Count actual newly heard words, independent of draft phoneme
                # budgets. A corrected transcript or changed opening invalidates
                # the baseline and must not wait for another hundred words.
                if words[:len(old_words)] == old_words:
                    added = LengthEstimator.count_words(' '.join(words[len(old_words):]))
                    if (self._body_pending is None
                            and added < self.config.listening_body_update_words):
                        now = time.perf_counter()
                        self.events.append(dict(kind='body', status='waiting_for_words',
                            added_words=added, required_words=self.config.listening_body_update_words,
                            start=now, end=now, source_stamp=key[0]))
                        return
            # Once work is queued, coalesce subsequent input into its newest
            # snapshot. Reset the word baseline only when that work starts.
            self._body_pending = (copy.deepcopy(data), copy.deepcopy(candidate))
            if self._body_thread is None:
                self._body_thread = threading.Thread(target=self._body_run, name='listening-body', daemon=True)
                self._body_thread.start()

    def _start_audio(self, candidate):
        key = (candidate['text'], self.config.voice, self.config.model)
        with self._lock:
            if self._stopped or not self._audio_preparer or key in self._audio_attempted:
                return
            self._audio_attempted.add(key)
            future = Future()
            future.set_running_or_notify_cancel()
            self._audio_futures[key] = future
            thread = threading.Thread(target=self._audio_run, args=(key, future),
                                      name='listening-prefix-audio', daemon=True)
            self._audio_threads.append(thread)
            thread.start()

    def _audio_run(self, key, future):
        text, voice, model = key
        event = dict(kind='prefix_audio', start=time.perf_counter(), text=text)
        try:
            audio = self._audio_preparer(text, self.config)
            completed = dict(text=text, voice=voice, model=model, tts_out=audio,
                             ready_monotonic=time.perf_counter())
            future.set_result(completed)
            with self._lock:
                if not self._stopped and self._latest and self._latest['text'] == text:
                    self._audio_latest = completed
            event['status'] = 'ready'
        except Exception as exc:
            future.set_exception(exc)
            event.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        finally:
            event['end'] = time.perf_counter()
            with self._lock:
                self.events.append(event)

    def _body_run(self):
        while True:
            with self._lock:
                if self._stopped or self._body_pending is None:
                    self._body_thread = None
                    return
                (data, candidate), self._body_pending = self._body_pending, None
                self._body_started_input = (tuple(data['heard_transcript'].split()), candidate['text'], source_stamp(data))
                previous = copy.deepcopy(self._body_latest)
                # The future belongs to this job, not to the mutable preparation
                # state. An already-dispatched draft may resolve it after freeze.
                draft_future = self._body_future = Future()
                draft_future.set_running_or_notify_cancel()
                self._body_binding = self._binding(candidate)
            event = dict(kind='body', start=time.perf_counter(), source_stamp=source_stamp(data))
            try:
                body_words = body_word_budget(data, candidate['text'])
                plan = resolve_body_plan(data, candidate['framework'])
                initial = candidate.get('body_preparation')
                if (initial and initial['source_stamp'] == source_stamp(data)
                        and initial['prefix_text'] == candidate['text']
                        and initial['framework'] == candidate['framework']):
                    draft = initial['draft']
                    event['source'] = 'shared_speech_draft'
                else:
                    writing_data = data
                    if self.evidence is not None:
                        selected = self.evidence.snapshot().selected
                        from utils.evidence_material import writing_evidence
                        evidence_context = (previous or {}).get('draft') or candidate.get('body_preparation', {}).get('draft', '')
                        writing_data = dict(data, prepared_revision_evidence=writing_evidence(
                            selected, evidence_context, json.dumps(plan, ensure_ascii=False), n_words=body_words))
                    prompt = 'LISTENING BODY DRAFT: ' + build_draft(
                        writing_data, candidate['text'], body_words, previous=previous, prompt_builder=self.prompt_builder)
                    parsed = _json_object(self._complete(body=True, authoring=True,
                        prompt=prompt,
                        sys=self.system_prompt, history_messages=authoring_history(data),
                        json_mode=True, **self.writing_options)[0])
                    draft = parsed.get('draft')
                if not isinstance(draft, str) or not draft.strip():
                    raise SegmentRejected('Invalid speculative body draft')
                draft = remaining_text(draft, candidate['text'])
                if not draft:
                    raise SegmentRejected('Speculative body only repeated the overview')
                needs_fit = speech_length.draft_word_count(draft) > body_words
                event['needs_fit'] = needs_fit
                feedback_status = 'deferred_to_final_input'
                event['feedback_status'] = feedback_status
                prepared = dict(draft=draft, feedback=None, feedback_status=feedback_status,
                    needs_fit=needs_fit, body_plan=plan, prefix_text=candidate['text'],
                    clash_records=data.get('clash_records', []),
                    framework=candidate['framework'], source_stamp=source_stamp(data),
                    heard_transcript=data['heard_transcript'], ready_monotonic=time.perf_counter(),
                    stage=data['stage'], turn=data['turn'])
                event['draft_ready_monotonic'] = prepared['ready_monotonic']
                # Save before requesting feedback. JSON detaches nested inputs
                # and cannot be mutated by either the worker or its consumer.
                draft_future.set_result(json.dumps(prepared, ensure_ascii=False, sort_keys=True))
                with self._lock:
                    if not self._stopped and self._latest and self._latest['text'] == candidate['text']:
                        self._body_latest = copy.deepcopy(prepared)
                event['status'] = 'body_saved_unreviewed'
            except Exception as exc:
                retained = draft_future.done() and draft_future.result() is not None
                if event.get('feedback_status') == 'pending':
                    event['feedback_status'] = 'failed'
                event.update(status='body_saved_unreviewed' if retained else 'body_rejected',
                             error=f'{type(exc).__name__}: {exc}')
            finally:
                if not draft_future.done():
                    draft_future.set_result(None)
            event['end'] = time.perf_counter()
            with self._lock:
                self.events.append(event)

    def freeze(self):
        # Snapshot completed work without waiting for either in-flight worker.
        if self.evidence is not None:
            self.evidence.freeze()
        with self._lock:
            self._stopped = True
            self._pending = self._body_pending = None
            candidate = copy.deepcopy(self._latest)
            if (candidate is not None and self._body_latest is not None
                    and self._body_latest['prefix_text'] == candidate['text']
                    and self._body_latest['framework'] == candidate['framework']):
                candidate['body_preparation'] = copy.deepcopy(self._body_latest)
            return candidate

    def close(self):
        self.freeze()
        if self.evidence is not None:
            self.evidence.close()
        with self._lock:
            self._closed = True
            threads = [self._thread, self._body_thread, *self._audio_threads]
        for thread in threads:
            if thread is not None:
                thread.join()

    def handoff(self, stage, turn, max_time):
        """Freeze reviewed text and completed or in-flight matching audio."""
        candidate = self.freeze()
        if not self.config.listening_prefix_overlap_final_update or not candidate:
            return None
        # Prefer the newest approved version whose audio is already available.
        # With no ready version, preserve transfer of the first in-flight audio.
        with self._lock:
            candidates = copy.deepcopy(self._candidates)
            futures = dict(self._audio_futures)
        for ready in reversed(candidates):
            future = futures.get((ready['text'], self.config.voice, self.config.model))
            if (ready['stage'] == stage and ready['turn'] == turn
                    and _valid_candidate(ready, self.config, max_time)
                    and not self.initial_review.invalidated(ready)
                    and future is not None and future.done() and future.exception() is None):
                if ready.get('review_stamp') != candidate.get('review_stamp'):
                    candidate = ready
                break
        data = candidate.get('reviewed_context')
        if (self.initial_review.invalidated(candidate)
                or not candidate.get('handoff_prechecked') or not data
                or candidate['stage'] != stage or candidate['turn'] != turn
                or not candidate['audits'][-1]['accepted']
                or candidate.get('review_stamp') != review_stamp(candidate, data)
                or framework_stance_conflict(review_payload(candidate, data)) is not None
                or not _valid_candidate(candidate, self.config, max_time)):
            return None
        audio = self.prepared_audio(candidate['text'], self.config)
        ready_future = futures.get((candidate['text'], self.config.voice, self.config.model))
        if ready_future is not None and ready_future.done() and ready_future.exception() is None:
            audio = copy.deepcopy(ready_future.result())
        audio_future = None
        if audio is None:
            with self._lock:
                audio_future = self._audio_futures.get(
                    (candidate['text'], self.config.voice, self.config.model))
            if audio_future is None:
                return None
        return dict(candidate=candidate, data=data, audio=audio, audio_future=audio_future,
                    body_future=self.pending_body(candidate))


def speak_with_listening_prefix(player, max_time, history, config, kwargs, *, start=None,
                                stage=None, handoff=None, complete_input=None, recognized_input=None):
    from agents import Debater
    from pydub import AudioSegment
    import tts_streaming

    start = time.perf_counter() if start is None else start
    stage = stage or player.status
    output = Path(player._speech_audio_file(stage=stage) if handoff else player._speech_audio_file())
    directory = output.parent / f'{output.stem}_chunks'
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.glob('chunk_*.mp3')):
        raise FileExistsError(f'Speech chunks already exist: {directory}')
    preparation = getattr(player, '_listening_prefix', None)
    trace = dict(mode='listening_prefix', status='gating', chunks=[], endpoint_reviews=[],
                 generation_started_monotonic=start, overlaps_final_update=handoff is not None)
    initial_review = preparation.initial_review if preparation is not None else InitialPrefixReview()
    trace['initial_prefix_review_enabled'] = config.listening_prefix_initial_review_enabled
    outer_callback = getattr(player, 'tts_chunk_callback', None)
    speech_plan = []
    save_lock = threading.Lock()
    from .body_audio import FirstBodyAudio
    body_audio = FirstBodyAudio(config, start)
    prefix_duration = Future()
    stream_body = config.listening_single_body_revision and config.listening_stream_body_revision
    tail_ready = Future()
    revision_streams = []
    selected_stream = None
    revision_stopped = threading.Event()
    snapshot_delivery = False
    snapshot_state_synced = False

    def offer_stream(stream):
        nonlocal selected_stream
        if revision_stopped.is_set():
            raise RuntimeError('Speech delivery cancelled')
        selected_stream = stream
        if stream not in revision_streams:
            revision_streams.append(stream)
        tail_ready.set_result(stream)

    def save():
        with save_lock:
            trace['initial_prefix_review'] = initial_review.snapshot()
            trace['first_body_audio'] = body_audio.snapshot()
            if selected_stream is not None:
                trace['body_stream'] = selected_stream.snapshot()
            trace['wall_seconds'] = time.perf_counter() - start
            trace['committed_text'] = '\n\n'.join(c['text'] for c in trace['chunks'])
            path = directory / 'listening_prefix.json'
            temporary = path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(path)

    def record(index, path, text, duration):
        ready = time.perf_counter() - start
        previous_end = trace['chunks'][-1]['estimated_playback_end_seconds'] if trace['chunks'] else ready
        playback_start = max(ready, previous_end)
        trace['chunks'].append(dict(index=index, path=str(path), text=text,
            audio_seconds=duration, ready_seconds=ready,
            estimated_gap_seconds=playback_start - previous_end,
            estimated_playback_end_seconds=playback_start + duration))
        save()
        if index == 0:
            # Use the decoded duration after seam/tempo processing, not TTS
            # metadata. Feedback may already be waiting when this is recorded.
            prefix_duration.set_result(duration)
        if outer_callback is not None:
            outer_callback(index, path, text, duration)

    try:
        data = copy.deepcopy(handoff['data']) if handoff else material(player, stage, history)
        stamp = source_stamp(data)
        candidate = (copy.deepcopy(handoff['candidate']) if handoff else
                     preparation.freeze() if preparation is not None else None)
        trace['speculative_candidate_available'] = candidate is not None
        trace['speculative_candidate'] = copy.deepcopy(candidate)
        if candidate is not None:
            trace['candidate_ready_seconds'] = candidate['ready_monotonic'] - start
        trace['final_transcript'] = None if handoff else data['final_transcript']
        if handoff:
            trace['pre_handoff_transcript'] = data['heard_transcript']
        if candidate is not None and (candidate['stage'] != stage
                or candidate['turn'] != data['turn'] or not _valid_candidate(candidate, config, max_time)):
            trace['discard_reason'] = 'Wrong stage/turn or invalid prefix.'
            candidate = None
        if candidate is not None and config.listening_prefix_initial_review_enabled:
            audits = candidate.get('audits', [])
            if (initial_review.invalidated(candidate) or not audits
                    or not audits[-1].get('semantic_review') or not audits[-1]['accepted']):
                trace['discard_reason'] = 'Opening lacks a valid semantic approval for publication.'
                candidate = None
        gate_data = dict(data, prefix_handoff=True) if config.listening_prefix_overlap_final_update else data
        # A ready handoff publishes from its frozen listening snapshot. Complete
        # recognition is awaited by body workers, never by opening publication.
        # Do not describe this opening as checked against unheard final words.
        trace['prefix_input_scope'] = 'listening_snapshot' if handoff else 'final_input'
        # Temporarily bypass model review; local candidate/format checks still apply.
        trace['prefix_review_enabled'] = config.listening_prefix_review_enabled
        if config.listening_prefix_review_enabled and candidate is not None and (not candidate.get('audits')
                or not candidate['audits'][-1]['accepted']
                or candidate.get('review_stamp') != review_stamp(candidate, gate_data)):
            checked = review(candidate, gate_data, player.helper_client, endpoint=True)
            trace['endpoint_reviews'].append(checked)
            if not checked['accepted']:
                gate_data = dict(gate_data, previous_speech=candidate['text'] + '\n\n'
                    + candidate.get('body_preparation', {}).get('draft', ''),
                    prefix_review_issues=checked['issues'] + checked['format_errors'])
                candidate = None
        if candidate is None:
            # Repair the entire unpublished speech against the same final input.
            trace['cold_prefix'] = True
            candidate = prepare(gate_data, player.helper_client, config, endpoint=True,
                                system_prompt=debater_system(player),
                                writing_options=authoring_options(player, **kwargs),
                                author=player._authoring_client(stage=stage), prompt_builder=player._prepare_stage_prompt,
                                max_time=max_time, initial_review=initial_review)
            trace['endpoint_reviews'].extend(candidate['audits'])
        else:
            trace['cold_prefix'] = False
        prefix = candidate['text']
        def matches_body(value):
            return (isinstance(value, dict) and isinstance(value.get('draft'), str)
                    and bool(value['draft'].strip()) and value.get('prefix_text') == prefix
                    and value.get('stage') == stage and value.get('turn') == data['turn']
                    and value.get('framework', candidate['framework']) == candidate['framework'])

        body_preparation = candidate.get('body_preparation')
        if not matches_body(body_preparation):
            body_preparation = None
        pending_body = (handoff.get('body_future') if handoff else
                        preparation.pending_body(candidate) if preparation is not None else None)
        trace['body_transfer'] = dict(offered=pending_body is not None, adopted=False)
        trace['body_preparation_source'] = 'snapshot' if body_preparation is not None else None

        def take_ready_body():
            nonlocal body_preparation
            if pending_body is None:
                return
            if not pending_body.done():
                trace['body_transfer']['status'] = 'in_flight'
                return
            try:
                raw = pending_body.result()  # Already done; never wait on first audio.
                value = json.loads(raw) if raw is not None else None
                if not matches_body(value):
                    trace['body_transfer']['status'] = 'unavailable_or_mismatched'
                    return
                if (body_preparation is not None and
                        value.get('ready_monotonic', 0) <= body_preparation.get('ready_monotonic', 0)):
                    return
            except Exception as exc:
                trace['body_transfer'].update(status='failed', error=f'{type(exc).__name__}: {exc}')
                return
            body_preparation = value
            trace['body_preparation_source'] = 'inflight_transfer'
            trace['body_transfer'].update(status='adopted', adopted=True,
                                          adopted_seconds=time.perf_counter() - start)
            trace['body_preparation'] = copy.deepcopy(value)

        take_ready_body()
        trace.update(fixed_prefix=prefix, gate_ready_seconds=time.perf_counter() - start,
                     endpoint_source_stamp=None if handoff else stamp, framework=candidate['framework'],
                     body_preparation=copy.deepcopy(body_preparation))
        prepared_audio = (handoff['audio'] if handoff else
                          preparation.prepared_audio(prefix, config) if preparation is not None else None)
        pending_prefix_audio = handoff.get('audio_future') if handoff else None
        if prepared_audio is not None and prepared_audio['text'] != prefix:
            prepared_audio = None
        trace['prepared_audio'] = ({k: v for k, v in prepared_audio.items() if k != 'tts_out'}
                                   if prepared_audio else None)
        save()

        feedback_future = None
        feedback_task, feedback_result, evidence_result, revision_offer = Future(), Future(), Future(), Future()
        frozen_data = copy.deepcopy(data)
        parallel_revision = (config.listening_parallel_endpoint_revision
            and config.listening_single_body_revision
            and (prepared_audio is not None or pending_prefix_audio is not None))
        snapshot_delivery = bool(config.listening_body_snapshot_delivery and parallel_revision
            and handoff and body_preparation and callable(recognized_input))
        trace['body_publication_mode'] = ('complete_asr_snapshot' if snapshot_delivery else 'final_analysis')

        # Capture only values and the stateless provider callable. The observer
        # continues to own player/planner/tree state until complete_input returns.
        feedback_helper, feedback_motion, feedback_side = player.helper_client, player.motion, player.side
        author = player._authoring_client(stage=stage)
        feedback_audiences = tuple(copy.copy(audience) for audience in player.simulated_audience)
        for audience, original in zip(feedback_audiences, player.simulated_audience):
            audience.config = copy.deepcopy(audience.config)
            audience._cost_owner = original
        authoring_system = debater_system(player)
        writing_options = authoring_options(player, **kwargs)
        evidence_cache = (preparation.evidence.snapshot() if preparation is not None
                          and preparation.evidence is not None
                          and preparation.evidence.scope == (stage, data['turn']) else None)
        initial_evidence = copy.deepcopy(getattr(player, 'evidence_pool', []))

        def cached_selection(statement, guidance, candidates, *, stage, call_id=None):
            from utils.evidence_material import EvidenceSelection
            # Reuse only unchanged, still-eligible sources. No final-input
            # selector fallback, including cold starts and changed pools.
            available = {e['id']: e for e in candidates} if stage != 'closing' else {}
            prepared = evidence_cache.selected if evidence_cache is not None else initial_evidence
            selected = EvidenceSelection(
                [e for e in prepared if available.get(e['id']) == e],
                evidence_cache.analysis if evidence_cache is not None else {})
            trace['prepared_evidence'] = dict(
                mode='reuse_only', endpoint_supplement_enabled=False,
                source='listening' if evidence_cache is not None else 'initial_pool',
                calls=evidence_cache.calls if evidence_cache is not None else 0,
                events=copy.deepcopy(evidence_cache.events) if evidence_cache is not None else [],
                selected_ids=[e['id'] for e in selected])
            return selected
        evidence_candidates = copy.deepcopy([e for e in player.high_quality_evidence_pool
            if e['id'] not in player.used_evidence]) if stage != 'closing' else []

        def body_task(draft, final_history, snapshot):
            # Full source history is authoritative; plan/records are derived hints.
            return BodyTask.create(motion=feedback_motion, side=feedback_side, stage=stage,
                history=final_history, prefix=prefix, framework=candidate['framework'], draft=draft,
                preparation=body_preparation, body_plan=resolve_body_plan(snapshot, candidate['framework']),
                clash_records=snapshot.get('clash_records', []),
                feedback_context=snapshot.get('feedback_context'))

        def early_feedback():
            raw_result = None
            try:
                final_history = copy.deepcopy(recognized_input())
                # Adopt any matching draft that completed while recognition was
                # pending. Never wait for unfinished work or change the task later.
                take_ready_body()
                snapshot = (preparation.body_context(stage, data['turn'])
                            if preparation is not None else None) or frozen_data
                trace['body_context_snapshot'] = dict(source_stamp=source_stamp(snapshot),
                    draft_source_stamp=body_preparation.get('source_stamp'),
                    prefix_review_stamp=candidate.get('review_stamp'),
                    data=copy.deepcopy(snapshot))
                task = body_task(body_preparation['draft'], final_history, snapshot)
                feedback_task.set_result(task)
                trace.setdefault('parallel_body_feedback', {}).update(
                    start_seconds=time.perf_counter() - start,
                    history=final_history, statement=task.statement)
                save()
                result = task.review(feedback_helper, feedback_audiences)
                trace['parallel_body_feedback'].update(end_seconds=time.perf_counter() - start)
                trace['parallel_evidence_selection'] = dict(start_seconds=time.perf_counter() - start)
                # Completed feedback must not wait behind an obsolete evidence
                # request; final handoff checks the evidence snapshot separately.
                feedback_result.set_result(result)
                evidence = cached_selection(task.statement, 'Revision Guidance:\n' + result[0],
                    evidence_candidates, stage=stage)
                trace['parallel_evidence_selection'].update(end_seconds=time.perf_counter() - start,
                    selected_ids=[e['id'] for e in evidence])
                evidence_result.set_result(evidence)
                offer = None
                # Publish the feedback and exact revision request before waiting
                # for revision text. A mismatched final request never waits on it.
                if parallel_revision:
                    trace['prefix_duration_wait'] = dict(start_seconds=time.perf_counter() - start)
                    save()
                    used = prefix_duration.result()
                    trace['prefix_duration_wait'].update(end_seconds=time.perf_counter() - start,
                                                         audio_seconds=used)
                    remaining = max_time - used
                    if remaining <= 0:
                        raise SegmentRejected('Opening exhausted the speech duration budget')
                    if task.needs_revision(result[0],
                            tts_streaming.estimate_statement_seconds(task.tail, config), remaining,
                            evidence=evidence):
                        prompt = task.revision_prompt(result[0], remaining, evidence, streaming=stream_body)
                        raw_result = Future()
                        offer = dict(prompt=prompt, system_prompt=authoring_system,
                                     writing_options=writing_options,
                                     history_messages=task.authoring_history(),
                                     future=raw_result, reused=False)
                        if stream_body:
                            from .revision_stream import RevisionStream
                            stream = RevisionStream(prefix=prefix, draft=task.tail,
                                n_words=speech_length.draft_word_budget(remaining), config=config,
                                on_chunk=body_audio.prepare_chunk)
                            revision_streams.append(stream)
                            offer['stream'] = stream
                        trace['parallel_body_revision'] = dict(
                            start_seconds=time.perf_counter() - start, reused=False)
                    else:
                        body_audio.prepare(task.tail, remaining)
                revision_offer.set_result(offer)
                save()
                if offer is not None:
                    if revision_stopped.is_set():
                        raise RuntimeError('Speech delivery cancelled')
                    raw = author(prompt=offer['prompt'], sys=authoring_system, json_mode=False,
                                          history_messages=offer['history_messages'],
                                          **({'on_text': offer['stream'].feed} if 'stream' in offer else {}),
                                          **writing_options)[0]
                    trace['parallel_body_revision']['end_seconds'] = time.perf_counter() - start
                    from utils.speech_text import clean_spoken_revision
                    tail = remaining_text(clean_spoken_revision(raw), prefix)
                    if 'stream' in offer:
                        offer['stream'].finish(raw)
                    else:
                        body_audio.prepare(tail, remaining)
                    # Register speculative audio before releasing the revision,
                    # so final handover can reuse even an in-flight synthesis.
                    raw_result.set_result(raw)
                    save()
            except BaseException as exc:
                if raw_result is not None and offer is not None and 'stream' in offer:
                    offer['stream'].fail(exc)
                trace.setdefault('parallel_body_feedback', {})['worker_error'] = f'{type(exc).__name__}: {exc}'
                save()
                for pending in (feedback_task, feedback_result, evidence_result, revision_offer, raw_result):
                    if pending is not None and not pending.done():
                        pending.set_exception(exc)
                raise

        def snapshot_tail():
            """Commit the complete-ASR task without reading observer-owned state."""
            began = time.perf_counter() - start
            task = feedback_task.result()
            feedback, _ = feedback_result.result()
            evidence = evidence_result.result()
            offer = revision_offer.result()
            context = json.loads(task.context)
            snapshot = dict(context=context, allocation=json.loads(task.allocation),
                draft=task.tail, feedback=task.guidance(feedback),
                prefix_audio_seconds=prefix_duration.result(), evidence=copy.deepcopy(list(evidence)), system_prompt=authoring_system,
                writing_options=copy.deepcopy(writing_options))
            trace['body_publication_snapshot'] = snapshot
            trace['body_publication_snapshot_sha256'] = hashlib.sha256(
                json.dumps(snapshot, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
            trace['body_task_binding'] = dict(source='complete_asr_snapshot', context=context,
                allocation=json.loads(task.allocation), final_derived_context_changed=None)
            trace['body_plan'] = context['body_plan']
            trace['clash_records'] = context['clash_records']
            trace['parallel_body_feedback']['reused'] = True
            trace['parallel_evidence_selection']['reused'] = True
            trace['body_snapshot_committed_seconds'] = time.perf_counter() - start
            save()
            if offer is None:
                tail = task.tail
            else:
                offer['reused'] = True
                trace['parallel_body_revision']['reused'] = True
                if 'stream' in offer:
                    offer_stream(offer['stream'])
                from utils.speech_text import clean_spoken_revision
                tail = remaining_text(clean_spoken_revision(offer['future'].result()), prefix)
            if not tail.strip():
                raise SegmentRejected('Revision omitted the remaining speech')
            return tail, dict(start_seconds=began, end_seconds=time.perf_counter() - start,
                              whole_feedback_passes=1)

        def sync_snapshot_state():
            """Return mutable state to the speaker only after analysis has drained."""
            nonlocal history, data, stamp, snapshot_state_synced
            if snapshot_state_synced:
                return
            final_history = complete_input()
            snapshot_state_synced = True
            history = final_history
            player.status = stage
            player.listen(history)
            player._prepare_speech_claims(stage, history)
            data = material(player, stage, history)
            stamp = source_stamp(data)
            trace.update(final_transcript=data['final_transcript'], endpoint_source_stamp=stamp,
                         final_input_ready_seconds=time.perf_counter() - start,
                         final_input_timing='observed_state_acquisition_after_audio_generation')
            task = feedback_task.result()
            if history != json.loads(task.context)['history']:
                raise SegmentRejected('Completed analysis changed the authoritative complete-ASR history')
            final_task = body_task(task.tail, history, data)
            trace['body_task_binding']['final_derived_context_changed'] = final_task != task
            evidence = evidence_result.result()
            player.used_evidence.update(e['id'] for e in evidence)
            feedback, audiences = feedback_result.result()
            player.debate_thoughts.append(dict(stage=stage, side=feedback_side, mode='revision',
                original_statement=task.statement, allocation_plan=task.allocation,
                simulated_audience_feedback=audiences, feedback_for_revision=task.guidance(feedback),
                selected_evidence_id=[e['id'] for e in evidence], source='complete_asr_snapshot'))
            save()

        def tail_work():
            nonlocal history, data, stamp
            if snapshot_delivery:
                return snapshot_tail()
            began = time.perf_counter() - start
            if handoff is not None:
                history = complete_input()
                player.status = stage
                player.listen(history)
                player._prepare_speech_claims(stage, history)
                data = material(player, stage, history)
                stamp = source_stamp(data)
                trace.update(final_transcript=data['final_transcript'], endpoint_source_stamp=stamp,
                             final_input_ready_seconds=time.perf_counter() - start)
            # The opening has passed publication review. Reconcile the body
            # through the original whole-speech feedback/revision.
            take_ready_body()
            trace['body_transfer']['selection_seconds'] = time.perf_counter() - start
            # Consume the existing planner output directly. Legacy stage setup
            # performed another opening selection whose prompt was discarded.
            plan = resolve_body_plan(data, candidate['framework'])
            trace['body_plan'] = plan
            trace['clash_records'] = exchange_records(
                (player.debate_tree, player.oppo_debate_tree), player.side)
            target_words = speech_length.draft_word_budget(max_time - tts_streaming.estimate_statement_seconds(prefix, config))
            if body_preparation:
                draft = body_preparation['draft']
            else:
                trace['body_preparation_source'] = 'cold_draft'
                final_data = dict(data, endpoint=True)
                instruction = build_draft(final_data, prefix, target_words, json_output=False,
                                          prompt_builder=player._prepare_stage_prompt)
                player._add_message('user', instruction)
                draft = player._authoring_client(stage=stage)(prompt=instruction, sys=debater_system(player),
                    history_messages=authoring_history(final_data), json_mode=False, **writing_options)[0]
            task = body_task(draft, history, data)
            matched_feedback = False
            speculative_revision = None
            if feedback_future is not None:
                # Do not wait even for the feedback of an obsolete source task.
                # When only derived planning changes, keep the task established
                # against complete ASR; publication still checks the source version.
                if feedback_task.done() and feedback_task.exception() is None:
                    early_task = feedback_task.result()
                    matched_feedback = early_task.same_input(task)
                    if matched_feedback:
                        trace['body_task_binding'] = dict(
                            source='complete_asr',
                            final_derived_context_changed=early_task != task)
                        task = early_task
                trace.setdefault('parallel_body_feedback', {})['reused'] = matched_feedback
            trace.setdefault('body_task_binding', dict(source='final_input',
                             final_derived_context_changed=False))
            trace['body_task_binding']['context'] = json.loads(task.context)
            trace['body_task_binding']['allocation'] = json.loads(task.allocation)
            allocation_plan, tail = task.allocation, task.tail
            evidence = []
            passes = 1 if config.listening_single_body_revision or kwargs.get('single_pass_revision',
                getattr(player.config, 'single_pass_revision', False)) else 2
            for i in range(passes):
                precomputed = None
                prepared_evidence = None
                if i == 0 and matched_feedback:
                    precomputed = feedback_result.result()
                    current_evidence = [e for e in player.high_quality_evidence_pool
                        if e['id'] not in player.used_evidence] if stage != 'closing' else []
                    evidence_matches = evidence_candidates == current_evidence
                    trace['parallel_evidence_selection']['reused'] = evidence_matches
                    if evidence_matches:
                        prepared_evidence = evidence_result.result()
                        speculative_revision = revision_offer.result()
                feedback, new, allocation, _ = player._get_revision_suggestion(
                    statement=prefix + '\n\n' + tail, history=history,
                    add_evidence=i == 0, frozen_prefix=prefix, precomputed_body_feedback=precomputed,
                    precomputed_revision_evidence=prepared_evidence,
                    revision_evidence_selector=cached_selection,
                    body_task=replace(task, tail=tail), **kwargs)
                allocation = allocation or allocation_plan
                if i == 0:
                    evidence = new
                used = prefix_duration.result()
                remaining = max_time - used
                if remaining <= 0:
                    raise SegmentRejected('Opening exhausted the speech duration budget')
                if not task.needs_revision(feedback,
                        tts_streaming.estimate_statement_seconds(tail, config), remaining,
                        include_prepared=i == 0, evidence=new if i == 0 else ()):
                    continue
                feedback = task.guidance(feedback, include_prepared=i == 0)
                tail = player._length_adjust(tail, feedback, evidence, allocation, remaining,
                    max_retry=1 if config.listening_single_body_revision else 2,
                    defer_duration_fit=config.listening_single_body_revision, frozen_prefix=prefix, history=history,
                    **({'on_revision_stream': offer_stream} if stream_body else {}),
                    **({'speculative_revision': speculative_revision} if i == 0 and speculative_revision else {}))
                if i == 0 and speculative_revision is not None:
                    trace['parallel_body_revision']['reused'] = speculative_revision['reused']
                tail = remaining_text(tail, prefix)
                if not tail.strip():
                    raise SegmentRejected('Revision omitted the remaining speech')
            return tail, dict(start_seconds=began, end_seconds=time.perf_counter() - start,
                              whole_feedback_passes=passes)

        def validate(index, text):
            if snapshot_delivery and index > 0:
                # The recognition contract supplies the final immutable history,
                # independently of slower tree/planning work. Never read those
                # mutable projections to authorize an already-bound body chunk.
                task = feedback_task.result()
                if recognized_input() != json.loads(task.context)['history']:
                    raise SegmentRejected('Complete-ASR history changed after body snapshot binding')
            elif not (handoff is not None and index == 0):
                current = material(player, stage, history, include_rehearsal=False)
                if source_stamp(current, include_rehearsal=False) != source_stamp(data, include_rehearsal=False):
                    raise SegmentRejected('Input changed after final handover')
            if index == 0 and text != prefix:
                raise SegmentRejected('TTS changed the fixed opening')
            if index > 0:
                remaining_text(text, prefix)
                if ' '.join(prefix.split()).casefold() in ' '.join(text.split()).casefold():
                    raise SegmentRejected('TTS repeated the opening')

        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='listening-tail') as executor:
            if (config.listening_parallel_body_feedback or config.listening_parallel_endpoint_revision) and handoff and body_preparation and recognized_input:
                feedback_future = executor.submit(early_feedback)
            def run_tail():
                try:
                    result = tail_work()
                    if not tail_ready.done():
                        tail_ready.set_result(result[0])
                    return result
                except BaseException as exc:
                    if not tail_ready.done():
                        tail_ready.set_exception(exc)
                    if selected_stream is not None:
                        selected_stream.fail(exc)
                    raise
            future = executor.submit(run_tail)
            def supply_tail():
                if stream_body:
                    return tail_ready.result()
                tail, timing = future.result()
                trace.update(tail_work=timing, revised_tail=tail)
                save()
                return tail
            trace['status'] = 'delivering'
            try:
                if prepared_audio is None and pending_prefix_audio is not None:
                    trace['prefix_audio_transfer'] = dict(status='waiting', start_seconds=time.perf_counter() - start)
                    save()
                    try:
                        prepared_audio = pending_prefix_audio.result()
                        if (prepared_audio['text'], prepared_audio['voice'], prepared_audio['model']) != (prefix, config.voice, config.model):
                            raise SegmentRejected('Transferred prefix audio does not match the reviewed text/voice/model')
                    except Exception as exc:
                        trace['prefix_audio_transfer'].update(status='failed', error=f'{type(exc).__name__}: {exc}')
                        raise
                    trace['prefix_audio_transfer'].update(status='reused', ready_seconds=time.perf_counter() - start)
                trace['prepared_audio'] = ({k: v for k, v in prepared_audio.items() if k != 'tts_out'}
                                           if prepared_audio else None)
                save()
                text, _, duration = tts_streaming.convert_text_to_speech_streaming(
                    prefix, str(output), max_time, config=config, on_chunk=record,
                    motion=player.motion, side=player.side, tail_supplier=supply_tail,
                    validate_chunk=validate, prepared_first_audio=prepared_audio,
                    prepared_body_audio=body_audio)
                tail, timing = future.result()
                trace.update(tail_work=timing, revised_tail=tail)
                if not prefix_duration.done():
                    raise SegmentRejected('Speech completed without first audio')
            except BaseException as exc:
                revision_stopped.set()
                for stream in revision_streams:
                    stream.fail(exc)
                # Release duration waiters before the executor joins them when
                # validation, decoding or publication fails before the first chunk.
                if not prefix_duration.done():
                    prefix_duration.set_exception(exc)
                raise
        if snapshot_delivery:
            sync_snapshot_state()
        trace.update(status='completed', audio_seconds=duration,
                     signed_duration_error_seconds=duration - max_time,
                     estimated_total_gap_seconds=sum(c['estimated_gap_seconds'] for c in trace['chunks']))
        response = Debater.post_process(player, text, max_time, time_control=False)
    except Exception as exc:
        trace.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        if snapshot_delivery and not snapshot_state_synced:
            # Do not write conversation/state while the observer still owns it,
            # even when generation failed after publishing only a partial body.
            try:
                complete_input()
                player.status = stage
            except Exception as state_error:
                trace['state_sync_error'] = f'{type(state_error).__name__}: {state_error}'
        if trace['chunks']:
            committed = '\n\n'.join(c['text'] for c in trace['chunks'])
            Debater.post_process(player, committed, max_time, time_control=False)
            combined = AudioSegment.silent(duration=0)
            for c in trace['chunks']:
                combined += AudioSegment.from_file(c['path'])
            buffer = BytesIO()
            combined.export(buffer, format='mp3')
            output.write_bytes(buffer.getvalue())
        raise
    finally:
        body_audio.close()
        if preparation is not None:
            preparation.close()
            trace['speculative_calls'] = preparation._calls
            trace['semantic_rewrites'] = preparation._rewrites
            trace['preparation_events'] = copy.deepcopy(preparation.events)
            for event in trace['preparation_events']:
                event['start_seconds'] = event['start'] - start
                event['end_seconds'] = event['end'] - start
            player._listening_prefix = None
        save()
        player.debate_thoughts.append(dict(mode='listening_prefix_delivery',
            stage=player.status, side=player.side, trace=trace))
    if player.use_debate_flow_tree:
        player._analyze_statement(response, player.side, planned_actions=speech_plan)
    return response
