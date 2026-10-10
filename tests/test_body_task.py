"""Offline equivalence and invalidation of early/committed body work."""
from dataclasses import replace
import json
from unittest.mock import Mock

import pytest

from streaming.body_task import BodyTask
from streaming.body_revision import revision_prompt
from utils import speech_length


def make_task(**overrides):
    args = dict(motion='Allow trials', side='for', stage='rebuttal',
        history=[dict(side='against', content='Only if funded.')], prefix='We support trials.',
        framework={'response_axes': ['Funding']}, draft='Funding can be secured.',
        preparation={'feedback': 'Qualify the funding claim.'},
        clash_records=[{'latest_response': 'Only if funded.'}])
    args.update(overrides)
    return BodyTask.create(**args)


def test_snapshot_owns_nested_inputs_and_reuse_binds_all_context():
    history = [dict(side='against', content='Only if funded.')]
    records = [{'latest_response': 'Only if funded.'}]
    task = make_task(history=history, clash_records=records)
    equivalent = make_task(history=history, clash_records=records)
    history[0]['content'] = 'We now reject trials.'
    records[0]['latest_response'] = 'We now reject trials.'
    assert task == equivalent
    assert task != make_task(history=history)
    assert task != make_task(clash_records=records)
    assert task.revision_prompt('No changes', 120) != make_task(clash_records=records).revision_prompt('No changes', 120)
    assert task != make_task(prefix='We support funded trials.')
    assert task != make_task(preparation={'feedback': 'No changes'})
    helper = Mock(return_value=['{"points":[]}'])
    task.review(helper)
    payload = json.loads(helper.call_args.kwargs['prompt'].rsplit('\n', 1)[-1])
    prompt = helper.call_args.kwargs['prompt']
    assert 'Only if funded.' in prompt and 'We now reject trials.' not in prompt
    assert payload['fixed_prefix'] == 'We support trials.'


def test_early_and_committed_revision_inputs_match():
    task = make_task(draft='Plan**Statement:** Funding can be secured.')
    feedback = 'Explain how funding is secured.'
    assert task.revision_prompt(feedback, 120) == revision_prompt(
        motion='Allow trials', side='for', stage='rebuttal', statement=task.tail,
        feedback=task.guidance('Revision Guidance:\n' + feedback), allocation_plan=task.allocation,
        evidence=[], prefix='We support trials.',
        n_words=speech_length.draft_word_budget(120))


def test_same_input_allows_only_derived_context_changes():
    task = make_task()
    assert task.same_input(make_task(body_plan=[{'issue': 'Funding', 'weight': 3}],
                                     clash_records=[{'latest_response': 'A new extraction.'}]))
    for changes in (
        dict(history=[dict(side='against', content='We now reject trials.')]),
        dict(draft='A different draft.'), dict(prefix='A different overview.'),
        dict(framework={'response_axes': ['Rights']}), dict(side='against'),
        dict(stage='closing'), dict(preparation={'feedback': None}),
        dict(preparation={'feedback': 'Qualify the funding claim.', 'needs_fit': True}),
    ):
        assert not task.same_input(make_task(**changes))


def test_overlong_preparation_requires_one_revision_even_if_duration_estimate_fits():
    task = make_task(preparation={'feedback': 'No changes', 'needs_fit': True})
    assert task.needs_revision('No changes', 120, 120)
    assert not task.needs_revision('No changes', 120, 120, include_prepared=False)
    assert not replace(task, needs_fit=False).needs_revision('No changes', 120, 120)
    assert task.needs_revision('Qualify funding.', 120, 120, include_prepared=False)
    assert task.needs_revision('No changes', 200, 120, include_prepared=False)


def test_unfinished_preparatory_feedback_is_distinct_from_a_clean_review():
    task = make_task(preparation={'feedback': None, 'feedback_status': 'pending'})
    clean = make_task(preparation={'feedback': 'No changes', 'feedback_status': 'completed'})
    assert task != clean and task.prepared_feedback is None
    assert 'Unresolved preparatory corrections' not in task.guidance('Apply the final condition.')
    assert task.needs_revision('Apply the final condition.', 120, 120)


def test_supplemental_evidence_requires_revision_even_when_feedback_and_length_pass():
    task = make_task(preparation={'feedback': 'No changes'})
    assert not task.needs_revision('No changes', 120, 120)
    evidence = [{'id': 'source', 'content': 'Funding was independently verified.'}]
    assert task.needs_revision('No changes', 120, 120, evidence=evidence)
    assert 'Funding was independently verified.' in task.revision_prompt('No changes', 120, evidence)
