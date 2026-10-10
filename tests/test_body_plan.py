"""Plan provenance, promised coverage and weighted body allocation, offline."""
import copy
import json
from unittest.mock import Mock

import pytest

from streaming.body_plan import parse_body_plan, resolve_body_plan, allocate_words
from streaming.body_task import BodyTask
from streaming.branch_planning import branch_prompt, parse_branch_state
from streaming.listening_prefix import material
from test_claim_constraints import indexed
from test_listening_prefix import prepared_player, FRAMEWORK, PREFIX, TAIL, trace
from test_full_speech import audio as audio


def item(**kwargs):
    return dict(dict(axis=0, target=None, issue='Funding', action='answer_objection',
                     point='Explain the funding condition.', weight=3), **kwargs)


def test_legacy_planning_call_carries_source_bound_body_plan(tmp_path):
    p, _, _ = prepared_player(tmp_path)
    context = p._planning_context()
    context.pop('listening_source_selection')  # Historical non-listening schema.
    raw = json.loads(indexed())
    raw.update(overview=FRAMEWORK, body_plan=[item(target=0)])
    state = parse_branch_state(json.dumps(raw), ' '.join(p.planner.chunks), context)
    plan = state['body_plan'][0]
    assert plan['covers'] == [FRAMEWORK['response_axes'][0]]
    source = context['tree_targets'][0]
    assert plan['target'] == dict(node_id=source['node_id'], version=source['version'], quote=source['sources'][-1])
    assert 'body_plan' in branch_prompt(context, p.planner.chunks, {})
    p.planner.state = state
    # Historical tactics may be inspected, but never enter current listening authoring.
    assert material(p, p.status)['body_plan'] == []


@pytest.mark.parametrize('changes', [dict(axis=3), dict(axis=True), dict(target=-1),
    dict(target=True), dict(weight=0), dict(weight=6), dict(weight=True),
    dict(action='attack_everything'), dict(point='')])
def test_bad_plan_fields_cannot_become_guidance(changes):
    with pytest.raises(ValueError):
        parse_body_plan([item(**changes)], FRAMEWORK, [])


def test_stale_target_is_dropped_but_promised_issue_still_has_coverage():
    target = dict(node_id='n1', version=1, sources=['Only externally funded trials.'])
    plan = parse_body_plan([item(target=0)], FRAMEWORK, [target])
    data = dict(body_plan=plan, current_targets=[dict(target, version=2)])
    resolved = resolve_body_plan(data, FRAMEWORK)
    assert {axis for entry in resolved for axis in entry['covers']} == set(FRAMEWORK['response_axes'])
    assert all(entry['target'] is None for entry in resolved)
    assert 'Explain the funding condition.' not in json.dumps(resolved)
    # Changing the overview also removes tactics attached to old promises.
    changed = dict(FRAMEWORK, response_axes=['Privacy'])
    assert resolve_body_plan(dict(body_plan=plan, current_targets=[target]), changed)[0]['issue'] == 'Privacy'


def test_missing_promises_survive_full_optional_plan_and_budget_is_exact():
    framework = dict(FRAMEWORK, response_axes=['Privacy', 'Security', 'Inclusivity'])
    plan = parse_body_plan([item(axis=None, issue=f'Other{i}') for i in range(4)], framework, [])
    resolved = resolve_body_plan(dict(body_plan=plan), framework)
    assert len(resolved) == 4
    assert {axis for entry in resolved for axis in entry['covers']} == set(framework['response_axes'])
    allocated = allocate_words(resolved, 257)
    assert sum(entry['words'] for entry in allocated) == 257
    assert allocated[-1]['words'] > allocated[0]['words']  # optional issue weight3 vs fallback1
    assert all(entry['words'] > 0 for entry in allocated)


def test_plan_consumed_by_review_and_revision_and_bound_to_task_identity():
    plan = parse_body_plan([item()], FRAMEWORK, [])
    args = dict(motion='Motion', side='for', stage='rebuttal', history=[],
        prefix=PREFIX, framework=FRAMEWORK, draft='STALE PLAN**Statement**' + TAIL, body_plan=plan)
    task = BodyTask.create(**args)
    mutated = copy.deepcopy(plan)
    mutated[0]['point'] = 'A changed final response.'
    assert task != BodyTask.create(**dict(args, body_plan=mutated))
    plan[0]['weight'] = 1
    helper = Mock(return_value=['{"points":[]}'])
    task.review(helper)
    payload = json.loads(helper.call_args.kwargs['prompt'].rsplit('\n', 1)[-1])
    assert payload['body_plan'][0]['weight'] == 3
    prompt = task.revision_prompt('No changes', 100)
    assert 'body_plan' in prompt and 'Explain the funding condition.' in prompt
    assert 'body_word_allocations' not in prompt
    assert 'STALE PLAN' not in prompt


def test_cold_opening_uses_one_plan_without_legacy_claim_selection(audio, tmp_path):
    p, _, _ = prepared_player(tmp_path)
    p.planner.chunks = []
    p.claim_selection = Mock(side_effect=AssertionError('Duplicate selection'))
    p.opening_generation([], 60, time_control=True)
    p.claim_selection.assert_not_called()
    p._prepare_stage_prompt.assert_called_once()
    saved = trace(tmp_path)
    assert {axis for entry in saved['body_plan'] for axis in entry['covers']} == set(saved['framework']['response_axes'])
    prompt = next(c.kwargs['prompt'] for c in p.helper_client.call_args_list
                  if c.kwargs['prompt'].startswith('LISTENING PREFIX DRAFT:'))
    data = json.loads(prompt.rsplit('\n', 1)[-1])['context']
    assert 'current_plan' not in data
    assert 'opening and body together' in prompt
    p._get_response.assert_called_once()


def test_selected_exchange_enters_review_revision_and_invalidates_old_task():
    target = dict(node_id='n1', version=1, sources=['Verified privately.'])
    plan = parse_body_plan([item(target=0)], FRAMEWORK, [target])
    chain = dict(target_node_id='n1', latest_opponent_reply=dict(quote='Names remain private.'))
    data = dict(body_plan=plan, current_targets=[target], clash_records=[dict(response_chains=[chain])])
    resolved = resolve_body_plan(data, FRAMEWORK)
    args = dict(motion='Motion', side='for', stage='rebuttal', history=[],
                prefix=PREFIX, framework=FRAMEWORK, draft=TAIL)
    task = BodyTask.create(**args, body_plan=resolved)
    assert 'Names remain private.' in task.revision_prompt('No changes', 100)
    helper = Mock(return_value=['{"points":[]}'])
    task.review(helper)
    assert 'Names remain private.' in helper.call_args.kwargs['prompt']
    chain['latest_opponent_reply']['quote'] = 'We now require independent consent.'
    assert task != BodyTask.create(**args, body_plan=resolve_body_plan(data, FRAMEWORK))
    assert resolved[0]['response_chain']['latest_opponent_reply']['quote'] == 'Names remain private.'
