"""Readable source ownership and exact indexed selections after projection."""
import copy
import json

from streaming.planning_view import readable_planning_context
from streaming.branch_planning import branch_prompt, parse_branch_state


def context():
    quote = 'We permit alternative verification, but only with independent funding.'
    condition = dict(kind='precondition', quote='only with independent funding',
                     source_node_id='opponent', constraint_id='condition1')
    source = dict(node_id='opponent', side='for', status='current', relation='reply',
                  responds_to='ours', quote=quote, constraints=[condition])
    objection = dict(node_id='ours', side='against', status='current', relation='attack',
                     responds_to='root', quote='Access needs an alternative to official documents.', constraints=[])
    target = dict(node_id='opponent', version='v1', claim='Alternative access with funding.',
                  sources=[quote], arguments=[quote, 'A distinct supplemental mechanism.'], constraints=[condition])
    return dict(motion='Verification', our_side='against', use_topology=False,
        prior_debate=[dict(role='assistant', content='Our full previous speech.')],
        tree_targets=[target], position_limits=[condition['quote'], quote, 'An unrelated safeguard.'],
        constraints=[dict(condition, node_id='opponent', claim=target['claim'])],
        correction_history=[], overview_preparation={'stage':'closing'},
        clash_records=[dict(topic='Access', entries=[], response_chains=[dict(
            target_node_id='opponent', target=copy.deepcopy(source),
            latest_opponent_reply=copy.deepcopy(source), our_objection=copy.deepcopy(objection),
            our_prior_position=copy.deepcopy(objection), other_replies=[], our_latest_response=None)])])


def test_exchange_keeps_full_text_and_inline_conditions_without_duplicate_views():
    original = context()
    frozen = copy.deepcopy(original)
    view = readable_planning_context(original)
    assert original == frozen
    chain = view['clash_records'][0]['response_chains'][0]
    assert 'target' not in chain and 'our_prior_position' not in chain
    reply = chain['latest_opponent_reply']
    assert reply['quote'] == original['tree_targets'][0]['sources'][0]
    assert reply['also_selected_target'] and chain['our_objection']['also_our_prior_position']
    assert reply['constraints'][0]['boundary_indexes'] == [0]
    assert reply['boundary_indexes'] == [1]
    assert view['additional_boundaries'] == [dict(index=2, quote='An unrelated safeguard.')]
    assert view['tree_targets'][0]['sources'] == []
    assert view['tree_targets'][0]['arguments'] == ['A distinct supplemental mechanism.']
    assert view['tree_targets'][0]['constraints'] == [] and view['constraints'] == []
    assert view['prior_debate'] == original['prior_debate']
    assert 'position_limits' not in view
    prompt = branch_prompt(original, ['Current complete sentence.'], {})
    payload = json.loads(prompt.rsplit('\n', 1)[-1])
    assert payload['context'] == view and payload['heard_prefix'] == ['Current complete sentence.']
    assert 'READABLE EXCHANGES' in prompt and 'context.position_limits' not in prompt
    assert 'quote_ref' not in prompt


def test_equal_text_never_merges_different_node_speaker_status_or_qualification():
    for difference in ('node_id', 'side', 'status'):
        original = context()
        chain = original['clash_records'][0]['response_chains'][0]
        for key in ('target', 'latest_opponent_reply'):
            chain[key][difference] = {'node_id':'other', 'side':'against', 'status':'superseded'}[difference]
        view = readable_planning_context(original)
        assert view['tree_targets'][0]['sources'] == original['tree_targets'][0]['sources']
        assert view['tree_targets'][0]['constraints']
    original = context()
    original['tree_targets'][0]['constraints'][0]['kind'] = 'scope'
    view = readable_planning_context(original)
    assert view['tree_targets'][0]['constraints'][0]['kind'] == 'scope'


def test_boundary_is_not_attached_to_our_matching_quote():
    original = context()
    text = original['position_limits'][0]
    chain = original['clash_records'][0]['response_chains'][0]
    chain['our_objection']['quote'] = text
    view = readable_planning_context(original)
    ours = view['clash_records'][0]['response_chains'][0]['our_objection']
    assert 'boundary_indexes' not in ours and 'boundary_options' not in ours
    theirs = view['clash_records'][0]['response_chains'][0]['latest_opponent_reply']
    assert theirs['constraints'][0]['boundary_indexes'] == [0]


def test_selection_indices_still_bind_original_sources_and_versions():
    original = context()
    original.pop('overview_preparation')
    response = json.dumps(dict(claims=[dict(target=0)], limits=[0], rebuttals=[]))
    before = parse_branch_state(response, original['tree_targets'][0]['sources'][0], original)
    readable_planning_context(original)
    after = parse_branch_state(response, original['tree_targets'][0]['sources'][0], original)
    assert before == after
    assert after['claims'][0]['node_id'] == 'opponent'
    assert after['position_limits'] == original['position_limits']
    assert after['constraints'] == original['constraints']
    # Non-listening comparisons retain their original source layout.
    payload = json.loads(branch_prompt(original, [], {}).rsplit('\n', 1)[-1])
    assert payload['context']['tree_targets'][0].pop('target_index') == 0
    assert payload['context'] == original


def test_historical_or_incompletely_covered_entry_stays_visible():
    original = context()
    source = original['clash_records'][0]['response_chains'][0]['target']
    entry = {k:source[k] for k in ('node_id','side','status','relation','responds_to')}
    entry.update(excerpt=source['quote'], excerpt_truncated=False)
    old = dict(entry, status='superseded')
    other = dict(entry, excerpt='Earlier different qualification.')
    original['clash_records'][0]['entries'] = [entry, old, other]
    result = readable_planning_context(original)['clash_records'][0]
    assert result['entries'] == [old, other]
    assert result['entries_in_response_chains'] == 1


def test_explicit_target_indices_bind_selected_sources_without_mutating_context():
    original = context()
    original['listening_source_selection'] = True
    for i in range(1, 3):
        target = copy.deepcopy(original['tree_targets'][0])
        target.update(node_id=f'opponent_{i}', claim=f'Claim {i}', sources=[f'Opponent source {i}.'])
        original['tree_targets'].append(target)
    frozen = copy.deepcopy(original)
    prompt = branch_prompt(original, ['Opponent source 2.'], {})
    displayed = json.loads(prompt.rsplit('\n', 1)[-1])['context']['tree_targets']
    assert [t['target_index'] for t in displayed] == [0, 1, 2]
    overview = dict(ready=True, core_dispute='Access', response_axes=[], prefix_action='keep', reason='unchanged')
    response = json.dumps(dict(claims=[dict(target=displayed[2]['target_index'])], limits=[], overview=overview))
    selected = parse_branch_state(response, 'Opponent source 2.', original)
    assert selected['claims'][0]['node_id'] == 'opponent_2'
    assert selected['claims'][0]['quote'] == 'Opponent source 2.'
    assert original == frozen
