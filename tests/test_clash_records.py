"""Source-bound exchange continuity, corrections and immutable prompt snapshots."""
import json
from unittest.mock import Mock

from debate_tree import DebateTree
from streaming.clash_records import exchange_records, prompt_records
from streaming.body_feedback import review_whole_speech
from streaming.listening_prefix import material
from test_retained_tree_selection import add, amend
from test_listening_prefix import prepared_player


def test_three_move_exchange_preserves_speakers_links_and_pending_reply():
    tree = DebateTree('Trial', 'for')
    first = add(tree, 'A trial improves access.')
    objection = add(tree, 'Who funds it?', first, 'against', 'attack')
    reply = add(tree, 'A grant funds the trial only.', objection, 'for', 'reply')
    for order, node in enumerate((first, objection, reply), 1):
        node.update_order = order
    record, = exchange_records([tree], 'against')
    assert record['clash_id'] == first.node_id
    assert [r['side'] for r in record['entries']] == ['for', 'against', 'for']
    assert record['entries'][-1]['responds_to'] == objection.node_id
    assert record['awaiting_response'] == [dict(node_id=reply.node_id,
        from_side='against', excerpt=reply.source_spans[0])]
    # Persistence uses existing serialized source trees, not a stale second store.
    restored = DebateTree.from_json(tree.get_tree_info())
    assert exchange_records([restored], 'against') == [record]


def test_revision_keeps_clash_identity_and_marks_old_responses_historical():
    tree = DebateTree('Trial', 'for')
    first = add(tree, 'All recordings will be stored.')
    first.update_order = 1
    objection = add(tree, 'Permanent storage is risky.', first, 'against', 'attack')
    objection.update_order = 2
    amend([tree], first, 'revise', 'Only consented recordings, deleted after seven days.')
    record, = exchange_records([tree], 'against')
    assert record['clash_id'] == first.node_id
    assert record['topic'] == 'Only consented recordings, deleted after seven days.'
    assert [r['status'] for r in record['entries']] == ['superseded', 'needs_review', 'current']
    assert record['latest_change'] == record['topic']
    assert record['awaiting_response'][0]['node_id'] == first.superseded_by
    assert record['awaiting_response'][0]['node_id'] != objection.node_id
    amend([tree], tree.root.children[-1], 'retract', 'I withdraw the recording proposal.')
    withdrawn, = exchange_records([tree], 'against')
    assert withdrawn['status'] == 'withdrawn' and not withdrawn['awaiting_response']
    assert withdrawn['latest_change'] == 'I withdraw the recording proposal.'


def test_bounded_view_retains_independent_branches_and_excludes_private_drafts():
    tree = DebateTree('Trial', 'for')
    first = add(tree, 'First main issue.')
    node = first
    for i in range(7):
        node = add(tree, f'Reply {i}.', node, 'against' if i % 2 == 0 else 'for', 'reply')
        node.update_order = i + 1
    private = add(tree, 'PRIVATE UNDELIVERED DRAFT')
    private.source_spans = []
    for i in range(4):
        add(tree, f'Independent issue {i}.')
    records = exchange_records([tree], 'against')
    assert len(records) == 5 and len(prompt_records(records)) == 3
    assert 'PRIVATE UNDELIVERED DRAFT' not in json.dumps(records)
    record = next(r for r in records if r['clash_id'] == first.node_id)
    assert len(record['entries']) == 4 and record['omitted_entries'] == 4
    assert record['entries'][0]['node_id'] == first.node_id
    assert record['entries'][-1]['node_id'] == node.node_id


def test_concession_and_clipped_excerpt_do_not_assert_resolution():
    tree = DebateTree('Trial', 'for')
    first = add(tree, 'Funding is needed before launch.')
    concession = add(tree, 'We accept this prerequisite. ' + 'Detail. ' * 30,
                     first, 'against', 'concede')
    record, = exchange_records([tree], 'against')
    assert record['awaiting_response'] == []
    assert record['entries'][-1]['relation'] == 'concede'
    assert record['entries'][-1]['excerpt_truncated']
    assert 'resolved' not in record
    assert len(concession.source_spans[0]) > len(record['entries'][-1]['excerpt'])


def test_listening_material_and_feedback_carry_detached_records(tmp_path):
    p, history, node = prepared_player(tmp_path)
    data = material(p, p.status, history)
    assert data['clash_records']
    snapshot = json.dumps(data['clash_records'])
    node.source_spans.append('Later input cannot mutate this frozen snapshot.')
    assert json.dumps(data['clash_records']) == snapshot
    helper = Mock(return_value=[json.dumps(dict(points=[]))])
    review_whole_speech(helper, motion=p.motion, side=p.side, stage=p.status,
        statement='A response.', history=history, prefix='An opening.',
        clash_records=data['clash_records'])
    prompt = helper.call_args.kwargs['prompt']
    payload = json.loads(prompt.rsplit('\n', 1)[-1])
    assert all(h['content'].replace('\n', ' ') in prompt for h in history)
    assert payload['fixed_prefix'] == 'An opening.'
    assert 'Core Message Clarity' in prompt
    assert 'Later input cannot mutate' not in prompt


def test_selected_path_keeps_middle_objection_and_full_reply_conditions():
    tree = DebateTree('Identity', 'for')
    root = add(tree, 'Verification increases account costs.')
    objection = add(tree, 'Publishing names exposes vulnerable speakers.', root, 'against', 'attack')
    reply = add(tree, 'We verify privately. ' + 'Technical detail. ' * 20 +
                'Legal names are never shown to other users.', objection, 'for', 'reply')
    reply.constraints = [dict(kind='scope', quote='Legal names are never shown to other users.')]
    # Later sibling exchanges used to crowd the selected objection out of the view.
    for i in range(5):
        sibling = add(tree, f'Other objection {i}.', root, 'against', 'attack')
        sibling.update_order = 10 + i
    reply.update_order = 3
    record, = exchange_records([tree], 'against', target_ids=[reply.node_id])
    assert objection.node_id not in [e['node_id'] for e in record['entries']]
    chain, = record['response_chains']
    assert chain['our_objection']['node_id'] == objection.node_id
    assert chain['latest_opponent_reply']['quote'] == reply.source_spans[-1]
    assert chain['latest_opponent_reply']['constraints'][0]['quote'].endswith('other users.')
    assert chain['target_node_id'] == reply.node_id
    assert 'resolved' not in chain


def test_chain_follows_descendant_reply_and_preserves_sibling_safeguard():
    tree = DebateTree('Trial', 'for')
    root = add(tree, 'We will run a trial.')
    objection = add(tree, 'No funding or consent?', root, 'against', 'attack')
    funding = add(tree, 'External grants fund it.', objection, 'for', 'reply')
    consent = add(tree, 'We require consent.', objection, 'for', 'concede')
    funding.update_order, consent.update_order = 2, 3
    record, = exchange_records([tree], 'against', target_ids=[root.node_id])
    chain, = record['response_chains']
    assert chain['latest_opponent_reply']['node_id'] == consent.node_id
    assert chain['other_replies'][0]['node_id'] == funding.node_id
    amend([tree], root, 'retract', 'We withdraw the trial.')
    record, = exchange_records([tree], 'against', target_ids=[root.node_id])
    assert not record['response_chains']


def test_selected_issue_wins_over_more_recent_unrelated_records():
    tree = DebateTree('Trial', 'for')
    targets = []
    for i in range(5):
        node = add(tree, f'Issue {i}.')
        node.update_order = i
        targets.append(node.node_id)
    records = exchange_records([tree], 'against', target_ids=targets)
    selected = prompt_records(records, target_ids=[targets[0]])
    assert selected[0]['clash_id'] == targets[0]
    assert sum(len(r['response_chains']) for r in records) == 4
