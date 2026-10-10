"""Local prepared arguments must reach drafts without becoming heard evidence."""
import copy
from unittest.mock import Mock

import pytest

from streaming.listening_prefix import material, PrefixPreparation
from test_listening_prefix import prepared_player, PREFIX, TAIL
from test_full_speech import audio as audio
from test_listening_overview import wait_for

pytestmark = pytest.mark.usefixtures('word_length_modes')
PRIVATE = 'PRIVATE_REHEARSAL_MATERIAL: verification can increase account acquisition costs.'


def enable(p):
    p.use_rehearsal_tree = True
    p.config.rehearsal_mode = 'hybrid'
    p.pool_file = '/prepared/gemma_pool_for.json'
    p._retrieve_on_prepared_tree = Mock(return_value=PRIVATE)


def test_retrieval_reaches_body_draft_but_not_opponent_sources(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    before = material(p, p.status, history)
    enable(p)
    data = material(p, p.status, history)
    assert data['prepared_rehearsal_materials']
    assert data['heard_transcript'] == before['heard_transcript']
    assert data['opponent_sources'] == before['opponent_sources']
    assert data['supplied_evidence'] == before['supplied_evidence']
    prep = PrefixPreparation(p.planner.turn, p.helper_client, p.streaming_output_config)
    try:
        prep.offer(data)
        wait_for(lambda: prep._body_latest is not None)
    finally:
        prep.close()
    prompts = [c.kwargs['prompt'] for c in p.helper_client.call_args_list]
    assert any(x.startswith('LISTENING PREFIX DRAFT:') and PRIVATE in x for x in prompts)
    assert all(PRIVATE not in x for x in prompts if x.startswith(
        ('ENDPOINT GATE:', 'LISTENING PREFIX REVIEW:')))


def test_local_recall_cache_revalidates_target_version_and_stage(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    enable(p)
    data = material(p, p.status, history)
    assert p._retrieve_on_prepared_tree.call_count > 0
    count = p._retrieve_on_prepared_tree.call_count
    p._listening_rehearsal_materials(data)
    assert p._retrieve_on_prepared_tree.call_count == count
    changed = copy.deepcopy(data)
    changed['current_targets'][0]['version'] += '-changed'
    p._listening_rehearsal_materials(changed)
    assert p._retrieve_on_prepared_tree.call_count == count + 1
    changed['stage'] = 'closing'
    p._listening_rehearsal_materials(changed)
    assert p._retrieve_on_prepared_tree.call_args.args[0]['stage'] == 'closing'
    assert len(p._listening_rehearsal_materials(changed)) <= 3


@pytest.mark.parametrize('change_input', [False, True])
def test_audio_publication_checks_sources_without_recalling_private_material(audio, tmp_path, change_input):
    p, history, _ = prepared_player(tmp_path)
    enable(p)
    calls_before_playback = []
    published = []
    def delivered(index, path, text, duration):
        published.append(text)
        calls_before_playback.append(p._retrieve_on_prepared_tree.call_count)
        if change_input and index == 0:
            p.planner.chunks.append('New opponent condition after publication.')
    p.tts_chunk_callback = delivered
    if change_input:
        from streaming.flat_speaking import SegmentRejected
        with pytest.raises(SegmentRejected, match='Input changed after final handover'):
            p.rebuttal_generation(history, 60, time_control=True)
        assert published == [PREFIX]
    else:
        assert TAIL in p.rebuttal_generation(history, 60, time_control=True)
        assert published == [PREFIX, TAIL]
        assert len(set(calls_before_playback)) == 1


def test_no_opponent_target_retrieves_own_support_without_fabricating_targets(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    enable(p)
    data = material(p, p.status, history)
    data['current_targets'] = []
    data['our_main_claims'] = ['Our current claim.']
    p._retrieve_on_prepared_tree.reset_mock()
    result = p._listening_rehearsal_materials(data)
    action = p._retrieve_on_prepared_tree.call_args.args[0]
    assert action['action'] == 'reinforce' and action['targeted_debate_tree'] == 'you'
    assert result[0]['target_node_id'] is None
    p.use_rehearsal_tree = False
    assert p._listening_rehearsal_materials(data) == []


def test_listening_retrieval_never_silently_enables_paid_relation_calls(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    enable(p)
    p.config.rehearsal_mode = 'llm'
    with pytest.raises(ValueError, match='offline'):
        material(p, p.status, history)
    p._retrieve_on_prepared_tree.assert_not_called()


def test_opening_selection_uses_outline_while_retrieval_keeps_full_pool(tmp_path, monkeypatch):
    from ouragents import TreeDebater
    p, _, _ = prepared_player(tmp_path)
    enable(p)
    p.claim_pool = [[dict(claim='Saved outline without a tree.', arguments=[], minimax_search_score=1)]]
    p.definition = 'Motion definition.'
    p.claim_preparation = None
    p.config.claim_pool_limit = 10
    p.build_evidence_pool = Mock()
    p._get_prepared_tree = Mock(return_value=['full prepared pool'])
    p._warm_rehearsal_indexes = Mock()
    select = Mock(return_value=(['Saved outline without a tree.'], [0], {'mode': 'choose_main_claims'}))
    monkeypatch.setattr('ouragents.build_logic_claims', select)
    TreeDebater.claim_selection(p)
    select.assert_not_called()
    p.build_evidence_pool.assert_called_once()
    assert p.main_claims_content == ['Saved outline without a tree.']
    assert p.use_rehearsal_tree and p.prepared_tree_list == ['full prepared pool']
    p._warm_rehearsal_indexes.assert_called_once()
