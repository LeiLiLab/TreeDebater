"""Publication and common preparation regressions; every provider is mocked."""
import copy
import json
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest

from streaming.config import OutputConfig, from_mapping
from streaming.flat_speaking import SegmentRejected
from streaming.listening_prefix import material, prepare
from test_final_input_overlap import ready_handoff
from test_full_speech import audio as audio
from test_listening_prefix import PREFIX, TAIL, conflict, prefix_helper, prepared_player, trace
from test_overview_review_gate import passing

pytestmark = pytest.mark.usefixtures('word_length_modes')


def test_v46_actual_stance_reversal_cannot_publish_despite_matching_framework(audio, tmp_path):
    record = json.loads((Path(__file__).parent / 'fixtures/listening_v46_stance_reversal.json').read_text())
    wrong = json.loads(record['prefix_response'])
    p, history, _ = prepared_player(tmp_path)
    p.motion = record['authoring_input']['motion']
    p.streaming_output_config = replace(p.streaming_output_config, first_chunk_seconds=20)
    assert wrong['framework']['position'] == p.side == 'against'
    wrong['draft'] = 'This body cannot repair an already spoken reversed opening.'
    reviewed = []
    def helper(*, prompt, **kwargs):
        if prompt.startswith('LISTENING PREFIX REVIEW:'):
            payload = json.loads(prompt.rsplit('\n', 1)[-1])
            assert payload['draft'] == wrong['text']
            reviewed.append(payload)
            # Even an inconsistent all-true flag cannot override the reviewer's
            # classification of the actual endorsed stance as the other side.
            return [json.dumps(passing(wrong['text'], 'for'))]
        return [json.dumps(wrong)]
    p.helper_client = Mock(side_effect=helper)
    p.tts_chunk_callback = Mock()
    with pytest.raises(SegmentRejected, match='publication review'):
        p.rebuttal_generation(history, 60, time_control=True)
    assert len(reviewed) == 2
    assert p.helper_client.call_count == 4  # Draft/review + one joint repair/review.
    audio[0].assert_not_called()
    p.tts_chunk_callback.assert_not_called()
    assert trace(tmp_path)['chunks'] == []


def test_handoff_publishes_snapshot_then_body_receives_final_qualification(audio, tmp_path):
    p, history, _ = prepared_player(tmp_path)
    _, handoff = ready_handoff(p, history, audio[1])
    final = copy.deepcopy(history)
    final[-1]['content'] += ' Funding is now guaranteed.'
    published = threading.Event()

    def complete():
        assert published.wait(3), 'Final input blocked prepared prefix'
        p.planner.chunks = [final[-1]['content']]
        return final

    def recognized():
        assert published.wait(3), 'ASR blocked prepared prefix'
        return final

    def publish(index, path, text, duration):
        if index == 0:
            assert text == PREFIX
            p.listen.assert_not_called()
            published.set()

    p.tts_chunk_callback = publish
    p.rebuttal_generation(history, 60, time_control=True, listening_handoff=handoff,
        listening_recognized_input=recognized, listening_input_completion=complete)
    saved = trace(tmp_path)
    assert saved['prefix_input_scope'] == 'listening_snapshot'
    assert saved['endpoint_reviews'] == []
    assert saved['prepared_audio']['text'] == PREFIX
    assert saved['chunks'][0]['text'] == PREFIX
    assert 'Funding is now guaranteed.' in p._get_revision_suggestion.call_args.kwargs['history'][-1]['content']


def test_prepared_prefix_review_is_reused_only_for_identical_text_and_sources(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    from streaming.overview_review import review_stamp
    data = material(p, p.status, history)
    candidate = prepare(data, p.helper_client, p.streaming_output_config)
    assert candidate['review_stamp'] == review_stamp(candidate, data)
    changed = dict(candidate, text='We support this trial without independent costing.')
    assert review_stamp(changed, data) != candidate['review_stamp']
    changed_data = dict(data, final_transcript='The costings are now independent.')
    assert review_stamp(candidate, changed_data) != candidate['review_stamp']


@pytest.mark.parametrize('rehearsal', [False, True])
def test_standard_pool_loading_prepares_claims_evidence_and_recall_once(tmp_path, rehearsal):
    p, history, _ = prepared_player(tmp_path)
    p.config.claim_pool_limit = 1
    p.use_rehearsal_tree, p.use_retrieval = rehearsal, False
    p._get_prepared_tree = Mock(return_value=['saved trees'])
    p._warm_rehearsal_indexes = Mock()
    pools = [[dict(claim=name, definition='Saved scope.', minimax_search_score=score,
                   arguments=[dict(title=name, content=name + ' documented evidence.', reliability=2)])]
             for name, score in [('Low', 1), ('High', 5)]]
    path = tmp_path / 'pool_against.json'
    path.write_text(json.dumps(pools))
    (tmp_path / 'pool_for.json').write_text(json.dumps(pools))
    p.pool_file = str(path)
    p.helper_client = Mock(side_effect=AssertionError('Saved preparation must not call a model'))
    p.claim_generation(4)
    assert p.main_claims_content == ['High']
    assert p.high_quality_evidence_pool[0]['content'] == 'High documented evidence.'
    assert p.rehearsal_claim_pool == pools
    prepared = copy.deepcopy(p.claim_preparation)
    p.build_evidence_pool = Mock(side_effect=AssertionError('Evidence prepared twice'))
    p.claim_selection(history)
    assert p.claim_preparation == prepared
    assert p._warm_rehearsal_indexes.call_count == int(rehearsal)
    assert len([x for x in p.debate_thoughts if x['mode'] == 'choose_main_claims']) == 1


def test_generated_pool_uses_same_common_selection(tmp_path):
    p, _, _ = prepared_player(tmp_path)
    p.claim_preparation = None
    p.definition = 'Generated scope.'
    p.config.claim_pool_limit = 2
    p.use_rehearsal_tree = p.use_retrieval = False
    p.claim_pool = [[dict(claim='Generated proposal', arguments=[])]]
    p.claim_selection()
    assert p.main_claims_content == ['Generated proposal']
    assert p.claim_preparation['claims'][0]['minimax_search_score'] is None


@pytest.mark.parametrize('mode', ['listening_prefix', 'overlap_prefix', 'incremental'])
def test_nonflat_speech_configuration_is_rejected_instead_of_ignored(tmp_path, mode):
    p, history, _ = prepared_player(tmp_path)
    p.planner.config.mode = 'linear'
    p.streaming_output_config = OutputConfig(speech_mode=mode)
    with pytest.raises(ValueError, match='requires planning mode flat_tree'):
        p.rebuttal_generation(history, 60, time_control=True)
    p._get_response.assert_not_called()


def test_removed_review_switch_fails_clearly():
    with pytest.raises(ValueError, match='verify_rewrites'):
        from_mapping(OutputConfig, {'verify_rewrites': True})


@pytest.mark.parametrize('parallel_feedback', [False, True])
def test_final_asr_failure_stops_body_after_prepared_prefix(audio, tmp_path, parallel_feedback):
    p, history, _ = prepared_player(tmp_path)
    _, handoff = ready_handoff(p, history, audio[1])
    p.streaming_output_config = replace(p.streaming_output_config,
        listening_parallel_body_feedback=parallel_feedback)
    published = threading.Event()
    p.tts_chunk_callback = lambda *args: published.set()

    def fail_input():
        assert published.wait(3), 'Final ASR blocked prepared prefix'
        raise RuntimeError('Final ASR unavailable')

    with pytest.raises(RuntimeError, match='Final ASR unavailable'):
        p.rebuttal_generation(history, 60, time_control=True, listening_handoff=handoff,
            listening_recognized_input=fail_input, listening_input_completion=fail_input)
    saved = trace(tmp_path)
    assert saved['status'] == 'failed'
    assert len(saved['chunks']) == 1 and saved['committed_text'] == PREFIX
    p.listen.assert_not_called()
    p._get_revision_suggestion.assert_not_called()
