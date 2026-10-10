"""Regressions from v49: oversized sources, rejected IDs and duplicate revision."""
import copy
import json
import threading
from dataclasses import replace
from pathlib import Path
from types import MethodType
from unittest.mock import Mock

import pytest

from ouragents import TreeDebater
from streaming.listening_prefix import material, prepare
from streaming.listening_prefix import PrefixFormatRejected
from test_final_input_overlap import ready_handoff
from test_full_speech import audio as audio
from test_listening_prefix import PREFIX, TAIL, prepared_player, trace

pytestmark = pytest.mark.usefixtures('word_length_modes')


def test_planning_and_writer_reuse_native_top_ten_without_raw_documents(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    p.main_claims = ['claim']
    evidence = [dict(id=str(i), title=f'Source {i}', content=f'EXCERPT_{i:02}_END',
                     raw_content=f'RAW_DOCUMENT_{i}', reliability=20-i) for i in range(15)]
    p._get_evidence = Mock(return_value=evidence)
    p.build_evidence_pool()
    assert len(p.high_quality_evidence_pool) == 15 and len(p.evidence_pool) == 10
    context = p._planning_context()
    assert [e['id'] for e in context['evidence']] == [e['id'] for e in p.evidence_pool]
    assert all('raw_content' not in e for e in context['evidence'])
    data = material(p, p.status, history)
    assert data['supplied_evidence'] == [e['content'] for e in p.evidence_pool]
    prepare(data, p.helper_client, p.streaming_output_config)
    prompt = p.helper_client.call_args_list[0].kwargs['prompt']
    for e in p.evidence_pool:
        assert prompt.count(e['content']) == 1
    for e in p.high_quality_evidence_pool[10:]:
        assert e['content'] not in prompt


def test_handoff_ignores_legacy_writer_target_ids_but_still_reviews_actual_text(tmp_path):
    p, history, _ = prepared_player(tmp_path)
    p.streaming_output_config = replace(p.streaming_output_config, listening_prefix_overlap_final_update=True)
    # The existing helper emits nonempty IDs, reproducing the v49 failure.
    candidate = prepare(material(p, p.status, history), p.helper_client, p.streaming_output_config)
    assert 'target_ids' not in candidate
    assert candidate['audits'][-1]['semantic_review'] and candidate['audits'][-1]['accepted']
    assert p.helper_client.call_count == 2


@pytest.mark.parametrize('repair_succeeds', [True, False])
def test_recorded_truncated_closing_gets_only_one_repair_before_publication(tmp_path, repair_succeeds):
    p, history, _ = prepared_player(tmp_path)
    record = json.loads((Path(__file__).parent/'fixtures/listening_v50_truncated_closing.json').read_text())
    base = p.helper_client
    def helper(**kwargs):
        prompt = kwargs['prompt']
        if prompt.startswith('LISTENING PREFIX DRAFT:'):
            return [record['response']]
        if prompt.startswith('LISTENING PREFIX REPAIR:'):
            assert 'Malformed speech segment JSON' in prompt
            assert 'Wait, the provided' not in prompt  # Do not recirculate the repetitive malformed output.
            if not repair_succeeds:
                return [record['response']]
        return base(**kwargs)
    checked = Mock(side_effect=helper)
    if repair_succeeds:
        result = prepare(material(p, p.status, history), checked, p.streaming_output_config)
        assert result['text'] == PREFIX and result['audits'][-1]['semantic_review']
        assert checked.call_count == 3
    else:
        with pytest.raises(PrefixFormatRejected):
            prepare(material(p, p.status, history), checked, p.streaming_output_config)
        assert checked.call_count == 2


@pytest.mark.parametrize('feedback', ['Explain the selected evidence.', 'No changes'])
@pytest.mark.parametrize('parallel', [True, False])
def test_prepared_evidence_reuse_and_revision_run_once_across_handoff(audio, tmp_path, monkeypatch, feedback, parallel):
    import tts_streaming
    # The body already fits the remaining 58 seconds: evidence alone must be
    # enough to trigger revision when the review reports no corrections.
    monkeypatch.setattr(tts_streaming, 'estimate_statement_seconds', lambda *a, **kw: 58)
    p, history, _ = prepared_player(tmp_path)
    _, handoff = ready_handoff(p, history, audio[1])
    handoff['candidate']['body_preparation'] = dict(draft=TAIL, feedback=None,
        prefix_text=PREFIX, stage=p.status, turn=p.planner.turn)
    p.streaming_output_config = replace(p.streaming_output_config,
        listening_parallel_endpoint_revision=parallel, listening_single_body_revision=True)
    p._get_revision_suggestion = MethodType(TreeDebater._get_revision_suggestion, p)
    p._length_adjust = MethodType(TreeDebater._length_adjust, p)
    p.high_quality_evidence_pool = [dict(id=f'e{i}', content=f'EVIDENCE_{i}', raw_content='RAW') for i in range(12)]
    # The cold-start reuse path retains initial sources, filtering already-used
    # IDs and documents that no longer match the current candidate pool.
    p.evidence_pool = [p.high_quality_evidence_pool[0], p.high_quality_evidence_pool[2],
                       dict(p.high_quality_evidence_pool[3], content='OBSOLETE_CONTENT')]
    p.used_evidence = {'e0'}
    p.config.temperature, p.config.max_tokens = .37, 2048
    first, revised = threading.Event(), threading.Event()
    selections, revisions = [], []
    revised_tail = TAIL + ' Source Two confirms the cost estimate.'
    original_helper = p.helper_client
    def helper(**kwargs):
        prompt = kwargs['prompt']
        if prompt.startswith('LISTENING PREFIX REVIEW:'):
            return original_helper(**kwargs)
        if 'LISTENING WHOLE SPEECH FEEDBACK:' in prompt:
            assert first.wait(3)
            return [feedback]
        if prompt.startswith('From the provided list of evidence dictionaries'):
            assert 'EVIDENCE_0"' not in prompt and 'RAW' not in prompt
            selections.append(prompt)
            return [json.dumps(dict(selected_ids=['e2']))]
        if kwargs.get('json_mode') is False:
            assert 'EVIDENCE_2' in prompt and 'RAW' not in prompt
            assert kwargs['temperature'] == .37 and kwargs['max_tokens'] == 2048
            revisions.append(prompt)
            revised.set()
            return [revised_tail]
        raise AssertionError(prompt[:100])
    p.helper_client = Mock(side_effect=helper)
    p.tts_chunk_callback = lambda *args: first.set()
    def complete():
        if parallel:
            assert revised.wait(3), 'Evidence selection/revision waited for final tree analysis'
        assert p.used_evidence == {'e0'}, 'Speculation mutated committed evidence use'
        return copy.deepcopy(history)
    result = p.rebuttal_generation(history, 60, time_control=True, listening_handoff=handoff,
        listening_input_completion=complete, listening_recognized_input=lambda: copy.deepcopy(history))
    assert result == PREFIX + '\n\n' + revised_tail
    assert not selections
    assert len(revisions) == 1
    assert p.used_evidence == {'e0', 'e2'}
    saved = trace(tmp_path)
    assert saved['prepared_evidence']['mode'] == 'reuse_only'
    assert saved['prepared_evidence']['selected_ids'] == ['e2']
    if parallel:
        assert saved['parallel_body_revision']['reused']
        assert saved['parallel_evidence_selection']['selected_ids'] == ['e2']
    else:
        assert 'parallel_body_revision' not in saved


def test_deferred_body_uses_native_citation_cleanup_before_tts(audio, tmp_path):
    import tts_streaming as tts
    from streaming.config import OutputConfig
    body = 'The evidence supports independent oversight [1_8].'
    received = []
    _, references, _ = tts.convert_text_to_speech_streaming(PREFIX, tmp_path/'speech.mp3', 60,
        config=OutputConfig(adaptive_delivery=True, max_refinements=0, early_max_refinements=0,
                            normalize_seams=False, speed_adjust_min=1, speed_adjust_max=1),
        tail_supplier=lambda: body + '\n\n**References**\n[1_8] Source title, 2025.',
        on_chunk=lambda i, path, text, duration: received.append(text))
    assert received == [PREFIX, 'The evidence supports independent oversight .']
    assert 'Source title' in references
    assert 'References' not in ' '.join(received) and '[1_8]' not in ' '.join(received)
