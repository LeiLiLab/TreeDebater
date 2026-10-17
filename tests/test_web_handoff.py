"""Web adapter -> native listening-prefix pipeline, real offline MP3 chunks."""
import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import sys
import threading
from unittest.mock import Mock

import pytest

app_root = os.environ.get('DEBATE_APP_ROOT')
if not app_root:
    pytest.skip('Set DEBATE_APP_ROOT to run the separate web app integration', allow_module_level=True)
backend = Path(app_root).expanduser().resolve() / 'backend'
if not (backend / 'debate_app' / 'engine_adapter.py').is_file():
    raise RuntimeError('DEBATE_APP_ROOT must point to the TreeDebater-app checkout')
os.environ.setdefault('DEBATE_ENGINE_ROOT', str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(backend))
from debate_app.engine_adapter import TreeDebaterEngine
from test_body_snapshot_delivery import audio as audio
from test_final_input_overlap import ready_handoff
from test_listening_prefix import PREFIX, TAIL, prepared_player, trace

pytestmark = pytest.mark.usefixtures('word_length_modes')


@pytest.mark.parametrize('stream_body', [False, True])
def test_web_adapter_publishes_prefix_before_asr_body_before_analysis(audio, tmp_path, stream_body):
    player, history, _ = prepared_player(tmp_path)
    _, encoded = audio
    ready_handoff(player, history, encoded, snapshot_delivery=True)
    player.streaming_output_config = replace(player.streaming_output_config,
        listening_parallel_endpoint_revision=True, listening_single_body_revision=True,
        listening_stream_body_revision=stream_body, min_chunk_words=1)
    player.high_quality_evidence_pool, player.used_evidence = [], set()
    engine = TreeDebaterEngine({'budgets': {'rebuttal': 60}}, tmp_path)
    engine.players = {player.side: player}
    engine.trees = Mock(return_value={'against': {'merged': True}})
    engine.open_input(input_id='recording-attempt', source_side='for', source_stage='opening',
                      target_side='against', target_stage='rebuttal')
    final = copy.deepcopy(history)
    final[-1]['recognized_content'] = final[-1]['content'] + ' FINAL_HEARD_QUALIFICATION.'
    final[-1]['content'] = 'Generated speech is deliberately different from what was heard.'
    expected = copy.deepcopy(final)
    expected[-1].update(content=expected[-1]['recognized_content'], tree_via_streaming=True)
    first, body, feedback = threading.Event(), threading.Event(), threading.Event()
    chunks = []
    base = player.helper_client
    second = 'The final exception preserves access for residents with existing commitments.'

    def helper(*, prompt, **kwargs):
        if 'LISTENING WHOLE SPEECH FEEDBACK:' in prompt:
            assert 'FINAL_HEARD_QUALIFICATION' in prompt
            assert 'Generated speech is deliberately different' not in prompt
            feedback.set()
            return [json.dumps(dict(points=[], other_corrections=['Retain the final qualification.']))]
        if kwargs.get('json_mode') is False:
            assert 'FINAL_HEARD_QUALIFICATION' in kwargs['history_messages'][-1]['content']
            if stream_body:
                kwargs['on_text'](TAIL + '\n\n')
                assert body.wait(5), 'Web adapter delayed streamed body until analysis finished'
                kwargs['on_text'](second)
            return [TAIL + '\n\n' + second]
        return base(prompt=prompt, **kwargs)

    player.helper_client = Mock(side_effect=helper)

    def emit(chunk):
        player.listen.assert_not_called()
        chunks.append(chunk)
        (first if chunk['index'] == 0 else body).set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(engine.generate_handoff, 'recording-attempt', 'against', 'rebuttal',
                             [], str(tmp_path), emit)
        try:
            assert first.wait(5), 'Prefix waited for complete ASR'
            assert not feedback.is_set() and not body.is_set()
            engine.recognition_ready('recording-attempt', final)
            final[-1]['recognized_content'] = 'Caller changed its history after publication.'
            assert body.wait(5), 'Body waited for mutable analysis state'
            assert not result.done()
            player.listen.assert_not_called()
            # Model the final observer update; the fixed speech task cannot change.
            player.planner.chunks = [expected[-1]['content']]
            engine.finish_input('recording-attempt')
            response = result.result(timeout=5)
        finally:
            engine.inputs.close()
    assert response['text'].startswith(PREFIX) and TAIL in response['text']
    assert chunks[0]['text'] == PREFIX
    assert [c['index'] for c in chunks] == list(range(len(chunks)))
    assert all(Path(c['path']).is_file() for c in chunks)
    player.listen.assert_called_once_with(expected)
    assert player.tts_chunk_callback is None
    saved = trace(tmp_path)
    assert saved['body_publication_mode'] == 'complete_asr_snapshot'
    assert saved['chunks'][1]['ready_seconds'] < saved['final_input_ready_seconds']
    assert saved['body_publication_snapshot']['context']['history'] == expected
