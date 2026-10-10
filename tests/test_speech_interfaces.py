"""Offline regressions for the six speech-interface audit findings."""
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from agents import Debater, HumanDebater
from streaming.config import OutputConfig
from streaming.experiment_client import BudgetedClient, BudgetExceeded
from streaming.listening_prefix import PrefixPreparation, material
from utils import constants, time_estimator
from utils.time_estimator import LengthEstimator
import tts_streaming as tts
from test_audio_probe_budget import AudioGuard
from test_full_speech import audio as audio
from test_listening_prefix import PREFIX, FRAMEWORK, prepared_player

pytestmark = pytest.mark.usefixtures('word_length_modes')


@pytest.mark.parametrize('mode', ['words', 'time', 'phonemes', 'syllables', 'fastspeech', 'openai'])
def test_estimator_preserves_scalar_and_batch_shapes(monkeypatch, mode):
    monkeypatch.setattr(LengthEstimator, 'count_phonemes', lambda text: 3)
    monkeypatch.setattr(LengthEstimator, 'count_syllables', lambda text: 3)
    monkeypatch.setitem(sys.modules, 'utils.fs_wrapper', SimpleNamespace(
        get_shared_wrapper=lambda **kw: SimpleNamespace(query_time=lambda items: [3.] * len(items))))
    estimator = LengthEstimator(mode, audio_duration=lambda text: 3.)
    single = estimator.query_time('One point.')
    assert isinstance(single, (int, float))
    assert estimator.query_time(['One point.']) == [single]
    assert estimator.query_time(('One point.', 'One point.')) == [single, single]
    assert estimator.query_time([]) == []
    assert estimator.query_time(['One point.'], mode='words') == [2]


def test_batch_input_is_validated_before_any_paid_call():
    synthesize = Mock(return_value=1.)
    estimator = LengthEstimator('openai', audio_duration=synthesize)
    with pytest.raises(TypeError, match='every content item'):
        estimator.query_time(['Valid.', None])
    synthesize.assert_not_called()


def test_openai_estimator_cannot_create_an_uninjected_client():
    with pytest.raises(ValueError, match='injected audio_duration'):
        LengthEstimator('openai')


@pytest.mark.parametrize('text', ['a' * 9000, 'word ' * 1800 + 'end.'])
def test_openai_estimation_covers_all_text_without_truncation(text):
    chunks = []
    def synthesize(chunk):
        chunks.append(chunk)
        return len(chunk) / 1000
    seconds = LengthEstimator('openai', audio_duration=synthesize).query_time(text)
    assert ''.join(chunks) == text
    assert all(0 < len(chunk) <= 4096 for chunk in chunks)
    assert seconds == pytest.approx(len(text) / 1000)


def test_openai_estimator_uses_renderer_factory_and_selected_model_voice(monkeypatch):
    monkeypatch.setattr(constants, 'TIME_MODE_FOR_STATEMENT', 'openai')
    clients, calls = [], []
    def factory():
        client = SimpleNamespace(close=Mock())
        clients.append(client)
        return client
    def synthesize(client, text, **kwargs):
        calls.append((client, text, kwargs))
        return {'audio_seconds': len(text) / 1000}
    monkeypatch.setattr(tts, 'OpenAI', factory)
    monkeypatch.setattr(tts, '_query_time_profiled', synthesize)
    cfg = OutputConfig(model='configured-model', voice='alloy')
    assert tts.estimate_statement_seconds('a' * 5000, cfg) == 5
    assert ''.join(text for _, text, _ in calls) == 'a' * 5000
    assert all(kwargs == {'voice': 'alloy', 'model': 'configured-model'} for _, _, kwargs in calls)
    assert all(client in clients for client, _, _ in calls)
    assert len(clients) == 2
    for client in clients:
        client.close.assert_called_once()


def test_openai_estimation_reuses_current_pipeline_client_and_voice_override(monkeypatch):
    monkeypatch.setattr(constants, 'TIME_MODE_FOR_STATEMENT', 'openai')
    factory = Mock(side_effect=AssertionError('Unexpected new client'))
    monkeypatch.setattr(tts, 'OpenAI', factory)
    query = Mock(return_value={'audio_seconds': 2.})
    monkeypatch.setattr(tts, '_query_time_profiled', query)
    client = object()
    assert tts._estimate_duration('One point.', client=client,
        config=OutputConfig(model='configured-model', voice='echo'), voice='alloy') == 2
    query.assert_called_once_with(client, 'One point.', voice='alloy', model='configured-model')
    factory.assert_not_called()


def test_openai_estimation_cannot_bypass_audio_budget_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(constants, 'TIME_MODE_FOR_STATEMENT', 'openai')
    ledger = BudgetedClient(tmp_path)
    provider_calls = []
    def provider(request):
        provider_calls.append(json.loads(request.read()))
        return httpx.Response(200, content=b'fake-audio')
    guard = AudioGuard(tmp_path, 'estimate', httpx.MockTransport(provider), allowance=.3)
    http = httpx.Client(transport=guard)
    closed = []
    def create(**kwargs):
        response = http.post('https://api.openai.com/v1/audio/speech', json=kwargs)
        response.raise_for_status()
        return SimpleNamespace(content=response.content)
    monkeypatch.setattr(tts, 'OpenAI', lambda: SimpleNamespace(
        audio=SimpleNamespace(speech=SimpleNamespace(create=create)), close=lambda: closed.append(True)))
    monkeypatch.setattr(tts, 'MP3', lambda _: SimpleNamespace(info=SimpleNamespace(length=1.)))
    try:
        with pytest.raises(BudgetExceeded):
            tts.estimate_statement_seconds('a' * 9000, OutputConfig(voice='alloy'))
        assert len(provider_calls) == 1
        assert provider_calls[0]['voice'] == 'alloy'
        assert len(provider_calls[0]['input']) == 4096
        assert guard.artifact['blocked_dispatches']
        assert len(closed) == 2
        assert ledger.summary()['accounted_exposure_usd'] <= .3
    finally:
        guard.finish()
        http.close()
        ledger.db.close()


@pytest.mark.parametrize('synthesize', [tts._query_time_profiled, tts._tts_with_retry])
def test_low_level_tts_rejects_oversized_text_instead_of_silently_truncating(monkeypatch, synthesize):
    client = Mock()
    monkeypatch.setattr(tts.time, 'sleep', Mock(side_effect=AssertionError('Invalid input must not retry')))
    with pytest.raises(ValueError, match='split it before synthesis'):
        synthesize(client, 'a' * 4097)
    client.audio.speech.create.assert_not_called()


def test_phoneme_counter_reuses_one_instance_and_serializes_concurrent_calls(monkeypatch):
    created, active = [], []
    class G2p:
        def __init__(self):
            created.append(self)
        def __call__(self, text):
            active.append(True)
            try:
                assert len(active) == 1
                time.sleep(.002)
                return ['W', ' ', 'ER', 'D']
            finally:
                active.pop()
    monkeypatch.setattr(time_estimator, '_G2P', None)
    monkeypatch.setitem(sys.modules, 'g2p_en', SimpleNamespace(G2p=G2p))
    with ThreadPoolExecutor(max_workers=4) as workers:
        result = list(workers.map(LengthEstimator.count_phonemes, ['A word.'] * 12))
    assert result == [3] * 12
    assert len(created) == 1


@pytest.mark.parametrize('kind', [Debater, HumanDebater])
@pytest.mark.parametrize('stage', ['opening', 'rebuttal', 'closing'])
def test_remaining_generation_entries_honor_draft_mode(monkeypatch, kind, stage):
    monkeypatch.setattr(constants, 'LENGTH_MODE_FOR_DRAFT', 'phonemes')
    player = SimpleNamespace(motion='A motion', act='SUPPORT', counter_act='OPPOSE',
        listen=Mock(), speak=Mock(return_value='Speech.'),
        get_multiline_input=Mock(return_value='Speech.'), post_process=Mock(return_value='Speech.'))
    method = getattr(kind, stage + '_generation')
    history = [{'content': 'Opponent speech.'}]
    if kind is HumanDebater and stage == 'opening':
        method(player, max_time=46)
    else:
        method(player, history, max_time=46)
    call = player.get_multiline_input if kind is HumanDebater else player.speak
    assert 'approximately 450 phonemes' in call.call_args.args[0]


@pytest.mark.parametrize('stage,budget,needs_fit', [
    ('opening', 88, False), ('rebuttal', 12, False), ('closing', 5, True)])
def test_preparatory_stage_budget_matches_prompt_payload_and_guard(tmp_path, stage, budget, needs_fit):
    player, history, _ = prepared_player(tmp_path)
    player.speech_budgets = dict(opening=46, rebuttal=11, closing=7.5)
    data = material(player, stage, history)
    helper = Mock(return_value=[json.dumps({'draft': 'One two three four five six.'})])
    prep = PrefixPreparation(data['turn'], helper, OutputConfig(listening_body_words=1))
    candidate = {'text': PREFIX, 'framework': FRAMEWORK, 'stage': stage, 'turn': data['turn']}
    prep._latest = candidate
    prep._body_pending = (data, candidate)
    try:
        prep._body_run()
        prompt = helper.call_args_list[0].kwargs['prompt']
        assert f'approximately {budget} words' in prompt
        assert 'IMMUTABLE SPOKEN PREFIX' in prompt and PREFIX in prompt
        assert prep._body_latest is not None
        assert prep._body_latest['needs_fit'] is needs_fit
        assert helper.call_count == 1
    finally:
        prep.close()


def test_worker_signals_estimator_error_and_preserves_original_cause(monkeypatch):
    error = OSError('FastSpeech checkpoint missing')
    monkeypatch.setattr(tts, '_estimate_duration', Mock(side_effect=error))
    ctx = tts._ChunkRefineContext(client=None, original_text='One point.', target_s=2,
        tol_s=.1, tol_upper_s=.1, prev_texts=[], next_chunk_text='', voice='echo',
        max_ref=0, kickoff_iter=0, kickoff_kind='', config=OutputConfig(max_parallel_tts=1))
    try:
        tts._refine_worker(ctx, 'normal')
        assert ctx.done_event.is_set() and ctx.stop_event.is_set()
        assert not ctx.candidates
        with pytest.raises(RuntimeError, match='FastSpeech checkpoint missing') as caught:
            ctx.raise_estimation_error()
        assert caught.value.__cause__ is error
    finally:
        ctx.executor.shutdown(wait=True)


def test_pipeline_propagates_worker_estimator_failure_after_first_audio(audio, monkeypatch, tmp_path):
    error = OSError('FastSpeech checkpoint missing')
    monkeypatch.setattr(tts, '_estimate_duration', Mock(side_effect=error))
    cfg = OutputConfig(adaptive_delivery=False, min_chunk_words=1, max_refinements=0,
        early_max_refinements=0, max_parallel_tts=1, normalize_seams=False,
        speed_adjust_min=1, speed_adjust_max=1)
    delivered = []
    with pytest.raises(RuntimeError, match='FastSpeech checkpoint missing') as caught:
        tts.convert_text_to_speech_streaming('First point.\n\nSecond point.', str(tmp_path/'speech.mp3'),
            4, config=cfg, on_chunk=lambda *args: delivered.append(args))
    assert len(delivered) == 1
    assert caught.value.__cause__ is error
