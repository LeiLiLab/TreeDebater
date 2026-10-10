"""Real native delivery consumes a whole revision before its final token arrives."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import csv
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from streaming.config import OutputConfig
from streaming.revision_stream import RevisionStream
from test_full_speech import audio as audio
import tts_streaming as tts

pytestmark = pytest.mark.usefixtures('word_length_modes')

def stream(**kwargs):
    return RevisionStream(prefix='Opening.', draft='First source paragraph. Second source paragraph.',
        n_words=40, config=OutputConfig(min_chunk_words=1, max_chunk_chars=900), **kwargs)


def test_delta_boundaries_and_eof_keep_the_released_chunk_immutable():
    s = stream()
    s.feed('Opening.\n\nFirst complete paragraph.\n')
    assert s.read()[0] == []
    s.feed('\nSecond')
    chunks, done, unseen = s.read()
    assert chunks == ['First complete paragraph.'] and not done and unseen > 0
    s.feed(' complete paragraph.')
    s.finish('Opening.\n\nFirst complete paragraph.\n\nSecond complete paragraph.')
    assert s.read() == (['Second complete paragraph.'], True, 0)


@pytest.mark.parametrize('envelope', [lambda x: json.dumps({'speech': x}), lambda x: '```json\n'+json.dumps({'speech': x})+'\n```'])
def test_legacy_envelopes_wait_for_full_decode(envelope):
    s = stream()
    raw = envelope('Opening.\n\nFirst complete paragraph.\n\nLast complete paragraph.')
    for char in raw:
        s.feed(char)
        assert not s.read()[0]
    s.finish(raw)
    assert s.read() == (['First complete paragraph.', 'Last complete paragraph.'], True, 0)


def test_error_wakes_waiter_and_does_not_deliver_buffered_text():
    s = stream()
    with ThreadPoolExecutor(max_workers=1) as pool:
        job = pool.submit(s.read, wait=True)
        s.fail(RuntimeError('Provider interrupted'))
        with pytest.raises(RuntimeError, match='Provider interrupted'):
            job.result(timeout=2)
    with pytest.raises(RuntimeError, match='Provider interrupted'):
        s.feed('Late text.\n\n')


def test_changed_final_response_cannot_silently_replace_released_text():
    s = stream()
    s.feed('First complete paragraph.\n\n')
    with pytest.raises(ValueError, match='differs'):
        s.finish('Changed paragraph.')
    with pytest.raises(ValueError, match='differs'):
        s.read()


@pytest.mark.parametrize('first_body_cap', [0, 2])
def test_native_body_audio_precedes_final_text_and_reserves_time_for_unseen_tail(audio, tmp_path, first_body_cap):
    query, encoded = audio
    first = 'First complete argument has its own source and qualification.'
    last = 'Last complete argument uses a distinct piece of evidence.'
    cfg = OutputConfig(adaptive_delivery=True, normalize_seams=False, min_chunk_words=1,
        max_refinements=0, early_max_refinements=0, max_chunk_chars=900,
        speed_adjust_min=1, speed_adjust_max=1, first_body_chunk_seconds=first_body_cap)
    s = RevisionStream(prefix='Opening.', draft=first+'\n\n'+last, n_words=20, config=cfg)
    body_published = threading.Event()
    published = []
    def synthesize(client, text, *args, **kwargs):
        return encoded(2 if text=='Opening.' else 3 if text==first else 5)
    def producer():
        s.feed(first+'\n\n')
        assert body_published.wait(3), 'Body publication waited for the complete revision'
        s.feed(last)
        s.finish(first+'\n\n'+last)
    def publish(index, path, text, duration):
        published.append(text)
        if index == 1:
            assert not s.snapshot()['finished']
            body_published.set()
    query.side_effect = synthesize
    with ThreadPoolExecutor(max_workers=1) as pool:
        job = pool.submit(producer)
        try:
            text, _, duration = tts.convert_text_to_speech_streaming('Opening.', str(tmp_path/'speech.mp3'),
                10, config=cfg, tail_supplier=lambda:s, on_chunk=publish)
        finally:
            body_published.set()
        job.result(timeout=3)
    assert published == ['Opening.',first,last]
    assert text == '\n\n'.join(published)
    rows = list(csv.DictReader((tmp_path/'speech_chunks/chunk_profile.csv').open()))
    assert 0 < float(rows[1]['target_s']) < 8
    if first_body_cap:
        assert float(rows[1]['target_s']) == pytest.approx(first_body_cap)
    assert float(rows[2]['target_s']) == pytest.approx(5, abs=.003)
    assert sum(float(r['audio_seconds']) for r in rows) == pytest.approx(10, abs=.003)
    assert query.call_count == 3


def test_streamed_chunks_still_use_native_length_refinement(audio, tmp_path, monkeypatch):
    query, encoded = audio
    first, last = 'First '+'word '*9, 'Last '+'word '*9
    cfg = OutputConfig(adaptive_delivery=True, normalize_seams=False, min_chunk_words=1,
        max_refinements=3, early_max_refinements=3, max_parallel_tts=1,
        speed_adjust_min=1, speed_adjust_max=1, max_chunk_chars=900)
    s = RevisionStream(prefix='Opening.', draft=first+'\n\n'+last, n_words=20, config=cfg)
    s.finish(first+'\n\n'+last)
    query.side_effect = lambda client,text,*args,**kw: encoded(2 if text=='Opening.' else len(text.split()))
    monkeypatch.setattr(tts, '_estimate_duration', lambda text,**kw: len(text.split()))
    revise = Mock(side_effect=lambda client,text,n_words,*args,**kw: ' '.join(text.split()[:n_words]))
    monkeypatch.setattr(tts, '_revise_to_n_words', revise)
    text,_,_=tts.convert_text_to_speech_streaming('Opening.',str(tmp_path/'speech.mp3'),12,
        config=cfg,tail_supplier=lambda:s)
    assert len(text.split()) == 11
    assert revise.call_count >= 2
    rows=list(csv.DictReader((tmp_path/'speech_chunks/chunk_profile.csv').open()))
    assert all(int(r['used_candidate_iter'])>0 for r in rows[1:])
    assert sum(float(r['audio_seconds']) for r in rows)==pytest.approx(12)


def test_helper_stream_forwards_deltas_preserves_usage_and_closes(monkeypatch):
    from utils import model
    from litellm import ModelResponse
    events=[]
    chunks=[ModelResponse(stream=True, model='gpt-test', choices=[dict(index=0,delta=dict(content=x),finish_reason=None)])
            for x in ('First.\n\n','Last.')]
    chunks.append(ModelResponse(stream=True,model='gpt-test',choices=[dict(index=0,delta={},finish_reason='stop')],
                                usage=dict(prompt_tokens=10,completion_tokens=4,total_tokens=14)))
    class ResponseStream:
        def __iter__(self):
            for i,chunk in enumerate(chunks):
                if i==1: assert events==['First.\n\n']
                yield chunk
        close=Mock()
    response=ResponseStream()
    completion=Mock(return_value=response)
    monkeypatch.setattr(model.litellm,'completion',completion)
    result=model.HelperClient('Write.',model='gpt-test',json_mode=False,on_text=events.append)
    assert result==['First.\n\nLast.']
    assert events==['First.\n\n','Last.']
    response.close.assert_called_once()
    assert completion.call_args.kwargs['num_retries']==0
    assert completion.call_args.kwargs['stream_options']=={'include_usage':True}


@pytest.mark.parametrize('reason',[None,'length','content_filter'])
def test_helper_rejects_incomplete_stream_without_retry(monkeypatch,reason):
    from utils import model
    chunks=[SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content='Unfinished'),finish_reason=reason)])]
    completion=Mock(return_value=iter(chunks))
    monkeypatch.setattr(model.litellm,'completion',completion)
    with pytest.raises(ValueError,match='Speech stream'):
        model.HelperClient('Write.',model='gpt-test',json_mode=False,on_text=lambda x:None)
    completion.assert_called_once()


@pytest.mark.parametrize('fail_after_first',[False,True])
def test_listening_handoff_streams_body_before_full_revision_and_preserves_committed_text(audio,tmp_path,fail_after_first):
    from test_first_body_audio import player_with_handoff
    from test_listening_prefix import PREFIX, trace, prefix_helper
    query,encoded=audio
    player,history,handoff=player_with_handoff(tmp_path,encoded)
    first='The source limits its finding to the funded trial. '+ 'The funding condition remains essential. '*5
    first=first.strip()
    last='The separate delivery question needs a distinct answer. '+ 'We must retain that qualification. '*5
    last=last.strip()
    opening_published,body_published=threading.Event(),threading.Event()
    observed=[]
    def helper(*,prompt,**kw):
        if prompt.startswith('LISTENING PREFIX REVIEW:'):
            return prefix_helper(None)(prompt=prompt,**kw)
        if 'LISTENING WHOLE SPEECH FEEDBACK:' in prompt:
            assert opening_published.wait(3)
            return ['Retain the funding qualification.']
        assert kw.get('on_text') is not None
        kw['on_text'](first+'\n\n')
        assert body_published.wait(3), 'Native delivery waited for the revision to finish'
        if fail_after_first:
            raise RuntimeError('Stream disconnected')
        kw['on_text'](last)
        return [first+'\n\n'+last]
    def publish(index,path,text,seconds):
        observed.append(text)
        (opening_published if index==0 else body_published).set()
    player.helper_client=Mock(side_effect=helper)
    player.tts_chunk_callback=publish
    options=dict(time_control=True,listening_handoff=handoff,
                 listening_input_completion=lambda:history,
                 listening_recognized_input=lambda:history)
    if fail_after_first:
        with pytest.raises(RuntimeError,match='Stream disconnected'):
            player.rebuttal_generation(history,60,**options)
        assert observed==[PREFIX,first]
        assert trace(tmp_path)['committed_text']==PREFIX+'\n\n'+first
    else:
        assert player.rebuttal_generation(history,60,**options)=='\n\n'.join([PREFIX,first,last])
        assert observed==[PREFIX,first,last]
        saved=trace(tmp_path)
        assert saved['parallel_body_revision']['reused']
        assert saved['body_stream']['first_chunk_ready_seconds']<saved['body_stream']['text_complete_seconds']
