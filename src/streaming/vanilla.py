"""Unplanned token-to-speech pipeline. No rewriting, fitting or model retries."""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
from queue import Queue, Full
import re
import threading
import time

from pydub import AudioSegment


class Segmenter:
    """Prefer sentence boundaries; bound punctuation-free output as well."""
    def __init__(self, words, chars):
        self.words, self.chars, self.pending = words, chars, ''

    def feed(self, delta, final=False):
        self.pending += delta
        while self.pending:
            end = None
            for match in re.finditer(r'[.!?][\"\u201d\u2019\']?(?:\s+|$)', self.pending):
                # At a token boundary, a trailing period may be part of a decimal.
                if match.end() == len(self.pending) and not final and not self.pending[-1].isspace():
                    continue
                if len(self.pending[:match.end()].split()) >= self.words:
                    end = match.end()
                    break
            if len(self.pending) >= self.chars and (end is None or end > self.chars):
                end = self.pending.rfind(' ', 0, self.chars + 1)
                if end <= 0:
                    end = self.chars
            if end is None:
                if not final:
                    break
                end = len(self.pending)
            part, self.pending = self.pending[:end], self.pending[end:]
            if part.strip():
                yield part.strip()


def generate(agent, messages, on_text, options):
    """Use the configured client so endpoint routing and metering are preserved."""
    import litellm
    stream = agent.client(messages=messages, stream=True, num_retries=0,
                          stream_options={'include_usage': True}, **options)
    chunks, finished = [], False
    try:
        for chunk in stream:
            chunks.append(chunk)
            for choice in chunk.choices:
                if choice.finish_reason is not None:
                    if choice.finish_reason != 'stop':
                        raise ValueError(f'Vanilla stream ended with {choice.finish_reason}')
                    finished = True
                text = getattr(choice.delta, 'content', None)
                if text:
                    on_text(text)
        if not finished:
            raise ValueError('Vanilla stream missing completion marker')
    finally:
        close = getattr(stream, 'close', None)
        if close:
            close()
        cost = next((getattr(c, '_hidden_params', {}).get('response_cost') for c in reversed(chunks)
                     if getattr(c, '_hidden_params', {}).get('response_cost') is not None), None)
        if cost is None and chunks:
            try:
                response = litellm.stream_chunk_builder(chunks, messages=messages)
                cost = litellm.completion_cost(completion_response=response, model=agent.config.model)
            except Exception:
                pass
        if cost is not None:
            with agent._cost_lock:
                agent.client_cost += cost


def speak(agent, prompt, config, **kwargs):
    from tts_streaming import create_tts_client, synthesize_audio
    if kwargs.get('n', 1) != 1:
        raise ValueError('Streaming Vanilla requires n=1')
    if '_completion' in kwargs:
        raise ValueError('Use _stream_completion(messages, on_text, **options) for a metered stream')
    completion = kwargs.pop('_stream_completion', None)
    budget = kwargs.pop('max_time', -1)
    for key in ('time_control', 'streaming_tts', 'streaming_listen', 'history', 'max_words', '_request_context'):
        kwargs.pop(key, None)
    for key in ('stream', 'stream_options', 'num_retries'):
        kwargs.pop(key, None)
    path = Path(agent._speech_audio_file())
    directory = path.parent / (path.stem + '_chunks')
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.glob('chunk_*.mp3')) or (directory / 'vanilla_streaming.json').exists():
        raise FileExistsError(f'Speech artifacts already exist: {directory}')
    client = create_tts_client(config)
    queue, cancelled = Queue(maxsize=2), threading.Event()
    start = time.perf_counter()
    trace = dict(mode='vanilla_streaming', status='running', target_seconds=budget,
                 chunks=[], generation_started_monotonic=start, raw_text='')
    segmenter = Segmenter(config.vanilla_chunk_words, config.vanilla_max_chunk_chars)
    agent._add_message('user', prompt)

    def save():
        tmp = directory / 'vanilla_streaming.json.tmp'
        tmp.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(directory / 'vanilla_streaming.json')

    def publish():
        combined = AudioSegment.empty()
        try:
            while not cancelled.is_set():
                text = queue.get()
                if text is None or cancelled.is_set():
                    break
                output = synthesize_audio(client, text, config, retry=False)
                audio = AudioSegment.from_file(BytesIO(output['mp3_bytes']), format='mp3')
                if cancelled.is_set():
                    break
                index = len(trace['chunks'])
                chunk = directory / f'chunk_{index:03d}.mp3'
                temporary = chunk.with_suffix('.tmp')
                temporary.write_bytes(output['mp3_bytes'])
                chunk.with_suffix('.txt').write_text(text, encoding='utf-8')
                temporary.replace(chunk)
                combined += audio
                seconds = len(audio) / 1000
                trace['chunks'].append(dict(index=index, path=str(chunk), text=text,
                    audio_seconds=seconds, ready_seconds=time.perf_counter() - start))
                save()
                callback = getattr(agent, 'tts_chunk_callback', None)
                if callback:
                    callback(index, str(chunk), text, seconds)
        finally:
            if len(combined):
                combined.export(path, format='mp3')

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='vanilla_tts') as pool:
        worker = pool.submit(publish)

        def put(value):
            while True:
                if worker.done():
                    worker.result()
                    raise RuntimeError('TTS worker stopped before generation')
                try:
                    queue.put(value, timeout=.05)
                    return
                except Full:
                    continue

        def on_text(delta):
            trace['raw_text'] += delta
            for text in segmenter.feed(delta):
                put(text)

        try:
            if completion:
                completion(messages=agent.conversation, on_text=on_text, **kwargs)
            else:
                generate(agent, agent.conversation, on_text, kwargs)
            trace['generation_finished_seconds'] = time.perf_counter() - start
            for text in segmenter.feed('', final=True):
                put(text)
            put(None)
            worker.result()
            if not trace['chunks']:
                raise ValueError('Vanilla generated no speech')
            trace['status'] = 'completed'
        except BaseException as exc:
            cancelled.set()
            # Wake an idle consumer without blocking on a failed worker.
            try:
                queue.put_nowait(None)
            except Full:
                pass
            try:
                worker.result()
            except Exception:
                pass
            trace.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            trace['committed_text'] = '\n\n'.join(c['text'] for c in trace['chunks'])
            trace['audio_seconds'] = sum(c['audio_seconds'] for c in trace['chunks'])
            trace['wall_seconds'] = time.perf_counter() - start
            if budget > 0:
                trace['signed_duration_error_seconds'] = trace['audio_seconds'] - budget
            save()
            if trace['committed_text']:
                agent._add_message('assistant', trace['committed_text'])
    return trace['committed_text']
