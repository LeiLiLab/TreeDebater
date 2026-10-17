"""Whole-script TTS transport with exact text-to-audio parts and measured trimming.

Parts are provider request limits, not incremental authorship. Nothing is
published until the complete (possibly shortened) script has finished synthesis.
"""
from io import BytesIO
import json
from pathlib import Path
import re

from pydub import AudioSegment


def split_text(text, limit=4096):
    if not text.strip():
        raise ValueError('Cannot synthesize empty text')
    parts = []
    while len(text) > limit:
        candidates = [m.end() for m in re.finditer(r'[.!?][\"\u201d\u2019\']?(?:\s+|$)', text[:limit])]
        end = candidates[-1] if candidates else text.rfind(' ', 0, limit)
        if end <= 0:
            end = limit
        parts.append(text[:end])
        text = text[end:]
    if text:
        parts.append(text)
    return parts


def render(client, text, voice='echo'):
    response = client.audio.speech.create(model='tts-1', voice=voice, input=text, response_format='mp3')
    return AudioSegment.from_file(BytesIO(response.content), format='mp3')


def local_render(client, text, voice='echo'):
    """FastSpeech2 + local vocoder; no provider client or fallback."""
    from utils.fs_wrapper import get_shared_wrapper
    waveform, sample_rate = get_shared_wrapper().synthesize_waveform(text)
    return AudioSegment(waveform.astype('<i2').tobytes(), frame_rate=sample_rate,
                        sample_width=2, channels=1)


def synthesis_options(backend):
    if backend == 'fastspeech':
        return None, local_render
    if backend == 'openai':
        from openai import OpenAI
        return OpenAI(max_retries=0), None
    raise ValueError('Unknown TTS backend: ' + str(backend))


def metadata_path(path):
    return Path(str(path) + '.parts.json')


def write_audio(audio, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with audio.export(str(path), format='mp3', bitrate='192k'):
        pass


def synthesize(client, text, path, voice='echo', *, renderer=None):
    render_audio = renderer or render
    path = Path(path)
    directory = path.with_suffix('.parts')
    directory.mkdir(parents=True, exist_ok=True)
    combined, records = AudioSegment.empty(), []
    for i, part in enumerate(split_text(text)):
        audio = render_audio(client, part, voice)
        part_path = directory / f'{i:03}.mp3'
        write_audio(audio, part_path)
        records.append(dict(text=part, path=str(part_path.resolve()), duration_ms=len(audio)))
        combined += audio
    write_audio(combined, path)
    metadata_path(path).write_text(json.dumps(dict(text=text, voice=voice, parts=records), ensure_ascii=False, indent=2))
    return len(combined) / 1000


def trim_measured(client, input_path, output_path, max_duration, *, renderer=None):
    """Keep complete request parts, then re-synthesize a sentence prefix to fit.

    Never infer text alignment from audio duration, call ASR, or claim unplayed
    words were delivered. The bounded search may underfill a budget; it cannot
    exceed it. Saved accepted audio is used verbatim, not synthesized again.
    """
    render_audio = renderer or render
    if max_duration <= 0:
        raise ValueError('Audio budget must be positive')
    metadata = json.loads(metadata_path(input_path).read_text())
    combined, texts, records = AudioSegment.empty(), [], []
    for item in metadata['parts']:
        audio = AudioSegment.from_file(item['path'], format='mp3')
        if len(combined) + len(audio) <= max_duration:
            combined += audio
            texts.append(item['text'])
            records.append(item)
            continue
        remaining = max_duration - len(combined)
        ends = [m.end() for m in re.finditer(r'[.!?][\"\u201d\u2019\']?(?=\s|$)', item['text'])]
        # Every candidate is re-synthesized and measured. No approximate audio slicing.
        lo, hi, attempts, best = 0, len(ends)-1, 0, None
        while lo <= hi and attempts < 12:
            mid = (lo + hi)//2
            candidate = item['text'][:ends[mid]]
            rendered = render_audio(client, candidate, metadata['voice'])
            attempts += 1
            if len(rendered) <= remaining:
                best = (candidate, rendered)
                lo = mid + 1
            else:
                hi = mid - 1
        if best:
            candidate, rendered = best
            combined += rendered
            texts.append(candidate)
            records.append(dict(text=candidate, duration_ms=len(rendered), resynthesized=True))
        break
    if not texts or len(combined) <= 0:
        raise ValueError('No complete sentence fits the measured speech budget')
    write_audio(combined, output_path)
    metadata_path(output_path).write_text(json.dumps(dict(text=''.join(texts), parts=records,
        duration_ms=len(combined), budget_ms=max_duration, source=str(input_path)), ensure_ascii=False, indent=2))
    return len(combined)/1000, texts
