"""Whole Flat speech with a fixed opening and overlapping tail revision/TTS.

The ordinary Flat draft and whole-draft audience feedback are retained. Only the
revision schedule changes: finalize the opening, then synthesize it concurrently
with revising the remaining draft. Published words are never revised or replayed.
"""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import re
import time



def split_opening(draft):
    """Choose a complete initial paragraph/sentence, without cutting words."""
    paragraphs = [part.strip() for part in re.split(r'\n\s*\n', draft) if part.strip()]
    if len(paragraphs) > 1:
        return paragraphs[0], '\n\n'.join(paragraphs[1:])
    parts = re.split(r'(?<=[.!?])\s+', draft.strip(), maxsplit=1)
    return parts[0], parts[1] if len(parts) > 1 else ''


def remaining_text(text, prefix):
    """Accept an accidentally echoed leading prefix once; never play it twice."""
    text = text.strip()
    if text.startswith(prefix):
        text = text[len(prefix):].strip()
    def normalized(value):
        return ' '.join(value.split()).casefold()
    if normalized(prefix) in normalized(text):
        raise ValueError('Remaining speech repeats the fixed prefix')
    return text


def speak_with_overlap_prefix(player, max_time, history, config, call_id, kwargs):
    from agents import Debater
    from pydub import AudioSegment
    import tts_streaming

    start = time.perf_counter()
    output = Path(player._speech_audio_file())
    directory = output.parent / f'{output.stem}_chunks'
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.glob('chunk_*.mp3')):
        raise FileExistsError(f'Speech chunks already exist: {directory}')
    trace = {'mode': 'flat_full_script_natural_prefix', 'status': 'preparing', 'chunks': []}
    prefix = ''
    tail_future = None
    outer_callback = getattr(player, 'tts_chunk_callback', None)

    def save():
        trace['wall_seconds'] = time.perf_counter() - start
        trace['committed_text'] = '\n\n'.join(item['text'] for item in trace['chunks'])
        path = directory / 'overlap_prefix.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)

    def record(index, path, text, duration):
        if index == 0 and prefix and text != prefix:
            raise ValueError('TTS changed the fixed prefix')
        trace['chunks'].append({'index': index, 'text': text, 'path': str(path),
                                'audio_seconds': duration, 'ready_seconds': time.perf_counter() - start})
        save()
        if outer_callback is not None:
            outer_callback(index, path, text, duration)

    def batch_fallback(draft, feedback, evidence, allocation, reason):
        trace.update(status='fallback_full_script', fallback_reason=reason)
        revised = player._length_adjust(draft, feedback, evidence, allocation, max_time,
                                        max_retry=1, call_id=call_id)
        save()
        # No prefix has been published; one ordinary full-script delivery is safe.
        text, _, duration = tts_streaming.convert_text_to_speech_streaming(
            revised, str(output), max_time, config=config, on_chunk=record,
            motion=player.motion, side=player.side)
        trace['audio_seconds'] = duration
        return Debater.post_process(player, text, max_time, time_control=False)

    try:
        draft = player._get_response(player.conversation, **kwargs)
        trace['draft_ready_seconds'] = time.perf_counter() - start
        feedback, evidence, allocation, draft = player._get_revision_suggestion(
            statement=draft, history=history, add_evidence=True, call_id=call_id, **kwargs)
        if not kwargs.get('single_pass_revision', getattr(player.config, 'single_pass_revision', False)):
            draft = player._length_adjust(draft, feedback, evidence, allocation, max_time,
                                          max_retry=1, call_id=call_id)
            feedback, evidence, _, draft = player._get_revision_suggestion(
                statement=draft, history=history, add_evidence=False, call_id=call_id, **kwargs)
            trace['whole_feedback_passes'] = 2
        else:
            trace['whole_feedback_passes'] = 1
        trace['feedback_ready_seconds'] = time.perf_counter() - start
        opening, tail = split_opening(draft)
        trace.update(draft=draft, initial_opening=opening, initial_tail=tail)
        prefix = opening.strip()
        trace.update(fixed_prefix=prefix, prefix_ready_seconds=time.perf_counter() - start)
        invalid = (not prefix or len(prefix) > config.max_chunk_chars
                   or '\n' in prefix or prefix[-1:] not in '.?!' or any(mark in prefix for mark in ('**', '```')))
        if invalid:
            prefix = ''  # Nothing is fixed/published when falling back.
            return batch_fallback(draft, feedback, evidence, allocation, 'Invalid/overlong opening')
        remaining_budget = max_time - tts_streaming.estimate_statement_seconds(prefix, config)
        if tail and remaining_budget < 3:
            prefix = ''
            return batch_fallback(draft, feedback, evidence, allocation, 'Insufficient budget for remaining draft')

        def revise_tail():
            began = time.perf_counter() - start
            revised = player._length_adjust(tail, feedback, evidence, allocation, remaining_budget,
                                            max_retry=1, call_id=call_id, frozen_prefix=prefix) if tail else ''
            revised = remaining_text(revised, prefix)
            if tail and not revised:
                raise ValueError('Revision omitted the remaining speech')
            return revised, began, time.perf_counter() - start

        def supply_tail():
            revised, began, ended = tail_future.result()
            trace.update(revised_tail=revised, tail_revision_start_seconds=began,
                         tail_revision_end_seconds=ended)
            save()
            return revised

        def validate_chunk(index, text):
            if index == 0 and text != prefix:
                raise ValueError('TTS changed the fixed prefix')
            if index > 0 and ' '.join(prefix.split()).casefold() in ' '.join(text.split()).casefold():
                raise ValueError('Adaptive TTS repeated the fixed prefix')

        # Only one text-revision task; its work overlaps first-chunk audio. The
        # executor is joined even if audio fails, before the turn's budget settles.
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='flat-tail') as executor:
            tail_future = executor.submit(revise_tail)
            trace['status'] = 'delivering'
            text, _, duration = tts_streaming.convert_text_to_speech_streaming(
                prefix, str(output), max_time, config=config, on_chunk=record,
                motion=player.motion, side=player.side, tail_supplier=supply_tail,
                validate_chunk=validate_chunk)
        trace.update(status='completed', audio_seconds=duration)
        save()
        return Debater.post_process(player, text, max_time, time_control=False)
    except Exception as exc:
        trace.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        # Preserve the exact published prefix; no full-speech retry after output.
        if trace['chunks']:
            committed = '\n\n'.join(item['text'] for item in trace['chunks'])
            Debater.post_process(player, committed, max_time, time_control=False)
            combined = AudioSegment.silent(duration=0)
            for item in trace['chunks']:
                combined += AudioSegment.from_file(item['path'])
            buffer = BytesIO()
            combined.export(buffer, format='mp3')
            output.write_bytes(buffer.getvalue())
        raise
    finally:
        save()
        player.debate_thoughts.append({'mode': 'overlap_prefix_delivery', 'stage': player.status,
                                      'side': player.side, 'trace': trace})
