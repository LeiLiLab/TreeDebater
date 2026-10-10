"""Local, pitch-preserving tempo adjustment. No provider or model calls."""
from dataclasses import dataclass
import math
import subprocess
import time

from pydub import AudioSegment


@dataclass
class TempoResult:
    input_seconds: float
    output_seconds: float
    target_seconds: float
    speed: float = 1.0
    clamped: bool = False
    status: str = 'within_deadband'
    processing_seconds: float = 0.0


def change_audio_tempo(audio: AudioSegment, speed: float, *, timeout_seconds: float = 3.0) -> AudioSegment:
    """Change tempo without changing sample rate/pitch or truncating to a deadline.

    Keep a single atempo stage within 0.5–2.0; more extreme changes are deliberately
    unsupported here. Raw PCM pipes avoid temporary files and a second MP3 encode.
    Errors propagate so a caller can explicitly retain the original audio.
    """
    if not math.isfinite(speed) or not .5 <= speed <= 2:
        raise ValueError('Local tempo speed must be between 0.5 and 2.0')
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError('Local tempo timeout must be positive and finite')
    if len(audio) == 0:
        raise ValueError('Cannot adjust empty audio')
    if abs(speed - 1.0) < 1e-6:
        return audio
    pcm = audio.set_sample_width(2)
    command = [AudioSegment.converter, '-hide_banner', '-loglevel', 'error', '-nostdin',
               '-f', 's16le', '-ar', str(pcm.frame_rate), '-ac', str(pcm.channels),
               '-i', 'pipe:0', '-filter:a', f'atempo={speed:.8f}',
               '-f', 's16le', '-acodec', 'pcm_s16le', '-ar', str(pcm.frame_rate),
               '-ac', str(pcm.channels), 'pipe:1']
    completed = subprocess.run(command, input=pcm.raw_data, capture_output=True,
                               check=True, timeout=timeout_seconds)
    if not completed.stdout:
        raise ValueError('Local tempo filter returned empty audio')
    return AudioSegment(data=completed.stdout, sample_width=2,
                        frame_rate=pcm.frame_rate, channels=pcm.channels)


def fit_audio_tempo(audio: AudioSegment, target_seconds: float, *, min_speed: float = .85,
                    max_speed: float = 1.15, deadband_seconds: float = .1):
    """Move existing audio toward a duration using bounded tempo, retaining all text.

    A clamped speed may leave an overlong/short result. Never pad silence or slice
    off speech to claim an exact duration. Measure the returned audio again.
    """
    if not math.isfinite(target_seconds) or target_seconds <= 0:
        raise ValueError('Target seconds must be positive and finite')
    if not (.5 <= min_speed <= 1 <= max_speed <= 2):
        raise ValueError('Local tempo bounds must satisfy 0.5 <= min <= 1 <= max <= 2')
    if not math.isfinite(deadband_seconds) or deadband_seconds < 0:
        raise ValueError('Tempo deadband must be finite and nonnegative')
    if len(audio) == 0:
        raise ValueError('Cannot adjust empty audio')
    started = time.perf_counter()
    duration = len(audio) / 1000
    result = TempoResult(duration, duration, target_seconds)
    if abs(duration - target_seconds) <= deadband_seconds:
        return audio, result
    requested = duration / target_seconds
    speed = max(min_speed, min(max_speed, requested))
    result.clamped = abs(speed - requested) > 1e-6
    if abs(speed - 1) < 1e-6:
        result.status = 'speed_limit'
        return audio, result
    adjusted = change_audio_tempo(audio, speed)
    adjusted_seconds = len(adjusted) / 1000
    if abs(adjusted_seconds - target_seconds) < abs(duration - target_seconds):
        result.speed = speed
        result.output_seconds = adjusted_seconds
        result.status = 'applied'
        audio = adjusted
    else:
        result.status = 'not_improved'
    result.processing_seconds = time.perf_counter() - started
    return audio, result


def main():
    """Adjust an existing recording, refusing to overwrite either input or output."""
    import argparse
    from dataclasses import asdict
    from io import BytesIO
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--target-seconds', type=float, required=True)
    parser.add_argument('--min-speed', type=float, default=.85)
    parser.add_argument('--max-speed', type=float, default=1.15)
    args = parser.parse_args()
    if args.output.exists() or args.input.resolve() == args.output.resolve():
        parser.error('Output must be a new file; the original recording is never overwritten')
    output_format = args.output.suffix.lower().lstrip('.')
    if output_format not in ('mp3', 'wav'):
        parser.error('Output must use .mp3 or .wav')
    source = AudioSegment.from_file(args.input)
    adjusted, result = fit_audio_tempo(source, args.target_seconds,
        min_speed=args.min_speed, max_speed=args.max_speed)
    buffer = BytesIO()
    adjusted.export(buffer, format=output_format)
    decoded = AudioSegment.from_file(BytesIO(buffer.getvalue()), format=output_format)
    result.output_seconds = len(decoded) / 1000
    with args.output.open('xb') as file:
        file.write(buffer.getvalue())
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
