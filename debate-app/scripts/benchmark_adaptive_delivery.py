"""Real-provider timing benchmark on saved pre-TTS speeches; no browser required.

Runs off/on comparisons against identical saved inputs. Audio playback is modeled
from chunk-ready times to expose potential underruns, without waiting to play it.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['opening', 'closing'], required=True)
    parser.add_argument('--adaptive', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for key, value in json.loads((ROOT / 'src/configs/api_key.json').read_text()).items():
        os.environ.setdefault(key, value)
    sys.path.insert(0, str(ROOT / 'src'))
    from openai import OpenAI
    import tts_streaming as tts
    from streaming.config import OutputConfig

    source = ROOT / 'debate-app/reports/recorded-deepseek-full-speech-2026-09-13-human-for-io.log'
    blocks = re.split(r'(?=2026-\d\d-\d\d \d\d:\d\d:\d\d DEBUG \[io\])', source.read_text())
    matches = [b for b in blocks if 'title=Time-Control-Statement ' in b.split('\n')[0]
               and f'stage={args.stage} side=against' in b.split('\n')[0]]
    if len(matches) != 1:
        raise ValueError(f'Expected one saved pre-TTS speech, found {len(matches)}')
    text = matches[0].split('\n', 2)[2].rsplit('\n' + '=' * 60, 1)[0].strip()
    clean = tts.remove_subtitles(tts.remove_citation(text)[0])
    (args.output / 'input.txt').write_text(clean)
    cfg = OutputConfig(budget_mode='audio_duration', refinement_model='deepseek-v4-flash',
                       adaptive_delivery=args.adaptive)
    budget = 240 if args.stage == 'opening' else 120
    delivered = []
    rewrite_calls = []
    original = tts._revise_to_n_words

    def rewrite(*a, **kw):
        record = {'start_seconds': time.perf_counter() - started}
        rewrite_calls.append(record)
        try:
            return original(*a, **kw)
        finally:
            record['end_seconds'] = time.perf_counter() - started

    tts._revise_to_n_words = rewrite
    def ready(index, path, content, seconds):
        delivered.append({'index': index, 'ready_seconds': time.perf_counter() - started,
                          'audio_seconds': seconds, 'text': content})
        print(f'chunk {index} ready={delivered[-1]["ready_seconds"]:.2f}s audio={seconds:.2f}s', flush=True)

    started = time.perf_counter()
    profiles, round_profile, _, texts = tts.run_pipeline(
        OpenAI(), tts.split_by_paragraphs(clean), budget, config=cfg,
        out_dir=args.output, on_chunk=ready,
        motion='Learning to be a good writer still matters in the age of AI', side='against',
    )
    playback_end = delivered[0]['ready_seconds']
    gap_total = 0.
    for chunk in delivered:
        chunk['playback_gap_seconds'] = max(0., chunk['ready_seconds'] - playback_end)
        gap_total += chunk['playback_gap_seconds']
        playback_end = max(playback_end, chunk['ready_seconds']) + chunk['audio_seconds']
    result = {'stage': args.stage, 'config': asdict(cfg), 'budget_seconds': budget,
              'first_audio_seconds': delivered[0]['ready_seconds'],
              'audio_seconds': sum(c['audio_seconds'] for c in delivered),
              'potential_playback_gaps_seconds': gap_total,
              'pipeline_seconds': time.perf_counter() - started,
              'rewrite_calls': rewrite_calls, 'chunks': delivered,
              'profiles': [asdict(p) for p in profiles], 'round_profile': asdict(round_profile)}
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['stage', 'first_audio_seconds', 'audio_seconds',
          'potential_playback_gaps_seconds', 'pipeline_seconds']}), flush=True)


if __name__ == '__main__':
    main()
