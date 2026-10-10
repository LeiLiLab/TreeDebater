"""Read saved delivery traces without loading providers or dispatching requests."""
import argparse
import json
from pathlib import Path


def elapsed(trace, key):
    value = trace.get(key) or {}
    if 'start_seconds' in value and 'end_seconds' in value:
        return value['end_seconds'] - value['start_seconds']
    return None


def audit(run):
    rows = []
    for path in sorted(run.glob('*/*/listening_prefix.json')):
        trace = json.loads(path.read_text())
        evidence = trace.get('prepared_evidence') or {}
        events = evidence.get('events', [])
        selections = []
        for event in events:
            candidates = event.get('candidate_ids', [])
            reused = event.get('reused_ids', [])
            additions = event.get('new_ids', [])
            selections.append(dict(
                candidate_count=len(candidates), reused_count=len(reused),
                added_count=len(additions), status=event.get('status'),
                repeated_candidate_ids=sorted(set(candidates) & set(reused)),
                repeated_added_ids=sorted(set(additions) & set(reused)),
                seconds=(event['end'] - event['start']) if 'end' in event else None))
        audio = trace.get('first_body_audio') or {}
        final_ready = trace.get('final_input_ready_seconds')
        audio_ready = audio.get('ready_seconds')
        rows.append(dict(
            turn=path.parts[-3], trace=str(path), cold_prefix=trace.get('cold_prefix'),
            body_preparation_source=trace.get('body_preparation_source'),
            feedback_seconds=elapsed(trace, 'parallel_body_feedback'),
            evidence_seconds=elapsed(trace, 'parallel_evidence_selection'),
            revision_seconds=elapsed(trace, 'parallel_body_revision'),
            prefix_duration_wait_seconds=elapsed(trace, 'prefix_duration_wait'),
            feedback_reused=(trace.get('parallel_body_feedback') or {}).get('reused'),
            evidence_reused=(trace.get('parallel_evidence_selection') or {}).get('reused'),
            revision_reused=(trace.get('parallel_body_revision') or {}).get('reused'),
            first_audio_reused=audio.get('reused'),
            final_input_ready_seconds=final_ready,
            first_body_audio_ready_seconds=audio_ready,
            audio_ready_before_final_input_seconds=(max(0, final_ready - audio_ready)
                if final_ready is not None and audio_ready is not None else None),
            selection_events=selections))
    if not rows:
        raise ValueError(f'No saved delivery traces in {run}')
    return dict(run=str(run), turns=rows, limitations=[
        'Phase times overlap; do not sum them as sequential end-to-end latency.',
        'Cold opening lacks parallel phase spans; missing values do not mean zero work.',
        'Repeated source IDs and semantic redundancy are different; this audit only checks IDs.',
        'Audio ready before final input identifies a publication dependency, not proven removable delay.',
        'Historical fresh conversations are not a frozen-input causal comparison.',
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f'Saved {len(result["turns"])} turns to {args.output}; no provider calls.')


if __name__ == '__main__':
    main()
