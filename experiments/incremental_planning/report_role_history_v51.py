"""Read-only live-run audit; no model calls or billing reconciliation."""
import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from streaming.experiment_accounting import exposure_query

D = ROOT / 'experiments/incremental_planning'
RUN = 'listening-motion-live-v51-retrieval'
R = D / 'run' / RUN
RUNS = ['listening-motion-live-v48', 'listening-motion-live-v48-retrieval',
        'listening-motion-live-v49-retrieval', 'listening-motion-live-v50-retrieval', RUN]


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def budget(db, runs, cap):
    where = ' WHERE (' + ' OR '.join(f"instr(c.label,'{run}/')=1" for run in runs) + ')'
    rows = [dict(row) for row in db.execute('SELECT * FROM calls c' + where)]
    known = sum(row['estimated_usd'] or 0 for row in rows)
    for row in rows:
        if row['estimated_usd'] is None:
            artifact = read(D / 'run' / f"call_{row['id']:06}.json", {})
            known += sum(call['estimated_usd'] for call in artifact.get('external_calls', [])
                         if call['state'] == 'ok')
    return dict(cap_usd=cap, known_usage_estimate_usd=known,
        conservative_exposure_usd=db.execute(exposure_query(where)).fetchone()[0],
        calls=len(rows), pending_calls=sum(row['state'] == 'pending' for row in rows),
        uncertain_calls=[row for row in rows if row['state'] not in ('ok', 'pending')],
        invoice_status='Provisional rate-based estimate; failed or unknown requests retain reservations.')


def turn(folder):
    result = read(folder / 'result.json', {})
    events = read(folder / 'events.json', [])
    played = read(folder / 'playback.json', [])
    paths = list(folder.glob('*_chunks/listening_prefix.json'))
    trace = read(paths[0], {}) if paths else {}
    first = result.get('endpoint_to_first_audio_seconds') if not folder.name.startswith('00') else result.get('generation_to_first_audio_seconds')
    if first is None and events:
        first = events[0].get('endpoint_wait_seconds') if not folder.name.startswith('00') else events[0]['ready_seconds']
    target = result.get('target_seconds', 120 if 'closing' in folder.name else 240)
    audio = sum(event['duration_seconds'] for event in events)
    gap = max((item['gap_seconds'] for item in played), default=None)
    text = '\n\n'.join(event['text'] for event in events)
    sentences = Counter(re.sub(r'\s+', ' ', item).strip().lower()
                        for item in re.split(r'(?<=[.!?])\s+', text) if len(item.split()) >= 5)
    return dict(turn=folder.name, status=result.get('status', 'running'),
        first_audio_seconds=first, audio_seconds=audio, target_seconds=target,
        duration_error_ratio=abs(audio-target)/target, max_gap_seconds=gap,
        timing_pass=(result.get('status') == 'completed' and first is not None and first <= 10
                     and gap is not None and gap <= 2 and abs(audio-target)/target <= .15),
        error=result.get('error'), words=len(text.split()), answer=text,
        repeated_sentences=[dict(sentence=sentence, count=count)
                            for sentence, count in sentences.items() if count > 1],
        reference_heading_in_audio=bool(re.search(r'(?im)^\W*(references|bibliography)\b', text)),
        prefix=trace.get('fixed_prefix'), cold_prefix=trace.get('cold_prefix'),
        gate_ready_seconds=trace.get('gate_ready_seconds'),
        last_review_accepted=(trace['endpoint_reviews'][-1].get('accepted')
                              if trace.get('endpoint_reviews') else None),
        parallel_evidence_selection=trace.get('parallel_evidence_selection'),
        parallel_body_revision=trace.get('parallel_body_revision'),
        tail_work=trace.get('tail_work'), chunks=[{k: event.get(k) for k in
            ('index', 'ready_seconds', 'duration_seconds')} for event in events])


def main():
    manifest = read(D / f'manifest_{RUN}.json', {})
    db = sqlite3.connect('file:' + str(D / 'run/cost.sqlite') + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    authors = []
    calls = []
    for row in db.execute('SELECT * FROM calls WHERE instr(label,?)=1 ORDER BY id', (RUN+'/',)):
        a = read(D / 'run' / f"call_{row['id']:06}.json", {})
        request = a.get('request', {})
        messages = request.get('messages', [])
        prompt = messages[-1]['content'] if messages else ''
        item = dict(id=row['id'], label=row['label'], state=row['state'], seconds=row['seconds'],
            estimated_usd=row['estimated_usd'], input_tokens=row['input_tokens'],
            output_tokens=row['output_tokens'], error=a.get('error'))
        calls.append(item)
        if any(marker in prompt for marker in ('LISTENING PREFIX DRAFT', 'LISTENING PREFIX REPAIR',
                                               'LISTENING BODY DRAFT', 'Return ONLY the revised remaining speech')):
            authors.append(dict(item, roles=[message['role'] for message in messages],
                message_characters=[len(message['content']) for message in messages],
                total_characters=sum(len(message['content']) for message in messages),
                historical_speeches_repeated_in_instruction=[i for i, message in enumerate(messages[1:-1], 1)
                    if message['content'] in prompt], response_format=request.get('response_format')))
    turns = [turn(folder) for folder in sorted(R.glob('[0-9]*')) if folder.is_dir()]
    baseline = D / 'run/listening-motion-live-v50-retrieval'
    comparison = []
    for item in turns:
        folder = baseline / item['turn']
        if item['turn'] == '05_closing_against' and (baseline / 'completion-v1' / item['turn']).exists():
            folder = baseline / 'completion-v1' / item['turn']
        old = turn(folder)
        keys = ('status', 'first_audio_seconds', 'max_gap_seconds', 'audio_seconds', 'words')
        comparison.append(dict(turn=item['turn'], before={key: old[key] for key in keys},
                               after={key: item[key] for key in keys}))
    report = dict(generated_utc=datetime.now(timezone.utc).isoformat(), run_id=RUN,
        status=manifest.get('status'), completed_turns=manifest.get('completed_turns'),
        motion=manifest.get('motion'), source_digest=manifest.get('source_digest'),
        stop_error=manifest.get('stop_error'), run_budget=budget(db, [RUN], 10),
        combined_budget=budget(db, RUNS, 60), turns=turns, authoring_calls=authors,
        calls=calls, comparison=comparison, manual_quality_review=read(D/'v51_manual_quality.json'),
        limitations=['One motion; no paired ablation or independent quality grader.',
            'Several related fixes changed since v50; timing comparison is observational.',
            'Saved retrieved evidence; real TTS and ASR; server-paced playback, not browser latency.'])
    quality = report['manual_quality_review'] or {}
    report['acceptance'] = dict(first_audio_max_seconds=10, max_gap_seconds=2,
                               duration_error_ratio=.15)
    report['summary'] = dict(completed_turns=sum(item['status'] == 'completed' for item in turns),
        timing_pass_turns=sum(item['timing_pass'] for item in turns),
        duration_pass_turns=sum(item['status'] == 'completed' and item['duration_error_ratio'] <= .15
                                for item in turns),
        authoring_calls_audited=len(authors), historical_speeches_repeated_in_instruction=sum(
            len(item['historical_speeches_repeated_in_instruction']) for item in authors),
        repeated_sentences=sum(len(item['repeated_sentences']) for item in turns))
    report['validation_pass'] = (len(turns) == 6 and all(item['timing_pass'] for item in turns)
                                and quality.get('all_speakers_kept_assigned_side') is True)
    path = D / f'{RUN}_validation.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(dict(path=str(path), status=report['status'], stop_error=report['stop_error'],
        run_budget=report['run_budget'], combined_budget={key: report['combined_budget'][key]
            for key in ('cap_usd', 'known_usage_estimate_usd', 'conservative_exposure_usd', 'pending_calls')},
        turns=[{key: item[key] for key in ('turn', 'status', 'first_audio_seconds',
            'audio_seconds', 'max_gap_seconds', 'timing_pass')} for item in turns]), indent=2))


if __name__ == '__main__':
    main()
