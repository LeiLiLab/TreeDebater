"""Read-only report for the additional streamed-revision motion."""
import json
from pathlib import Path
import sqlite3
import statistics

import report_role_history_v51 as audit
from streaming.experiment_accounting import exposure_query

RUN = 'listening-motion-live-v52-refactor-streamed-motion02-v61'
OUT = audit.D / 'run' / RUN
PARENT = 'listening-motion-live-v52-refactor'


def main():
    manifest = audit.read(audit.D / f'manifest_{RUN}.json', {})
    db = sqlite3.connect('file:' + str(audit.D / 'run/cost.sqlite') + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    turns = []
    for folder in sorted(OUT.glob('[0-9]*')):
        if not folder.is_dir():
            continue
        row = audit.turn(folder)
        result = audit.read(folder / 'result.json', {})
        events = audit.read(folder / 'events.json', [])
        row['first_audio_seconds'] = (result.get('generation_to_first_audio_seconds')
            if folder.name.startswith('00') else result.get('endpoint_to_first_audio_seconds'))
        row['first_body_after_prefix_seconds'] = (
            events[1]['ready_monotonic'] - events[0]['ready_monotonic'] if len(events) > 1 else None)
        row['listener_backlog_seconds'] = result.get('listener_backlog_seconds')
        turns.append(row)
    costs = audit.budget(db, [RUN], manifest.get('run_cap_usd', 8))
    parent = dict(cap_usd=50,
        known_usage_estimate_usd=db.execute('select coalesce(sum(estimated_usd),0) from calls where instr(label,?)=1', (PARENT,)).fetchone()[0],
        conservative_exposure_usd=db.execute(exposure_query(' WHERE instr(c.label,?)=1'), (PARENT,)).fetchone()[0])
    streams = [dict(r) for r in db.execute("select id,label,state,seconds from calls where instr(label,?)=1 and label like '%whole_revision_stream'", (RUN,))]
    completed = [r for r in turns if r['status'] == 'completed']
    summary = dict(status=manifest.get('status'), motion=manifest.get('motion'),
        stop_error=manifest.get('stop_error'), completed_turns=len(completed),
        within_five_percent=sum(r['duration_error_ratio'] <= .05 for r in completed),
        turns_with_gap_over_100ms=sum((r['max_gap_seconds'] or 0) > .1 for r in completed),
        run_cost=costs, parent_cost=parent, streamed_requests=streams, turns=turns,
        quality_review=audit.read(OUT / 'quality_review.json'),
        limitations=['One fresh motion, no matched baseline or independent quality scoring.',
            'Real TTS and ASR with server-paced playback; excludes browser/network playback latency.',
            'Saved evidence pool, not newly retrieved material. Costs are provisional usage estimates.'])
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    lines = [f"Motion: {summary['motion']}", f"Status: {summary['status']}; completed {len(completed)}/6",
        'First audio: cold generation wait for 00; opponent-endpoint wait for later turns.',
        'turn | status | first audio s | first body after prefix s | audio/target s | max gap s']
    for r in turns:
        lines.append(' | '.join(str(v) for v in (r['turn'],r['status'],r['first_audio_seconds'],
            r['first_body_after_prefix_seconds'],f"{r['audio_seconds']:.3f}/{r['target_seconds']}",r['max_gap_seconds'])))
    lines += ['', f"Within ±5%: {summary['within_five_percent']}/{len(completed)}",
        f"Usage estimate USD {costs['known_usage_estimate_usd']:.6f}; conservative exposure USD {costs['conservative_exposure_usd']:.6f}/{costs['cap_usd']}",
        f"Parent exposure USD {parent['conservative_exposure_usd']:.6f}/50; pending calls {costs['pending_calls']}",
        '', json.dumps(summary['quality_review'],ensure_ascii=False,indent=2), '', *summary['limitations']]
    (OUT / 'report.txt').write_text('\n'.join(lines)+'\n')
    (OUT / 'speeches.txt').write_text('\n\n'.join(r['turn']+'\n'+r['answer'] for r in turns)+'\n')
    print('\n'.join(lines[:4+len(turns)]))
    db.close()


if __name__ == '__main__':
    main()
