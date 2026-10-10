"""Read-only combination experiment report and historical motion02 comparison."""
import json
import statistics
import sqlite3

import report_streamed_motion_v61 as report

BASELINE = report.OUT
report.RUN = 'listening-motion-live-v52-refactor-streamed-motion02-v64'
report.OUT = report.audit.D / 'run' / report.RUN


def delivery(out):
    summary = json.loads((out / 'summary.json').read_text())
    rows = []
    for row in summary['turns']:
        events = report.audit.read(out / row['turn'] / 'events.json', [])
        row = dict(row)
        row['prefix_seconds'] = events[0]['duration_seconds'] if events else None
        row['prefix_words'] = len(events[0]['text'].split()) if events else None
        row['first_body_words'] = len(events[1]['text'].split()) if len(events) > 1 else None
        row['first_body_seconds'] = events[1]['duration_seconds'] if len(events) > 1 else None
        rows.append(row)
    complete = [r for r in rows if r['status'] == 'completed']
    def median(key, values=complete):
        return statistics.median(r[key] for r in values) if values else None
    return dict(turns=rows, completed=len(complete),
        prefix_median_seconds=median('prefix_seconds'),
        first_body_words_median=median('first_body_words'),
        first_body_wait_median_seconds=median('first_body_after_prefix_seconds'),
        post_cold_first_audio_median_seconds=median('first_audio_seconds', complete[1:]),
        maximum_gap_seconds=max((r['max_gap_seconds'] for r in complete), default=None),
        gaps_over_100ms=sum(r['max_gap_seconds'] > .1 for r in complete),
        within_five_percent=sum(r['duration_error_ratio'] <= .05 for r in complete))


def main():
    report.main()
    db = sqlite3.connect('file:' + str(report.audit.D / 'run/cost.sqlite') + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    attempts = ['listening-motion-live-v52-refactor-streamed-motion02-' + v for v in ('v62','v63','v64')]
    costs = report.audit.budget(db, attempts, 8)
    db.close()
    summary_path = report.OUT / 'summary.json'
    summary = json.loads(summary_path.read_text())
    summary['combination_attempts_budget'] = costs
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    before, after = delivery(BASELINE), delivery(report.OUT)
    result = dict(before=before, after=after,
        all_attempts_cost=costs,
        limitation='Historical same-motion fresh conversation, not frozen-input paired ablation. Includes the separately requested no-new-evidence closing prompt correction.')
    (report.OUT / 'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    keys = ['completed','prefix_median_seconds','first_body_words_median',
            'first_body_wait_median_seconds','post_cold_first_audio_median_seconds',
            'maximum_gap_seconds','gaps_over_100ms','within_five_percent']
    lines = ['metric | v61 | combination v64'] + [f'{key} | {before[key]} | {after[key]}' for key in keys]
    lines += ['', 'turn | prefix words/seconds | first body words/seconds | gap seconds']
    lines += [f"{r['turn']} | {r['prefix_words']}/{r['prefix_seconds']} | {r['first_body_words']}/{r['first_body_seconds']} | {r['max_gap_seconds']}" for r in after['turns']]
    lines += ['',result['limitation']]
    (report.OUT / 'comparison.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:9]))


if __name__ == '__main__':
    main()
