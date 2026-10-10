"""Saved-run report: normal body paragraphs, no endpoint evidence supplements."""
import json

import report_streamed_motion_v64 as comparison

report = comparison.report
report.RUN = 'listening-motion-live-v52-refactor-streamed-motion02-v69'
report.OUT = report.audit.D / 'run' / report.RUN


def main():
    report.main()
    summary_path = report.OUT / 'summary.json'
    summary = json.loads(summary_path.read_text())
    summary['parent_cost']['cap_usd'] = 80
    report_path = report.OUT / 'report.txt'
    report_path.write_text(report_path.read_text().replace('/50; pending calls', '/80; pending calls'))
    evidence = []
    for row in summary['turns']:
        folder = report.OUT / row['turn']
        paths = list(folder.glob('*_chunks/listening_prefix.json'))
        trace = report.audit.read(paths[0], {}) if paths else {}
        prepared = trace.get('prepared_evidence') or {}
        selection = trace.get('parallel_evidence_selection') or {}
        evidence.append(dict(turn=row['turn'], mode=prepared.get('mode'),
            supplement_enabled=prepared.get('endpoint_supplement_enabled'),
            retained_ids=prepared.get('selected_ids', []),
            endpoint_selection_seconds=(selection['end_seconds'] - selection['start_seconds']
                if 'end_seconds' in selection else None)))
    summary['endpoint_evidence_reuse'] = evidence
    summary['failure_audit'] = report.audit.read(report.OUT / 'failure_audit.json')
    summary['initial_review_audit'] = report.audit.read(report.OUT / 'initial_review_audit.json')
    summary['handoff_latency_audit'] = report.audit.read(report.OUT / 'handoff_latency_audit.json')
    summary['handoff_fix_audit'] = report.audit.read(report.OUT / 'handoff_fix_audit.json')
    summary['audio_verification'] = report.audit.read(report.OUT / 'audio_verification.json')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    values = {}
    for version in ('v68', 'v69'):
        out = report.audit.D / 'run' / ('listening-motion-live-v52-refactor-streamed-motion02-' + version)
        values[version] = comparison.delivery(out)
    result = dict(runs=values, endpoint_evidence_reuse=evidence,
        limitation='Fresh conversations, not paired ablations. v69 includes version-bound prefix review, independent prefix TTS/body preparation, latest completed body context, feedback before transferred audio completion, and a second nonblocking body-draft check after complete ASR. Timing differences cannot be assigned causally to one change.')
    (report.OUT / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    keys = ['completed', 'prefix_median_seconds', 'first_body_words_median',
            'first_body_wait_median_seconds', 'post_cold_first_audio_median_seconds',
            'maximum_gap_seconds', 'gaps_over_100ms', 'within_five_percent']
    lines = ['metric | v68 | v69'] + [key + ' | ' + ' | '.join(str(values[v][key])
             for v in ('v68', 'v69')) for key in keys]
    lines += ['', result['limitation']]
    review = summary.get('initial_review_audit') or {}
    audio = summary.get('audio_verification') or {}
    lines += ['', f"Published prefixes match accepted review versions: {review.get('all_published_prefixes_reviewed')}; review requests: {review.get('review_request_count')}",
              f"Audio chunks decoded: {audio.get('chunk_count')}; all decoded: {audio.get('all_decoded')}",
              f"Changed snapshotted sources: {review.get('source_changed')}",
              'Branch coverage: no pending prefix-audio transfer and no late body-draft adoption were observed. This run does not independently validate those two boundary cases.']
    if summary.get('stop_error'):
        lines += ['', 'INCOMPLETE RUN: ' + summary['stop_error'],
                  'Only completed turns contribute to timing medians; failed or partial closings are not successful observations.',
                  json.dumps(summary['failure_audit'], ensure_ascii=False, indent=2)]
    (report.OUT / 'comparison.txt').write_text('\n'.join(lines) + '\n')
    with (report.OUT / 'report.txt').open('a') as handle:
        handle.write('\n' + '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
