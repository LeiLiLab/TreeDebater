"""Offline final report for the approved FastSpeech six-turn debate."""
import json
from pathlib import Path
import sqlite3

import audit_handoff_v70 as handoff
import verify_audio_v70 as audio
import report_streamed_motion_v61 as report
from report_streamed_motion_v64 import delivery

D = Path(__file__).resolve().parent
RUN = 'listening-motion-live-v52-refactor-fastspeech-motion03-v71'
OUT = D / 'run' / RUN


def main():
    handoff.RUN, handoff.O = RUN, OUT
    handoff.main()
    audio.O = OUT
    audio.main()
    report.RUN, report.OUT = RUN, OUT
    report.main()
    summary_path = OUT / 'summary.json'
    summary = json.loads(summary_path.read_text())
    manifest = json.loads((D / f'manifest_{RUN}.json').read_text())
    summary['parent_cost']['cap_usd'] = 80
    summary['tts_backend'] = 'fastspeech'
    summary['output_config'] = manifest['output_config']
    db = sqlite3.connect(f'file:{D}/run/cost.sqlite?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    calls = [dict(row) for row in db.execute(
        'select id,label,state,estimated_usd from calls where instr(label,?)=1', (RUN+'/',))]
    tts_calls = [row for row in calls if '/turn_' in row['label'] and row['label'].endswith('/tts')]
    summary['tts_transport_audit'] = dict(paid_tts_calls=tts_calls,
        all_audio_local=not tts_calls,
        note='All synthesis routes use the FastSpeech config. Launcher rejects the OpenAI TTS transport; ASR remains paid Whisper.')
    summary['initial_review_audit'] = json.loads((OUT/'initial_review_audit.json').read_text())
    summary['audio_verification'] = json.loads((OUT/'audio_verification.json').read_text())
    summary['handoff_latency_audit'] = json.loads((OUT/'handoff_latency_audit.json').read_text())
    summary['handoff_fix_audit'] = json.loads((OUT/'handoff_fix_audit.json').read_text())
    batches=[]
    for p in sorted(OUT.glob('[0-9]*/result.json')):
        row=json.loads(p.read_text())
        heard = {item['sequence']: item for item in row.get('heard', [])}
        for batch in row.get('analysis_batches', []):
            members = [heard[i] for i in batch.get('sequences', []) if i in heard]
            if members and 'analysis_end_monotonic' in batch:
                delay = batch['analysis_end_monotonic'] - members[-1]['heard_end_monotonic']
                batches.append(dict(turn=p.parent.name, batch=batch['index'],
                                    words=batch['words'], backlog_seconds=delay))
    backlog=dict(max_backlog_seconds=max((x['backlog_seconds'] for x in batches),default=None),
                 batches_over_30_seconds=[x for x in batches if x['backlog_seconds']>30],
                 limitation='Listening processing backlog is distinct from audio playback gaps.')
    (OUT/'listening_backlog_audit.json').write_text(json.dumps(backlog,indent=2)+'\n')
    summary['listening_backlog_audit']=backlog
    summary['limitations'] += ['Local FastSpeech2 LJSpeech voice, native speed. No paid independent debate quality scoring.',
        'Historical v70 uses the same motion but fresh generation and earlier source; timing is descriptive, not a controlled TTS-only ablation.']
    summary['source_snapshot_note'] = 'Workspace files changed after launch; archived startup copies are in source_snapshot. Later edits are not validated by this run.'
    summary['source_files_added_after_launch'] = sorted(
        set(str(p) for p in Path('src').rglob('*.py')) -
        set(json.loads((OUT/'source_snapshot/files.json').read_text())))
    for key, name in [('final_settlement', 'final_settlement.json'), ('combined_audio', 'debate_audio.json')]:
        if (OUT/name).exists():
            summary[key] = json.loads((OUT/name).read_text())
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    previous=D/'run'/'listening-motion-live-v52-refactor-streamed-motion03-v70'
    comparison=dict(v70=delivery(previous),v71=delivery(OUT),limitation=summary['limitations'][-1])
    (OUT/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2)+'\n')
    text=(OUT/'report.txt').read_text().replace('/50; pending calls','/80; pending calls').replace('\nnull\n', '\nNo independent quality grading was run.\n')
    text += ('\nTTS backend: FastSpeech2 LJSpeech + HiFi-GAN, native speed.\n'
             f'Paid TTS calls: {len(tts_calls)}\n'
             f"All published prefixes reviewed: {summary['initial_review_audit']['all_published_prefixes_reviewed']}\n"
             f"Decoded audio chunks: {summary['audio_verification']['chunk_count']}\n"
             f"Sources changed since launch: {summary['initial_review_audit']['source_changed']}\n"
             + summary['limitations'][-1]+'\n'+summary['source_snapshot_note']+'\n')
    (OUT/'report.txt').write_text(text)
    print(text)
    db.close()


if __name__ == '__main__':
    main()
