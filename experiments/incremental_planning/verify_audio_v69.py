"""Decode selected published MP3s and compare saved duration measurements."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
O = Path('experiments/incremental_planning/run/listening-motion-live-v52-refactor-streamed-motion02-v69')
def verify(item):
    turn, e = item
    path=e['path']
    decode=subprocess.run(['ffmpeg','-v','error','-i',path,'-f','null','-'],capture_output=True,text=True)
    probe=subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','json',path],capture_output=True,text=True,check=True)
    seconds=float(json.loads(probe.stdout)['format']['duration'])
    return dict(turn=turn,index=e['index'],path=path,decoded=decode.returncode==0,
        decode_stderr=decode.stderr,reported_seconds=e['duration_seconds'],container_seconds=seconds,
        difference_seconds=seconds-e['duration_seconds'])
def main():
    items=[(p.parent.name,e) for p in sorted(O.glob('[0-9]*/events.json')) for e in json.loads(p.read_text())]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        rows=list(ex.map(verify,items))
    result=dict(chunk_count=len(rows),all_decoded=all(r['decoded'] for r in rows),
        max_absolute_duration_difference_seconds=max(abs(r['difference_seconds']) for r in rows),
        limitation='Container duration includes MP3 padding; decoding is not a subjective listening quality score.',chunks=rows)
    (O/'audio_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print({k:v for k,v in result.items() if k!='chunks'})
if __name__=='__main__': main()
