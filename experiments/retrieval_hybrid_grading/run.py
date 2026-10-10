"""Complete frozen hybrid grades only after explicit approval of this run."""
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


class MalformedCompletion(ValueError):
    pass


def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def save(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def summarize(rows):
    materials=[m for r in rows for m in r['materials']]
    valid=sum(m['valid'] is True for m in materials)
    unknown=sum(m['valid'] is None for m in materials)
    queries=sum(any(m['valid'] is True for m in r['materials']) for r in rows)
    unresolved=sum(not any(m['valid'] is True for m in r['materials']) and
                   any(m['valid'] is None for m in r['materials']) for r in rows)
    return dict(queries=len(rows),returned_materials=len(materials),valid_materials=valid,
                invalid_materials=len(materials)-valid-unknown,ungraded_materials=unknown,
                valid_queries=queries,unresolved_queries=unresolved,
                valid_material_rate=valid/len(materials) if not unknown else None,
                valid_query_rate=queries/len(rows) if not unresolved else None,
                material_rate_bounds=[valid/len(materials),(valid+unknown)/len(materials)],
                query_rate_bounds=[queries/len(rows),(queries+unresolved)/len(rows)])


def main():
    lock=(HERE/'run.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((HERE/'manifest.json').read_text())
    if not manifest.get('approved') or not manifest.get('approval'):
        raise RuntimeError('This grading configuration and USD2 cap need explicit approval before launch')
    for path,digest in manifest['hashes'].items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Frozen source/input changed: '+path)
    client=load('hybrid_budget_client',HERE/'budget_client.py')
    judge=load('hybrid_judging',HERE/'judging.py')
    client.HERE=HERE
    guard=client.Guard(manifest)
    source=json.loads((HERE/'inputs.json').read_text())
    rows=json.loads((HERE/'results.json').read_text()) if (HERE/'results.json').exists() else source['rows']
    jobs=json.loads((HERE/'jobs.json').read_text())
    def complete(prompt):
        result=guard.post(manifest['judge_model'],dict(model=manifest['judge_model'],
            messages=[{'role':'user','content':prompt}],temperature=0,max_tokens=2048,
            thinking={'type':'disabled'},response_format={'type':'json_object'}))
        try:
            if result['choices'][0]['finish_reason']=='length':
                raise ValueError('Truncated completion')
            return json.loads(result['choices'][0]['message']['content'])
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise MalformedCompletion(str(exc)) from exc
    def checkpoint():
        save(HERE/'results.json',rows)
        save(HERE/'summary.json',summarize(rows))
        cost=guard.summary()
        cost.update(prior_estimated_usd=manifest['prior_estimated_usd'],
                    cumulative_estimated_usd=manifest['prior_estimated_usd']+cost['estimated_peak_usd'],
                    cumulative_exposure_usd=manifest['prior_exposure_usd']+cost['exposure_usd'])
        save(HERE/'cost_summary.json',cost)
    try:
        manifest['status']='running';save(HERE/'manifest.json',manifest)
        calibrator=judge.CalibratedJudge(complete)
        try:
            calibrator.calibrate()
        finally:
            save(HERE/'calibration.json',calibrator.calibration)
        for number,job in enumerate(jobs):
            row=rows[job['query_index']];material=row['materials'][job['material_index']]
            if material['valid'] is not None or material.get('attempts',0)>=2:
                continue
            c=row['case']
            prompt=judge.judge_prompt(row['motion'],c['action'],c['target'],c['target_argument'],[material['text']])
            for attempt in range(material.get('attempts',0),2):
                current=prompt
                if attempt:
                    current+='\nYour previous response failed validation: '+material['errors'][-1]+'. Return complete JSON. Copy short quotes exactly from the supplied text; do not paraphrase quotes.'
                try:
                    response=complete(current)
                    grade=judge.parse_grades(response,c['action'],c['target'],c['target_argument'],[material['text']])[0]
                except (MalformedCompletion, judge.InvalidJudgement) as exc:
                    material['attempts']=attempt+1
                    material.setdefault('errors',[]).append(type(exc).__name__+': '+str(exc))
                    checkpoint()
                    continue
                material.update(valid=grade['valid'],grade=grade,grade_source='hybrid_supplement',attempts=attempt+1)
                checkpoint();break
            print(number+1,'/',len(jobs),summarize(rows),flush=True)
        manifest['status']='completed' if summarize(rows)['ungraded_materials']==0 else 'completed_with_gaps'
        manifest['completed_unix']=time.time()
    except BaseException as exc:
        manifest.update(status='stopped',stop_reason=type(exc).__name__+': '+str(exc))
        raise
    finally:
        checkpoint();save(HERE/'manifest.json',manifest)
        guard.db.close()


if __name__=='__main__':main()
