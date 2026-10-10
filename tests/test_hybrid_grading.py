"""Frozen grading coverage and publication/approval gates (no API access)."""
import importlib.util
import json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'experiments/retrieval_hybrid_grading'
FIXTURES=Path(__file__).parent/'fixtures/experiments/retrieval_hybrid_grading'


def module():
    spec=importlib.util.spec_from_file_location('hybrid_grading',HERE/'run.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_partition_covers_every_missing_grade_exactly_once():
    rows=json.loads((FIXTURES/'inputs.json').read_text())['rows']
    jobs=json.loads((FIXTURES/'jobs.json').read_text())
    missing={(qi,mi) for qi,r in enumerate(rows) for mi,m in enumerate(r['materials']) if m['valid'] is None}
    targets=[(j['query_index'],j['material_index']) for j in jobs]
    assert len(missing)==len(targets)==76 and set(targets)==missing
    assert sum(len(r['materials']) for r in rows)==90


def test_partial_scores_are_not_published_as_overall_rates():
    rows=json.loads((FIXTURES/'inputs.json').read_text())['rows']
    s=module().summarize(rows)
    assert s['valid_material_rate'] is None and s['valid_query_rate'] is None
    assert s['valid_materials']==13 and s['valid_queries']==8
    for row in rows:
        for m in row['materials']:
            if m['valid'] is None:m['valid']=False
    s=module().summarize(rows)
    assert s['valid_material_rate']==13/90 and s['valid_query_rate']==8/30


def test_unapproved_launch_stops_before_client_load(tmp_path,monkeypatch):
    mod=module();monkeypatch.setattr(mod,'HERE',tmp_path)
    (tmp_path/'manifest.json').write_text(json.dumps(dict(approved=False)))
    def forbidden(*a,**k):raise AssertionError('client must not be loaded')
    monkeypatch.setattr(mod,'load',forbidden)
    with pytest.raises(RuntimeError,match='explicit approval'):mod.main()
