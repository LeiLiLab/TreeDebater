"""Corrected policy keeps the native refinement loop and raw fallback semantics."""
import json
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest

D=Path(__file__).resolve().parents[1]/'experiments/incremental_planning'
sys.path.insert(0,str(D))
import compare_native_revision_v57 as probe
from streaming.config import OutputConfig


def case():
    return dict(writing_task='Use evidence.',material=dict(unpublished_draft='All original paragraphs.',
        feedback='Explain access.',evidence=[dict(title='Trial',content='Funding increased access.')]),
        request=dict(messages=[dict(role='system',content='Assigned speaker.'),dict(role='user',content='old')]))


def test_joint_paragraph_prompt_contains_global_context_and_exact_local_scope():
    messages=probe.paragraph_prompt(case(),'Current proposal.',40,['Already spoken.'],'Next untouched.','Original source.')
    data=json.loads(messages[-1]['content'].split('Context and material (data):\n')[1])
    assert messages[0]['content']=='Assigned speaker.'
    assert data['feedback']=='Explain access.' and data['evidence'][0]['title']=='Trial'
    assert data['unpublished_draft']=='All original paragraphs.'
    assert data['current_paragraph']==dict(source='Original source.',current_proposal='Current proposal.',target_words=40,
        committed_preceding=['Already spoken.'],next_paragraph_readonly='Next untouched.')


@pytest.mark.parametrize('raw_words,expected_rewrites',[(40,2),(20,0)])
def test_native_loop_still_retries_until_actual_audio_fits_or_keeps_fitting_raw(monkeypatch,raw_words,expected_rewrites):
    import tts_streaming as tts
    proposals=['First '+'word '*29,'Second '+'word '*19]
    calls=[]
    def revise(client,text,n_words,prev_texts,next_chunk_text='',**kwargs):
        calls.append(probe.paragraph_prompt(case(),text,n_words,prev_texts,next_chunk_text,kwargs['source_text']))
        return proposals[len(calls)-1].strip()
    monkeypatch.setattr(tts,'_revise_to_n_words',revise)
    monkeypatch.setattr(tts,'_estimate_duration',lambda text,**kw:float(len(text.split())))
    monkeypatch.setattr(tts,'_tts_with_retry',lambda client,text,*a,**kw:dict(audio_seconds=float(len(text.split()))))
    cfg=OutputConfig(adaptive_delivery=True,max_parallel_tts=1,allow_expansion=True)
    ctx=tts._ChunkRefineContext(client=None,original_text=('Original '+'word '*(raw_words-1)).strip(),
        target_s=20,tol_s=1,tol_upper_s=1,prev_texts=['Delivered.'],next_chunk_text='Later.',voice='echo',
        max_ref=3,kickoff_iter=1,kickoff_kind='',config=cfg)
    try:
        tts._refine_worker(ctx,'normal')
        assert len(calls)==expected_rewrites
        assert ctx.chosen_tts_out['audio_seconds']==20
        assert len(ctx.candidates)==expected_rewrites+1
        assert ctx.chosen_cand.intra_iter==expected_rewrites
    finally:
        ctx.stop_event.set();ctx.executor.shutdown(wait=True)


def test_experiment_keeps_old_native_correction_settings():
    cfg=probe.config()
    assert cfg.max_refinements==10 and cfg.early_max_refinements==3
    assert cfg.allow_expansion and cfg.adaptive_delivery and cfg.max_parallel_tts==8
    assert cfg.speed_adjust_min==cfg.speed_adjust_max==1
