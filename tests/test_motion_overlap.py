"""Natural opening routing and matched-context experiment constraints."""
import json
import sys
import threading
from pathlib import Path
from unittest.mock import Mock
from dataclasses import replace
from test_full_speech import audio, overlap_speaker


def test_natural_opening_is_not_shortened_and_tail_overlaps(audio, tmp_path):
    p, history = overlap_speaker(tmp_path)
    p.streaming_output_config = replace(p.streaming_output_config, speech_mode='overlap_prefix')
    p.config.single_pass_revision = False
    opening = 'Identity checks may deter some abuse, but vulnerable speakers need a safe way to participate without exposing their legal names to other users.'
    draft = opening+'\n\nCosts and implementation need scrutiny.'
    p._get_revision_suggestion.return_value = ('Preserve qualifications.', [], '', draft)
    first_ready = threading.Event()
    calls = []
    def revise(text, *args, **kwargs):
        calls.append((text, kwargs))
        if 'frozen_prefix' not in kwargs:
            return draft
        assert kwargs['frozen_prefix'] == opening
        assert first_ready.wait(3)
        return 'Costs and implementation still need scrutiny.'
    p._length_adjust.side_effect = revise
    p.helper_client = Mock(side_effect=AssertionError('Natural paragraph should not use short-prefix helper'))
    def emit(i, path, text, seconds):
        if i == 0:
            assert text == opening
            first_ready.set()
    p.tts_chunk_callback = emit
    result = p.speak('Whole speech.', 240, time_control=True, history=history)
    assert result.startswith(opening)
    assert len(calls) == 2
    assert calls[-1][0] == 'Costs and implementation need scrutiny.'
    trace=json.loads(next(tmp_path.glob('*_chunks/overlap_prefix.json')).read_text())
    assert trace['mode']=='flat_full_script_natural_prefix'
    assert trace['whole_feedback_passes']==2
    assert trace['tail_revision_start_seconds'] < trace['chunks'][0]['ready_seconds'] < trace['tail_revision_end_seconds']
    assert 'self-contained substantive point' not in p._get_response.call_args.args[0][-1]['content']


def test_experiment_contexts_have_no_future_and_audio_controls(historical_motion_archive, monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/incremental_planning'))
    import benchmark_motion_overlap_v2 as run
    root, motion_file = historical_motion_archive
    monkeypatch.setattr(run, 'ROOT', root)
    monkeypatch.setattr(run, 'MOTIONS', motion_file)
    cases, sources=run.build_cases()
    assert len(cases)==18 and len(sources)==3
    for i,case in enumerate(cases):
        assert len(case['history'])==i%6
        assert case['budget']==(120 if case['stage']=='closing' else 240)
        if case['history']: assert case['history'][-1]['side']!=case['side']
    for config in run.CONFIGS.values():
        assert not config.first_chunk_local_tempo
        assert config.max_refinements==config.early_max_refinements==0
        assert config.speed_adjust_min==config.speed_adjust_max==1


def test_generated_reasoning_is_not_misrepresented_as_sourced_evidence(tmp_path, monkeypatch):
    import benchmark_motion_overlap_v2 as run
    plan={'definition':'A test motion.', 'claims':[{'claim':f'Claim {i}.','argument':'Logical hypothesis.'} for i in range(3)]}
    client=Mock()
    client.complete.return_value=json.dumps(plan)
    monkeypatch.setattr(run,'OUTPUT',tmp_path)
    monkeypatch.setenv('DEBATE_LLM_API_BASE','http://127.0.0.1:4000/v1')
    case={'motion_number':1,'motion':'A test motion.','side':'for'}
    p=run.prepare_base(case,client)
    assert len(p.claim_pool)==3
    assert p.high_quality_evidence_pool==[]
    assert all(c[0]['arguments']==[] for c in p.claim_pool)
    assert 'not a prior speech or sourced evidence' in p.conversation[-1]['content']
