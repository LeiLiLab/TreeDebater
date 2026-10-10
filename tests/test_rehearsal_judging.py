import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'experiments/retrieval_eval_v2/judging.py'
spec = importlib.util.spec_from_file_location('rehearsal_judging', path)
judging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(judging)


def verdict(effect='supports', scope=True):
    return {'grades': [{'id': 0, 'effect': effect, 'scope_compatible': scope,
                        'target_quote': 'target', 'material_quote': 'material', 'reason': 'An explanation.'}]}


def test_same_effect_has_opposite_validity_for_attack_and_reinforce():
    for action in ['attack', 'rebut']:
        assert not judging.parse_grades(verdict(), action, 'target', '', ['material'])[0]['valid']
        assert judging.parse_grades(verdict('challenges'), action, 'target', '', ['material'])[0]['valid']
    for action in ['reinforce', 'propose']:
        assert judging.parse_grades(verdict(), action, 'target', '', ['material'])[0]['valid']
        assert not judging.parse_grades(verdict('challenges'), action, 'target', '', ['material'])[0]['valid']


def test_scope_and_unknown_effect_do_not_count_as_valid():
    assert not judging.parse_grades(verdict('supports', False), 'reinforce', 'target', '', ['material'])[0]['valid']
    assert not judging.parse_grades(verdict('uncertain'), 'reinforce', 'target', '', ['material'])[0]['valid']


def test_generic_valid_flag_cannot_override_direction():
    result = verdict('supports')
    result['grades'][0]['valid'] = True
    assert not judging.parse_grades(result, 'attack', 'target', '', ['material'])[0]['valid']


def test_missing_quotes_or_ids_are_not_silent_negatives():
    with pytest.raises(judging.InvalidJudgement):
        judging.parse_grades({'grades': []}, 'attack', 'target', '', ['material'])
    result = verdict('challenges')
    result['grades'][0]['material_quote'] = 'invented content'
    with pytest.raises(judging.InvalidJudgement):
        judging.parse_grades(result, 'attack', 'target', '', ['material'])


def test_replay_cannot_be_graded_before_calibration():
    judge = judging.CalibratedJudge(lambda prompt: verdict())
    with pytest.raises(judging.InvalidJudgement, match='calibration'):
        judge.grade('motion', 'attack', 'target', '', ['material'])


def control_answer(prompt):
    data = json.loads(prompt.split('\n')[-1])
    control = next(c for c in judging.CONTROLS
                   if c['target'] == data['target'] and c['material'] == data['materials'][0]['text'])
    return {'grades': [{'id': 0, 'effect': control['effect'], 'scope_compatible': True,
                        'target_quote': data['target'], 'material_quote': data['materials'][0]['text'],
                        'reason': 'Control response.'}]}


def test_direction_confused_judge_fails_calibration_and_cannot_grade():
    def confused(prompt):
        result = control_answer(prompt)
        result['grades'][0]['effect'] = 'supports'
        return result
    judge = judging.CalibratedJudge(confused)
    with pytest.raises(judging.InvalidJudgement, match='failed action-direction'):
        judge.calibrate()
    with pytest.raises(judging.InvalidJudgement, match='calibration'):
        judge.grade('motion', 'attack', 'target', '', ['material'])


def test_calibration_gate_accepts_consistent_direction_controls():
    judge = judging.CalibratedJudge(control_answer)
    checks = judge.calibrate()
    assert len(checks) == 10 and all(c['passed'] for c in checks)
    control = judging.CONTROLS[0]
    assert judge.grade('motion', control['action'], control['target'], '', [control['material']])[0]['valid']
