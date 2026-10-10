"""Make the source package available for offline tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


import pytest


@pytest.fixture
def historical_motion_archive(tmp_path):
    """Three complete transcripts without reading local experiment outputs."""
    import json
    motions = ['First motion', 'Second motion', 'Third motion']
    motion_file = tmp_path / 'motions.txt'
    motion_file.write_text('\n'.join(motions))
    for number, motion in enumerate(motions, 1):
        path = tmp_path / f'experiments/debater_baseline_gemma4/motion_{number:02}_baseline_for/result.json'
        path.parent.mkdir(parents=True)
        history = [dict(stage=stage, side=side, content=f'{stage}: {side}')
                   for stage in ('opening', 'rebuttal', 'closing') for side in ('for', 'against')]
        path.write_text(json.dumps(dict(status='complete', config=dict(env=dict(motion=motion)), history=history)))
    return tmp_path, motion_file


@pytest.fixture
def word_length_modes(monkeypatch):
    """Offline delivery tests use deterministic estimators; backend dispatch is tested separately."""
    from utils import constants
    monkeypatch.setattr(constants, 'LENGTH_MODE_FOR_DRAFT', 'words')
    monkeypatch.setattr(constants, 'TIME_MODE_FOR_STATEMENT', 'time')
