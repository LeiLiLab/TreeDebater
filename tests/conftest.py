"""Make the source package available for offline tests."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


import pytest


@pytest.fixture
def word_length_modes(monkeypatch):
    """Offline delivery tests use deterministic estimators; backend dispatch is tested separately."""
    from utils import constants
    monkeypatch.setattr(constants, 'LENGTH_MODE_FOR_DRAFT', 'words')
    monkeypatch.setattr(constants, 'TIME_MODE_FOR_STATEMENT', 'time')
