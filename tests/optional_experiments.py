"""Keep local experiment tests optional without hiding core test failures."""
from pathlib import Path

import pytest


def require_local_experiment(relative_path, *, module=False):
    path = Path(__file__).resolve().parents[1] / relative_path
    if not path.exists():
        pytest.skip(
            f"Local experiment material is not distributed: {relative_path}",
            allow_module_level=module,
        )
    return path
