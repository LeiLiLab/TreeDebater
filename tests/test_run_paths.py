"""Run IDs remain unique across launches and orphaned output directories."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from utils.run_paths import reserve_run_path


def test_skips_artifacts_without_main_logs(tmp_path):
    (tmp_path / "120.log").touch()
    (tmp_path / "123.json").write_text("{}")
    (tmp_path / "125_outputs").mkdir()
    (tmp_path / "129_watch").mkdir()
    path = Path(reserve_run_path(tmp_path))
    assert path.name == "130.log"
    assert path.is_file()
    assert (tmp_path / "123.json").read_text() == "{}"


def test_parallel_launches_reserve_distinct_paths(tmp_path):
    with ProcessPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(reserve_run_path, [str(tmp_path)] * 32))
    assert len(set(paths)) == 32
    assert all(Path(path).is_file() for path in paths)
