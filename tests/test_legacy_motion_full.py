"""The full experiment must deliver every callback, not stop at first audio."""
import sys
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from pydub import AudioSegment
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/incremental_planning'))
import benchmark_legacy_motion_full as run


@pytest.mark.parametrize('module', ['benchmark_legacy_motion_full', 'benchmark_legacy_motion_full_v3'])
def test_full_run_delivers_all_chunks_and_keeps_revision_controls(tmp_path, monkeypatch, module):
    run = importlib.import_module(module)
    class Player:
        config = SimpleNamespace()
        planner = SimpleNamespace(config=SimpleNamespace(mode='legacy'), state={}, plan='')
        debate_thoughts = []
        def opening_generation(self, *args, **kwargs):
            texts = ['First complete paragraph.', 'Final complete paragraph.']
            for i, text in enumerate(texts):
                path = Path(self.audio_output_dir) / f'{i}.mp3'
                AudioSegment.silent(duration=100).export(path, format='mp3')
                self.tts_chunk_callback(i, path, text, .1)
            return '\n\n'.join(texts)
    player = Player()
    for name in ['_get_response', '_get_revision_suggestion', '_length_adjust',
                 '_get_feedback_from_audience', 'claim_selection', '_add_additional_info']:
        setattr(player, name, Mock())
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run, 'OUTPUT', tmp_path / 'run')
    monkeypatch.setattr(run, 'require_room', Mock())
    guard = SimpleNamespace(request_id=1, artifact={'external_calls': [], 'blocked_dispatches': []}, finish=Mock())
    monkeypatch.setattr(run, 'AudioGuard', Mock(return_value=guard))
    client = Mock(); client.summary.return_value = {}
    case = dict(id='test', kind='matched_historical_context', stage='opening', side='for', budget=240)
    result = run.run_arm(player, [], {'estimated_listener_backlog_seconds': 0}, case, 0, 'legacy', client, None)
    assert result['status'] == 'returned' and result['error'] is None
    assert result['chunks'] == 2 and result['audio_seconds'] == .2
    assert result['answer'] == result['returned_text']
    assert result['output_config']['max_refinements'] == 10
    assert result['output_config']['early_max_refinements'] == 3
    assert result['output_config']['allow_expansion']
    guard.finish.assert_called_once()
