"""First-audio probes preserve Legacy prestarts and drain them on cancellation."""
import sys
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from test_full_speech import audio as audio
import tts_streaming as tts

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments/incremental_planning'))
import benchmark_legacy_motion_timing as probe


def test_probe_keeps_prestart_rewrite_and_drains_after_first_audio(audio, monkeypatch, tmp_path):
    query, encoded = audio
    started, published, finished = threading.Event(), threading.Event(), threading.Event()
    monkeypatch.setattr(tts, '_estimate_duration', lambda text, **kw: 100.)
    def rewrite(*args, **kwargs):
        started.set()
        assert published.wait(3)
        finished.set()
        return 'A revised later paragraph.'
    monkeypatch.setattr(tts, '_revise_to_n_words', rewrite)
    def synthesize(*args, **kwargs):
        assert started.wait(3)
        return encoded()
    query.side_effect = synthesize
    seen = []
    def emit(index, path, text, duration):
        seen.append(text)
        published.set()
        raise probe.FirstAudioMeasured()
    config = replace(probe.CONFIGS['legacy'], min_chunk_words=1, normalize_seams=False)
    with pytest.raises(probe.FirstAudioMeasured):
        tts.run_pipeline(Mock(), ['Original opening.', 'Second paragraph.', 'Last paragraph.'],
            total_budget_s=120, out_dir=tmp_path, config=config, on_chunk=emit)
    assert seen == ['Original opening.']
    assert started.is_set() and finished.is_set()


def test_legacy_probe_reuses_claims_without_model_call(tmp_path, monkeypatch, historical_motion_archive):
    monkeypatch.setenv('DEBATE_LLM_API_BASE', 'http://127.0.0.1:4000/v1')
    monkeypatch.setattr(probe, 'OUTPUT', tmp_path)
    import json
    root, motion_file = historical_motion_archive
    monkeypatch.setattr(probe, 'ROOT', root)
    monkeypatch.setattr(probe, 'MOTIONS', motion_file)
    monkeypatch.setattr(probe, 'LEDGER', tmp_path)
    source = tmp_path / 'flat-motion-overlap-v2/motion_01_for_claims.json'
    source.parent.mkdir()
    source.write_text(json.dumps(dict(definition='A test motion.', claims=[
        dict(claim=f'Claim {i}', argument='Hypothetical reasoning.') for i in range(3)])))
    client = Mock()
    cases, _ = probe.build_cases()
    player = probe.prepare_base(cases[0], client)
    client.complete.assert_not_called()
    assert len(cases) == 18
    assert player.planner.config.mode == 'legacy'
    assert player.config.single_pass_revision is False
    assert player.high_quality_evidence_pool == []
    config = probe.CONFIGS['legacy']
    assert config.max_refinements == 10 and config.early_max_refinements == 3
    assert config.allow_expansion and config.max_parallel_tts == 8
    assert not config.first_chunk_local_tempo
    assert config.speed_adjust_min == config.speed_adjust_max == 1
