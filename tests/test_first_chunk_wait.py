from scripts.analyze_streaming_performance import (
    ChunkMetrics, TurnMetrics, load_final_audio_durations,
    apply_batch_sequential_metrics, generate_summary, print_summary,
)


def test_first_chunk_wait_includes_handoff_and_decode():
    turn = TurnMetrics(stage='opening', side='against', turn_start=110,
                       previous_opponent_audio_end=100)
    turn.chunks[1] = ChunkMetrics(chunk_idx=1, playback_start_time=135)
    assert turn.time_to_first_chunk == 35


def test_missing_transition_is_not_zero_wait():
    turn = TurnMetrics(stage='opening', side='for')
    turn.chunks[1] = ChunkMetrics(chunk_idx=1, playback_start_time=135)
    assert turn.time_to_first_chunk is None
    turn.previous_opponent_audio_end = 100
    turn.chunks.clear()
    assert turn.time_to_first_chunk is None


def test_batch_audio_duration_uses_untrimmed_or_final_trimmed_value(tmp_path):
    log = tmp_path / 'batch.log'
    log.write_text(
        '[timing] phase=tts_wall_clock duration_s=5 stage=opening side=for audio_duration_s=200\n'
        '[timing] phase=tts_wall_clock duration_s=6 stage=opening side=against audio_duration_s=260\n'
        '[timing] phase=tts_trim_wall_clock duration_s=2 stage=opening side=against audio_duration_s=239\n'
    )
    assert load_final_audio_durations(log) == {('opening','for'):200,('opening','against'):239}


def test_incomplete_batch_without_audio_can_be_reported(tmp_path, capsys):
    log = tmp_path / 'failed.log'
    log.write_text('')
    turns = {('opening','for'):TurnMetrics(stage='opening',side='for',turn_start=100,
                                         turn_end=110,streaming_tts=False,streaming_listen=False)}
    apply_batch_sequential_metrics(turns,log,False,False,False)
    summary = generate_summary(turns)
    print_summary(summary)
    assert summary['turns']['opening_for']['audio_duration'] is None
    assert summary['turns']['opening_for']['time_to_first_chunk'] is None
    assert summary['turns']['opening_for']['estimated_batch_preparation_gap_s'] == 10
    assert 'audio N/A' in capsys.readouterr().out
