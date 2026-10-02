from scripts.analyze_streaming_performance import ChunkMetrics, TurnMetrics


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
