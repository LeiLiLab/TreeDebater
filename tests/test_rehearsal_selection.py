"""Saved preparation ranking is local, stable and preserves full retrieval inputs."""
import copy
from types import SimpleNamespace

import pytest

from streaming.rehearsal_selection import select_saved_claims


def test_full_pool_rank_before_limit_preserves_arguments_and_original_scores():
    pool = [[dict(claim=label, minimax_search_score=score, arguments=[label + ' mechanism'])]
            for label, score in [('Low', 1), ('High', 5), ('Tie', 5), ('Medium', 3)]]
    original = copy.deepcopy(pool)
    player = SimpleNamespace(rehearsal_claim_pool=pool, definition='Saved definition')
    artifact = select_saved_claims(player, limit=3, main_count=2)
    assert player.main_claims_content == ['High', 'Tie']
    assert [g[0]['minimax_search_score'] for g in player.claim_pool] == [5, 5, 3]
    assert player.main_claims[0]['arguments'] == ['High mechanism']
    assert [c['pool_index'] for c in artifact['candidates']] == [1, 2, 3]
    player.main_claims[0]['arguments'].append('Private change')
    assert pool == original


def test_missing_or_invalid_scores_rank_last_without_fabricating_a_score():
    pool = [[dict(claim='Unscored')], [dict(claim='Scored', minimax_search_score=-1)],
            [dict(claim='Invalid', minimax_search_score=True)]]
    player = SimpleNamespace(rehearsal_claim_pool=pool, definition=None)
    artifact = select_saved_claims(player)
    assert player.main_claims_content == ['Scored', 'Unscored', 'Invalid']
    assert artifact['claims'][1]['minimax_search_score'] is None
    assert 'minimax_search_score' not in player.claim_pool[1][0]
    with pytest.raises(ValueError, match='no usable'):
        select_saved_claims(SimpleNamespace(rehearsal_claim_pool=[[], {}]))
