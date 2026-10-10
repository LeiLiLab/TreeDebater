"""Optional saved-score policy behind TreeDebater's claim-selection interface."""
import copy
import math


def select_saved_claims(player, *, limit=10, main_count=3):
    pool = getattr(player, 'rehearsal_claim_pool', None) or player.claim_pool
    candidates = []
    for index, group in enumerate(pool):
        if not isinstance(group, list) or not group or not isinstance(group[0], dict):
            continue
        root = group[0]
        if not isinstance(root.get('claim'), str) or not root['claim'].strip():
            continue
        score = root.get('minimax_search_score')
        valid = type(score) in (int, float) and math.isfinite(score)
        candidates.append((index, group, score if valid else None))
    if not candidates:
        raise ValueError('Saved rehearsal pool has no usable claim groups')
    candidates.sort(key=lambda item: (item[2] is None, -(item[2] or 0), item[0]))
    chosen = candidates[:limit]
    player.claim_pool = [copy.deepcopy(group) for _, group, _ in chosen]
    player.main_claims = [group[0] for group in player.claim_pool[:main_count]]
    player.main_claims_content = [claim['claim'] for claim in player.main_claims]
    return dict(definition=player.definition,
        claims=[dict(claim=group[0]['claim'], minimax_search_score=score, pool_index=index)
                for index, group, score in chosen[:main_count]],
        candidates=[dict(claim=group[0]['claim'], minimax_search_score=score, pool_index=index)
                    for index, group, score in chosen],
        ranking='Saved root minimax score descending; unscored roots last; stable ties. '
                'Internal preparation preference, not verified evidence or argument truth.')
