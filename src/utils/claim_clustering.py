"""Deterministic average-linkage clustering without model or API dependencies."""
import numpy as np


def embedding_text(claim):
    return f"Claim: {claim['claim']}\nExplanation: {claim.get('explanation', '')}"


def cluster_claim_embeddings(claims, embeddings, max_groups=10, similarity_threshold=0.8):
    if isinstance(max_groups, bool) or not isinstance(max_groups, int) or max_groups < 1:
        raise ValueError('max_groups must be a positive integer')
    if not -1 <= similarity_threshold <= 1:
        raise ValueError('similarity_threshold must be between -1 and 1')
    if not claims:
        return [], {'groups': [], 'merges': [], 'similarity_matrix': []}
    vectors = np.asarray(embeddings, dtype=float)
    if vectors.ndim != 2 or vectors.shape[0] != len(claims) or vectors.shape[1] == 0:
        raise ValueError('One nonempty embedding is required per claim')
    norms = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(vectors).all() or not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError('Embeddings must be finite and nonzero')
    vectors = vectors / norms[:, None]
    similarities = np.clip(vectors @ vectors.T, -1, 1)
    # Canonical order makes equal-score choices independent of input ordering.
    order = sorted(range(len(claims)), key=lambda i: (embedding_text(claims[i]), i))
    groups = [[i] for i in order]
    merges = []
    while len(groups) > 1:
        candidates = [(float(similarities[np.ix_(a, b)].mean()), i, j)
                      for i, a in enumerate(groups) for j, b in enumerate(groups) if i < j]
        score, i, j = max(candidates, key=lambda x: (x[0], -x[1], -x[2]))
        forced = score < similarity_threshold
        if forced and len(groups) <= max_groups:
            break
        merges.append({'left': groups[i][:], 'right': groups[j][:],
                       'average_similarity': score, 'forced_by_cap': forced})
        groups[i] += groups[j]
        del groups[j]
    # Preserve the existing highest-strength representative policy.
    for group in groups:
        group.sort(key=lambda i: (-claims[i].get('strength', 5), embedding_text(claims[i]), i))
    groups.sort(key=lambda g: (-claims[g[0]].get('strength', 5), embedding_text(claims[g[0]])))
    audit = {'max_groups': max_groups, 'similarity_threshold': similarity_threshold,
             'method': 'average_linkage_cosine', 'groups': groups, 'merges': merges,
             'similarity_matrix': similarities.tolist(),
             'representatives': [g[0] for g in groups]}
    return groups, audit
