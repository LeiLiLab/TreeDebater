"""Offline checks for bounded clustering and its prepare.py integration."""
import json
import logging
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest

from utils.claim_clustering import cluster_claim_embeddings, embedding_text
from test_rehearsal_retrieval import SRC, load_function


def claims(n):
    return [{'claim': f'claim {i:02d}', 'explanation': f'mechanism {i}', 'strength': i + 1} for i in range(n)]


def test_hard_cap_keeps_every_claim_even_when_all_similarities_are_low():
    groups, audit = cluster_claim_embeddings(claims(25), np.eye(25))
    assert len(groups) == 10
    assert sorted(i for g in groups for i in g) == list(range(25))
    assert all(m['forced_by_cap'] for m in audit['merges'])
    assert all(g[0] == max(g) for g in groups)


def test_near_duplicates_can_merge_below_cap_without_forcing_minimum():
    groups, audit = cluster_claim_embeddings(claims(3), [[1, 0], [1, 0], [0, 1]])
    assert {frozenset(g) for g in groups} == {frozenset([0, 1]), frozenset([2])}
    assert not audit['merges'][0]['forced_by_cap']


def test_average_linkage_does_not_merge_on_one_similar_pair_only():
    # A-B and B-C are close, but average(A-C, B-C) is below the threshold.
    angles = np.radians([0, 20, 45])
    groups, _ = cluster_claim_embeddings(claims(3), np.c_[np.cos(angles), np.sin(angles)],
                                         similarity_threshold=.85)
    assert {frozenset(g) for g in groups} == {frozenset([0, 1]), frozenset([2])}


def test_ties_are_stable_when_input_is_reordered():
    original = claims(12)
    vectors = np.eye(12)
    a, _ = cluster_claim_embeddings(original, vectors, max_groups=4)
    order = list(reversed(range(12)))
    b, _ = cluster_claim_embeddings([original[i] for i in order], vectors[order], max_groups=4)
    assert [[original[i]['claim'] for i in g] for g in a] == [
        [original[order[i]]['claim'] for i in g] for g in b]


@pytest.mark.parametrize('limit', [0, -1, 1.5, True])
def test_invalid_cap(limit):
    with pytest.raises(ValueError):
        cluster_claim_embeddings([], [], max_groups=limit)


@pytest.mark.parametrize('vectors', [[[0, 0]], [[float('nan'), 1]], [[1, 0], [0, 1]]])
def test_invalid_embeddings(vectors):
    with pytest.raises(ValueError):
        cluster_claim_embeddings(claims(1), vectors)


def test_empty_and_singleton():
    assert cluster_claim_embeddings([], [])[0] == []
    assert cluster_claim_embeddings(claims(1), [[1, 0]])[0] == [[0]]


def test_prepare_embeds_explanations_aligns_response_indices_and_saves_audit(tmp_path):
    client = NS(embeddings=NS(create=Mock(return_value=NS(data=[
        NS(index=1, embedding=[0., 1.]), NS(index=0, embedding=[1., 0.])]))))
    fn = load_function(SRC/'prepare.py', 'cluster_claims', {
        'OpenAI': lambda: client, 'EMBEDDING_MODEL': 'test-embedding',
        'embedding_text': embedding_text, 'cluster_claim_embeddings': cluster_claim_embeddings,
        'json': json, 'logger': logging.getLogger('test')}, cls='ClaimPool')
    subject = NS(max_claim_groups=1, cluster_similarity_threshold=.8,
                 cluster_audit_file=str(tmp_path/'audit.json'))
    assert fn(subject, claims(2)) == [[1, 0]]
    assert client.embeddings.create.call_args.kwargs['input'][0] == 'Claim: claim 00\nExplanation: mechanism 0'
    audit = json.loads((tmp_path/'audit.json').read_text())
    assert audit['embeddings'] == [[1., 0.], [0., 1.]]
    assert audit['groups'] == [[1, 0]]
    assert audit['merges'][0]['forced_by_cap']
