"""Persistent indexes must reproduce rankings without serializing live tree nodes."""
from unittest.mock import Mock
import numpy as np
import pytest

from utils.hybrid_rehearsal import HybridRehearsalRetriever
from utils import rehearsal_index_cache as disk
from test_hybrid_rehearsal import setup, run
from test_rehearsal_relations import pool


def fixture(directory, identity='model-v1'):
    _, encoder, anchor, material, query = setup()
    encoder.cache_identity = lambda: {'model': identity}
    return HybridRehearsalRetriever(encoder, cache_dir=directory), encoder, anchor, material, query


def test_fresh_instance_loads_both_indexes_without_document_encoding(tmp_path):
    r, e, a, m, q = fixture(tmp_path)
    expected = run(r, a, q)
    scores = r._scores(q['target_claim'], '', [])[0].copy()
    second, enc, anchor, material, query = fixture(tmp_path)
    second.prepare([pool(anchor)], [], 'for', 'against', True)
    assert second.disk_cache_status == 'hit' and not enc.calls
    assert second.records[0]['node'] is material and material is not m
    actual = run(second, anchor, query)
    assert actual[0] == expected[0]
    np.testing.assert_array_equal(second._scores(query['target_claim'], '', [])[0], scores)
    assert all(batch == [query['target_claim']] for batch in enc.calls)
    assert len(list(tmp_path.glob('*.npz'))) == 1


@pytest.mark.parametrize('change', ['model', 'material', 'parent'])
def test_cache_invalidates_changed_identity_or_content(tmp_path, change):
    r, e, a, m, q = fixture(tmp_path); run(r, a, q)
    second, enc, anchor, material, query = fixture(tmp_path, 'model-v2' if change == 'model' else 'model-v1')
    if change == 'material': material.argument = 'Different premise'
    if change == 'parent': anchor.claim = 'A different target'
    second.prepare([pool(anchor)], [], 'for', 'against', True)
    assert second.disk_cache_status == 'written' and enc.calls
    assert len(list(tmp_path.glob('*.npz'))) == 2


def test_corrupt_cache_rebuilt_and_replaced(tmp_path):
    r, e, a, m, q = fixture(tmp_path); run(r, a, q)
    path = next(tmp_path.glob('*.npz')); path.write_bytes(b'not an archive')
    second, enc, anchor, material, query = fixture(tmp_path)
    assert run(second, anchor, query)[0]
    assert second.disk_cache_status == 'written'
    third, enc, anchor, material, query = fixture(tmp_path)
    third.prepare([pool(anchor)], [], 'for', 'against', True)
    assert third.disk_cache_status == 'hit' and not enc.calls


def test_unwritable_destination_does_not_break_retrieval(tmp_path):
    path = tmp_path / 'file'; path.write_text('not a directory')
    r, e, a, m, q = fixture(path)
    assert run(r, a, q)[0]
    assert not list(tmp_path.glob('*.tmp'))


def test_explicit_disable_never_persists(tmp_path):
    r, e, a, m, q = fixture(None)
    assert run(r, a, q)[0] and r.disk_cache_status == 'disabled_or_unavailable'


def test_local_weight_change_changes_encoder_cache_identity(tmp_path):
    from utils.local_encoder import LocalEncoder
    weight = tmp_path / 'model.safetensors'; weight.write_bytes(b'first')
    a = LocalEncoder(str(tmp_path)); before = a.cache_identity()
    weight.write_bytes(b'replacement weights')
    b = LocalEncoder(str(tmp_path)); after = b.cache_identity()
    assert before != after
    assert a.process is None and b.process is None


def test_failed_rebuild_can_return_to_previous_materials_without_mixed_indexes():
    r, e, a, m, q = fixture(None)
    expected = run(r, a, q)
    original_argument = m.argument
    original_encode = e.encode
    m.argument = 'Changed material'
    e.encode = Mock(side_effect=RuntimeError('temporary failure'))
    with pytest.raises(RuntimeError):
        run(r, a, q)
    m.argument = original_argument
    e.encode = original_encode
    assert run(r, a, q)[0] == expected[0]
