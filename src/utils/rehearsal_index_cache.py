"""Atomic, non-pickle storage for derived rehearsal indexes; never stores live nodes."""
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / '.cache' / 'rehearsal_indexes'
VERSION = 1
LOG = logging.getLogger(__name__)


def cache_path(directory, encoder, signature):
    if directory is None or not callable(getattr(encoder, 'cache_identity', None)):
        return None
    try:
        identity = encoder.cache_identity()
        if identity is None:
            return None
        code = {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ('local_rehearsal.py', 'hybrid_rehearsal.py', 'local_encoder.py', 'rehearsal_index_cache.py')}
        key = hashlib.sha256(json.dumps([VERSION, identity, code, signature], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        return Path(directory).expanduser() / (key + '.npz')
    except (OSError, ValueError, ImportError) as exc:
        LOG.warning('Rehearsal disk cache unavailable: %s', exc)
        return None


def lexical_state(index):
    return dict(idf=index.idf, postings=index.postings)


def validate_lexical(state, rows):
    assert isinstance(state, dict)
    assert set(state) == {'idf', 'postings'}
    assert isinstance(state['idf'], dict) and isinstance(state['postings'], dict)
    assert set(state['idf']) == set(state['postings'])
    for term, weight in state['idf'].items():
        assert isinstance(term, str) and np.isfinite(weight) and weight > 0
        for idx, score in state['postings'][term]:
            assert type(idx) is int and 0 <= idx < rows and np.isfinite(score) and score > 0


def load(path, rows):
    if path is None:
        return None
    try:
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata'].item()))
            assert isinstance(meta, dict)
            assert meta['version'] == VERSION and meta['rows'] == rows
            anchors, materials = data['anchors'], data['materials']
            assert anchors.ndim == 2 and anchors.shape == materials.shape and anchors.shape[0] == rows
            assert anchors.shape[1] > 0
            for vectors in (anchors, materials):
                assert np.isfinite(vectors).all()
                assert np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5)
            validate_lexical(meta['lexical'], rows)
            validate_lexical(meta['anchor_lexical'], rows)
            return meta, anchors, materials
    except (OSError, ValueError, KeyError, TypeError, AssertionError, EOFError, zipfile.BadZipFile) as exc:
        LOG.warning('Ignoring invalid rehearsal index cache %s: %s', path, exc)
        return None


def save(path, index):
    if path is None:
        return False
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(version=VERSION, rows=len(index.records), lexical=lexical_state(index),
                    anchor_lexical=lexical_state(index.anchor_index))
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, metadata=json.dumps(meta), anchors=index.anchor_vectors, materials=index.material_vectors)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        return True
    except OSError as exc:
        LOG.warning('Cannot persist rehearsal index %s: %s', path, exc)
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
