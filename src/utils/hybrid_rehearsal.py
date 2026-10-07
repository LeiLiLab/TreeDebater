"""Offline hybrid retrieval with separate target-anchor and reply representations.

Scores measure relevance, not action validity. All relation/scope verdicts remain
subject to the same strict cache checks as lexical-only retrieval.
"""
import numpy as np
from .local_rehearsal import LocalRehearsalRetriever
from . import rehearsal_index_cache as disk_cache


class HybridRehearsalRetriever(LocalRehearsalRetriever):
    def __init__(self, encoder, semantic_min_score=.35, max_per_anchor=None, cache_dir=disk_cache.DEFAULT_CACHE_DIR):
        super().__init__()
        if not 0 <= semantic_min_score <= 1:
            raise ValueError('semantic_min_score must be between zero and one')
        if max_per_anchor is not None and (type(max_per_anchor) is not int or max_per_anchor < 1):
            raise ValueError('max_per_anchor must be a positive integer or None')
        self.max_per_anchor = max_per_anchor
        self.encoder = encoder
        self.semantic_min_score = semantic_min_score
        self.anchor_index = LocalRehearsalRetriever()
        self.anchor_vectors = None
        self.material_vectors = None
        self.hybrid_signature = None
        self.cache_dir = cache_dir
        self.disk_cache_status = 'unused'

    def _index(self, records):
        signature = tuple(r['key'] for r in records)
        if self.hybrid_signature == signature:
            self.records = records
            return False
        path = disk_cache.cache_path(self.cache_dir, self.encoder, signature) if records else None
        cached = disk_cache.load(path, len(records))
        if cached is not None:
            meta, self.anchor_vectors, self.material_vectors = cached
            self.records = records
            self.signature = self.hybrid_signature = signature
            self.idf, self.postings = meta['lexical']['idf'], meta['lexical']['postings']
            self.anchor_index.idf = meta['anchor_lexical']['idf']
            self.anchor_index.postings = meta['anchor_lexical']['postings']
            self.anchor_index.records = records
            self.anchor_index.signature = None
            self.disk_cache_status = 'hit'
            return True
        # A failed rebuild must not leave the previous signature paired with
        # newly built lexical tables and old vectors.
        self.hybrid_signature = None
        super()._index(records)
        anchors = [dict(r, material={'claim':'','argument':''}) for r in records]
        self.anchor_index._index(anchors)
        if records:
            anchor_texts = [r['candidate']['claim'] for r in records]
            material_texts = ['\n'.join((r['material']['claim'],r['material']['argument'])) for r in records]
            vectors = self.encoder.encode(anchor_texts+material_texts)
            norms = np.linalg.norm(vectors,axis=1,keepdims=True)
            if not np.isfinite(vectors).all() or np.any(norms == 0):
                raise ValueError('Invalid document embeddings')
            vectors = vectors/norms
            self.anchor_vectors, self.material_vectors = np.split(vectors,2)
        else:
            self.anchor_vectors = self.material_vectors = None
        # Only commit after successful encoding; a failed rebuild can be retried.
        self.hybrid_signature = self.signature
        self.disk_cache_status = 'written' if records and disk_cache.save(path, self) else 'disabled_or_unavailable'
        return True

    def prepare(self, own, opponent, side, oppo_side, attack):
        return self._index(self._collect(own,opponent,side,oppo_side,attack))

    def _scores(self, target, argument, caches):
        if not self.records:
            self.lexical_scores = self.semantic_scores = np.empty(0,dtype=np.float32)
            return self.lexical_scores, 0
        texts = [target] + ([target+'\n'+argument] if argument else [])
        query_vectors = self.encoder.encode(texts)
        norms = np.linalg.norm(query_vectors,axis=1,keepdims=True)
        if not np.isfinite(query_vectors).all() or np.any(norms == 0):
            raise ValueError('Invalid query embeddings')
        query_vectors = query_vectors/norms
        query = query_vectors[0] if not argument else .8*query_vectors[0]+.2*query_vectors[1]
        # Attack/rebut records point to the opponent claim being answered, not
        # merely to a topically similar response. Material content is secondary.
        anchor = np.einsum('ij,j->i',self.anchor_vectors,query)
        material = np.einsum('ij,j->i',self.material_vectors,query)
        self.semantic_scores = .85*anchor+.15*material
        lexical = np.maximum(self.anchor_index._lexical(target), self._lexical(target))
        if argument:
            lexical = .8*lexical+.2*np.maximum(self.anchor_index._lexical(argument),self._lexical(argument))
        self.lexical_scores = lexical
        return .7*np.maximum(0,self.semantic_scores)+.3*lexical, len(self.records)

    def _eligible(self, score, index, min_score):
        return self.lexical_scores[index] >= min_score or self.semantic_scores[index] >= self.semantic_min_score

    def retrieve(self, *args, **kwargs):
        kwargs.setdefault('max_per_anchor', self.max_per_anchor)
        result = super().retrieve(*args, **kwargs)
        self.stats.update(mode='hybrid',semantic_min_score=self.semantic_min_score,
                          disk_cache_status=self.disk_cache_status,
                          semantic_candidates=int(np.sum(self.semantic_scores >= self.semantic_min_score)),
                          # Hybrid never consumes remote or model-ambiguous cached vectors.
                          cached_vector_matches=0, local_vector_matches=len(self.records))
        return result
