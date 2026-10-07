"""Hybrid regressions without downloading or invoking real models."""
from types import SimpleNamespace as NS
from unittest.mock import Mock
import logging
from contextlib import nullcontext
import numpy as np
import pytest
from utils.hybrid_rehearsal import HybridRehearsalRetriever
from utils.local_rehearsal import verdict_key
from test_rehearsal_retrieval import node, load_function, SRC
from test_rehearsal_relations import pool


class Encoder:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []
    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array([self.vectors.get(t,[0.,1.]) for t in texts],dtype=np.float32)


def setup():
    target = 'Labeling safeguards artists.'
    reply = node('Detection fails for edited images.', side='for')
    anchor = node('Labels protect creators.', [reply])
    vectors = {target:[1.,0.],anchor.claim:[1.,0.],reply.claim+'\nsupport':[.8,.2]}
    encoder = Encoder(vectors)
    return HybridRehearsalRetriever(encoder),encoder,anchor,reply,dict(action='attack',target_claim=target)


def run(r,a,q,**kw):
    return r.retrieve('motion',q,'for','against',[pool(a)],[],1,**kw)


def test_semantic_paraphrase_uses_parent_and_no_remote_vectors():
    r,e,a,m,q=setup()
    r.prepare([pool(a)],[],'for','against',True)
    documents = list(e.calls)
    info,matches=run(r,a,q,embedding_caches=[{q['target_claim']:[-1.,0.],a.claim:[1.,0.]}],min_score=.99)
    assert m.claim in info[0] and matches[0][3]==a.claim
    assert e.calls[:-1] == documents and e.calls[-1]==[q['target_claim']]
    assert not r.stats['index_rebuilt'] and r.stats['cached_vector_matches']==0
    assert r.stats['local_vector_matches']==1


def test_obvious_polarity_and_negative_verdict_override_similarity():
    r,e,a,m,q=setup(); q['target_claim']='Labels harm creators.'
    e.vectors[q['target_claim']]=[1.,0.]
    assert run(r,a,q)==([],[])
    q['target_claim']='Labeling safeguards artists.'
    c={'claim':a.claim,'argument':'support','parent_claim':''}
    cache={verdict_key('motion',q,c,{'claim':m.claim,'argument':'support'}):False}
    assert run(r,a,q,verdicts=cache)==([],[])


def test_material_change_rebuilds_vectors_and_query_argument_is_encoded():
    r,e,a,m,q=setup(); run(r,a,q)
    m.argument='New evidence'; q['target_argument']='Only publicly shared images.'
    run(r,a,q)
    assert r.stats['index_rebuilt']
    assert m.claim+'\nNew evidence' in e.calls[-2]
    assert e.calls[-1][-1]==q['target_claim']+'\n'+q['target_argument']
    a.position_status='superseded'
    assert run(r,a,q)==([],[])


def test_failed_document_encoding_can_retry_without_stale_index():
    r,e,a,m,q=setup(); original=e.encode
    e.encode=Mock(side_effect=RuntimeError('encoder unavailable'))
    with pytest.raises(RuntimeError): run(r,a,q)
    e.encode=original
    assert run(r,a,q)[0]


@pytest.mark.parametrize("anchor_limit", [None, 1])
def test_agent_hybrid_prepares_both_groups_and_never_calls_remote(monkeypatch, anchor_limit):
    import utils.local_encoder as module
    r,e,a,m,q=setup()
    monkeypatch.setattr(module,'get_local_encoder',lambda *args:e)
    forbidden=Mock(side_effect=AssertionError('remote call'))
    scope={'get_retrieval_from_rehearsal_tree':forbidden,
           'REMAINING_ROUND_NUM':{'opening_for':3},'logger':logging.getLogger('test'),
           'timed_phase':lambda *a,**k:nullcontext()}
    method=load_function(SRC/'ouragents.py','_retrieve_on_prepared_tree',scope,cls='TreeDebater')
    warm=load_function(SRC/'ouragents.py','_warm_rehearsal_indexes',{},cls='TreeDebater')
    p=NS(motion='motion',use_rehearsal_tree=True,prepared_tree_list=[pool(a)],prepared_oppo_tree_list=None,
         _get_prepared_tree=lambda side:[],side='for',oppo_side='against',status='opening',
         debate_tree=NS(get_all_nodes=lambda:[]),oppo_debate_tree=NS(get_all_nodes=lambda:[]),
         config=NS(rehearsal_mode='hybrid', rehearsal_max_per_anchor=anchor_limit),debate_thoughts=[],
         _get_embedding_from_cache=forbidden,_validate_rehearsal_candidates=forbidden)
    warm(p)
    assert set(p._local_rehearsal_indexes)=={('hybrid','attack'),('hybrid','support')}
    assert method(p,q)
    assert p.debate_thoughts[-1]['local_stats']['index_rebuilt'] is False
    assert p.debate_thoughts[-1]['local_stats']['max_per_anchor'] == anchor_limit
    p._local_rehearsal_indexes = {}
    assert method(p,q)
    assert p.debate_thoughts[-1]['local_stats']['max_per_anchor'] == anchor_limit
    forbidden.assert_not_called()


def diversity_fixture(max_per_anchor=1):
    replies=[node('Reply '+str(i),side='for') for i in range(3)]
    first=node('Primary premise.',replies)
    second=node('Another relevant premise.',[node('Alternative response.',side='for')])
    third=node('A third related premise.',[node('Third response.',side='for')])
    r=HybridRehearsalRetriever(Encoder({}),max_per_anchor=max_per_anchor)
    q=dict(action='attack',target_claim=first.claim)
    return r,[first,second,third],q


def test_unverified_siblings_do_not_fill_all_results():
    r,anchors,q=diversity_fixture()
    _,matches=r.retrieve('motion',q,'for','against',[pool(*anchors)],[],1)
    assert len(matches)==3
    assert len({m[3] for m in matches})==3
    assert r.stats['diversity_skipped']==2


def test_disabling_diversity_reproduces_global_top_materials():
    r,anchors,q=diversity_fixture(None)
    _,matches=r.retrieve('motion',q,'for','against',[pool(*anchors)],[],1)
    assert len(matches)==3 and {m[3] for m in matches}=={anchors[0].claim}


def test_grounded_positive_verdicts_are_not_discarded_for_diversity():
    r,anchors,q=diversity_fixture()
    first=anchors[0]
    cache={verdict_key('motion',q,{'claim':first.claim,'argument':'support','parent_claim':''},
                       {'claim':child.claim,'argument':'support'}):True for child in first.children}
    _,matches=r.retrieve('motion',q,'for','against',[pool(*anchors)],[],1,verdicts=cache)
    assert len(matches)==3 and {m[3] for m in matches}=={first.claim}
    assert r.stats['verified_materials']==3


def test_anchor_allowance_is_shared_across_source_pools():
    r,anchors,q=diversity_fixture()
    duplicate=node(' PRIMARY PREMISE. ',[node('Different reply.',side='for')])
    _,matches=r.retrieve('motion',q,'for','against',[pool(anchors[0])],[pool(duplicate,anchors[1],anchors[2])],1)
    assert len(matches)==3
    assert len({m[3].strip().casefold() for m in matches})==3


@pytest.mark.parametrize('limit',[0,-1,True,1.5])
def test_invalid_anchor_limits_are_rejected(limit):
    with pytest.raises(ValueError,match='max_per_anchor'):
        HybridRehearsalRetriever(Encoder({}),max_per_anchor=limit)
