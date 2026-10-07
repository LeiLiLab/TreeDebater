"""Local retrieval must remain usable with every remote entry point disabled."""
from contextlib import nullcontext
import logging
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from utils.local_rehearsal import LocalRehearsalRetriever, remember_verdicts, verdict_key
from test_rehearsal_retrieval import SRC, load_function, node, material_decisions
from test_rehearsal_relations import pool


def fixture():
    reply = node('Labels cannot reliably detect AI content.', side='for')
    anchor = node('Labels protect creators.', [reply])
    return anchor, reply, {'action': 'attack', 'target_claim': anchor.claim}


def run(r, anchor, action, **kw):
    return r.retrieve('motion', action, 'for', 'against', [pool(anchor)], [], 1, **kw)


def test_cold_warm_and_material_mutation():
    a, m, q = fixture(); r = LocalRehearsalRetriever()
    first = run(r,a,q)
    assert m.claim in first[0][0] and r.stats['index_rebuilt']
    assert run(r,a,q) == first and not r.stats['index_rebuilt']
    m.argument = 'new explanation'
    assert 'new explanation' in run(r,a,q)[0][0] and r.stats['index_rebuilt']
    m.position_status = 'superseded'
    assert run(r,a,q) == ([],[])


def test_polarity_rejects_despite_high_cached_similarity():
    a, _, q = fixture(); r = LocalRehearsalRetriever()
    q['target_claim'] = 'Labels harm creators.'
    assert run(r,a,q, embedding_caches=[{a.claim:[1.,0.],q['target_claim']:[1.,0.]}]) == ([],[])


def test_cached_verdict_respects_context_and_explicit_rejection():
    a, m, q = fixture(); r = LocalRehearsalRetriever()
    c = {'id':0,'claim':a.claim,'argument':'support','parent_claim':'',
         'materials':[{'id':0,'claim':m.claim,'argument':'support'}]}
    cache = {}; decisions = material_decisions([c],q['target_claim'],'unrelated')
    remember_verdicts(cache,'motion',q,[c],decisions)
    assert cache[verdict_key('motion',q,c,c['materials'][0])] is False
    assert run(r,a,q,verdicts=cache) == ([],[])
    assert run(r,a,dict(q,target_argument='new premise'),verdicts=cache)[0]
    remember_verdicts(cache,'motion',q,[c],material_decisions([c],q['target_claim']))
    assert run(r,a,q,verdicts=cache)[0] and r.stats['verified_materials'] == 1


def test_bad_vector_dimensions_and_nan_fall_back():
    a,_,q = fixture(); r = LocalRehearsalRetriever()
    assert run(r,a,q,embedding_caches=[{q['target_claim']:[float('nan')]}])[0]
    q['target_claim'] = 'Labels safeguard creators.'
    assert run(r,a,q,embedding_caches=[{q['target_claim']:[1,0],a.claim:[1]}])[0]
    assert r.stats['cached_vector_matches'] == 0


def test_side_and_current_state_filter():
    a,m,q = fixture(); r = LocalRehearsalRetriever()
    m.side = 'against'
    assert run(r,a,q) == ([],[])
    m.side = 'for'; a.position_status = 'superseded'
    assert run(r,a,q) == ([],[])


def test_agent_default_local_has_no_remote_calls_and_reuses_index():
    a,_,q = fixture()
    forbidden = Mock(side_effect=AssertionError('remote call'))
    method = load_function(SRC/'ouragents.py','_retrieve_on_prepared_tree', {
        'get_retrieval_from_rehearsal_tree':forbidden,
        'REMAINING_ROUND_NUM':{'opening_for':3},
        'timed_phase':lambda *a,**k:nullcontext(),'logger':logging.getLogger('test'),
    },cls='TreeDebater')
    p = NS(motion='motion',use_rehearsal_tree=True,prepared_tree_list=None,prepared_oppo_tree_list=None,
           _get_prepared_tree=lambda side:[pool(a)],side='for',oppo_side='against',status='opening',
           _get_embedding_from_cache=forbidden,_validate_rehearsal_candidates=forbidden,
           debate_tree=NS(get_all_nodes=lambda:[],get_embedding_from_cache=forbidden),
           oppo_debate_tree=NS(get_all_nodes=lambda:[]),config=NS(rehearsal_mode="local"),debate_thoughts=[])
    assert method(p,q)
    assert method(p,q)
    forbidden.assert_not_called()
    assert p.debate_thoughts[-1]['local_stats']['index_rebuilt'] is False
    p.config.rehearsal_mode = 'typo'
    with pytest.raises(ValueError,match='rehearsal_mode'):
        method(p,q)
