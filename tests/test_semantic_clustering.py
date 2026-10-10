import copy
import json
from unittest.mock import Mock
import pytest
from utils.semantic_clustering import (semantic_cluster_claims, materialize_groups, validate_partition,
                                      validate_review, apply_local_patch)


def sources(n=2):
    return [{'claim': f'Claim {i}', 'explanation': f'Mechanism {i}', 'strength': 8+i} for i in range(n)]


def group(ids):
    return {'theme':'topic','common_mechanism':'common cause','member_ids':ids,
            'root_claim':'A common claim','root_explanation':'Grounded explanation',
            'coverage':[{'id':i,'connection':'Specific supporting connection'} for i in ids]}


def proposal(ids=None):
    return {'groups':[group([0,1] if ids is None else ids)]}


def issue(**kwargs):
    return dict({'member_ids':[0], 'member_mechanism':'cause0', 'group_mechanism':'cause1',
                 'reason':'Root overstates certainty', 'action':'rewrite_root', 'target_group_index':None}, **kwargs)


def review(issues=None, n=1, single=False):
    return json.dumps({'checks':[{'group_index':i,'member_assessments':[{'id':j,'mechanism':'specific cause','root_connection':'root covers same cause','fits_root':True} for j in ([i] if n>1 or single else [0,1])], 'issues':issues or [] if i==0 else []} for i in range(n)]})


def patch(groups=None, indices=None):
    return json.dumps({'replace_group_indices':[0] if indices is None else indices,
                       'replacement_groups':[group([0,1])] if groups is None else groups})


@pytest.mark.parametrize('ids', [[0], [0,0], [0,2], [0,True], []])
def test_partition_complete_exactly_once(ids):
    with pytest.raises(ValueError):validate_partition(proposal(ids),2,10)


def test_member_coverage_required():
    p=proposal();p['groups'][0]['coverage'].pop()
    with pytest.raises(ValueError):validate_partition(p,2,10)


def test_review_format_retry_does_not_regenerate_or_mutate_partition():
    bad='{"checks":["broken-array",true]}'
    llm=Mock(side_effect=[[json.dumps(proposal())],[bad],[review()]])
    groups,a=semantic_cluster_claims(sources(),llm,'motion','for')
    assert a['accepted'] and len(a['attempts'])==1
    assert [c['phase'] for c in a['calls']]==['proposal','review','review']
    assert bad in llm.call_args_list[2].kwargs['prompt']
    assert groups==a['initial_groups']


def test_semantic_issue_triggers_local_patch_and_fresh_review():
    llm=Mock(side_effect=[[json.dumps(proposal())],[review([issue()])],[patch()],[review()]])
    groups,a=semantic_cluster_claims(sources(),llm,'motion','for')
    assert a['accepted'] and len(a['attempts'])==2
    assert [c['phase'] for c in a['calls']]==['proposal','review','repair','review']
    assert 'Root overstates certainty' in llm.call_args_list[2].kwargs['prompt']


def test_review_format_exhaustion_fails_before_any_repair():
    llm=Mock(side_effect=[[json.dumps(proposal())],['{}'],['{}']])
    audits=[]
    with pytest.raises(ValueError,match='review response'):
        semantic_cluster_claims(sources(),llm,'m','for',max_format_attempts=2,audit_sink=audits.append)
    assert [c['phase'] for c in audits[-1]['calls']]==['proposal','review','review']
    assert audits[-1]['failure_phase']=='review'


def test_patch_format_retry_does_not_regroup_or_rereview():
    llm=Mock(side_effect=[[json.dumps(proposal())],[review([issue()])],['{}'],[patch()],[review()]])
    _,a=semantic_cluster_claims(sources(),llm,'m','for')
    assert [c['phase'] for c in a['calls']]==['proposal','review','repair','repair','review']


def test_round_limit_does_not_silently_accept():
    llm=Mock(side_effect=[[json.dumps(proposal())],[review([issue()])]])
    with pytest.raises(ValueError,match='did not pass review'):
        semantic_cluster_claims(sources(),llm,'m','for',max_attempts=1)
    assert llm.call_count==2


def test_singleton_is_canonical_and_acceptance_is_allowed():
    llm=Mock(side_effect=[[json.dumps(proposal([0]))],[review(single=True)]])
    groups,a=semantic_cluster_claims(sources(1),llm,'m','for')
    assert groups[0]['root_claim']==sources(1)[0]['claim']
    assert groups[0]['root_explanation']==sources(1)[0]['explanation']
    assert a['accepted']


@pytest.mark.parametrize('action',['split','rewrite_root'])
def test_reviewer_cannot_reject_canonical_singleton_for_rewrite_or_split(action):
    with pytest.raises(ValueError,match='singleton'):
        validate_review(json.loads(review([issue(action=action)],single=True)),[group([0])])


def test_move_requires_target_mechanism_and_source_membership():
    groups=[group([0]),group([1])]
    with pytest.raises(ValueError,match='target mechanism'):
        validate_review(json.loads(review([issue(action='move',target_group_index=1)],2)),groups)
    bad=issue(member_ids=[2])
    with pytest.raises(ValueError,match='outside'):
        validate_review(json.loads(review([bad],2)),groups)


def test_valid_singleton_duplicate_can_be_merged_with_specific_target():
    i=issue(action='merge',target_group_index=1,target_mechanism='Same specific cause')
    assert validate_review(json.loads(review([i],2)),[group([0]),group([1])])


def test_review_must_cover_all_groups_and_have_concrete_issues():
    with pytest.raises(ValueError):validate_review(json.loads(review()),[group([0]),group([1])])
    with pytest.raises(ValueError):validate_review(json.loads(review([{'reason':'too broad'}])),[group([0,1])])


def test_local_patch_preserves_untouched_group_and_all_members():
    old=[group([0,1]),group([2])];before=copy.deepcopy(old)
    out=apply_local_patch(old,json.loads(patch()),[dict(issue(),group_index=0)],sources(3),10)
    assert old==before and out[0]['member_ids']==[2]
    # A real input singleton would already be canonical; untouched multi-member example below.
    old=[group([0,1]),group([2,3])]
    out=apply_local_patch(old,json.loads(patch()),[dict(issue(),group_index=0)],sources(4),10)
    assert out[0]==old[1]


@pytest.mark.parametrize('bad',[
    {'replace_group_indices':[0,1],'replacement_groups':[group([0,1,2])]},
    {'replace_group_indices':[0],'replacement_groups':[group([0,2])]},
    {'replace_group_indices':[0],'replacement_groups':[group([0]),group([0,1])]},
])
def test_patch_rejects_unaffected_edits_omissions_and_duplicate_members(bad):
    with pytest.raises(ValueError):
        apply_local_patch([group([0,1]),group([2])],bad,[dict(issue(),group_index=0)],sources(3),10)


def test_split_cannot_exceed_cap():
    with pytest.raises(ValueError):
        apply_local_patch([group([0,1])],json.loads(patch([group([0]),group([1])])),
                          [dict(issue(action='split'),group_index=0)],sources(),1)


def test_synthetic_root_retains_originals_without_mutation():
    original=sources();before=copy.deepcopy(original)
    out=materialize_groups(original,proposal()['groups'])
    assert original==before and out[0][1:]==before
    assert out[0][0]['synthetic_group_root'] and out[0][0]['source_claim_ids']==[0,1]


def test_empty_pool_and_saved_partition_skip_generation():
    llm=Mock(return_value=[review()])
    assert semantic_cluster_claims([],llm,'m','for')[0]==[]
    llm.assert_not_called()
    _,a=semantic_cluster_claims(sources(),llm,'m','for',initial_proposal=proposal())
    assert [c['phase'] for c in a['calls']]==['review']


def test_prepare_evaluates_only_validated_common_root():
    from types import SimpleNamespace as NS
    from test_rehearsal_retrieval import SRC,load_function
    items=[dict(c,perspective='perspective') for c in sources()]
    llm=Mock(side_effect=[['None'],[json.dumps(proposal())],[review()]])
    fn=load_function(SRC/'prepare.py','create_claim',{
        'propose_definition_prompt':Mock(),'claim_propose_prompt':Mock(),'log_llm_io':Mock(),'logger':Mock(),
        'get_response_with_retry':lambda *a,**kw:(items,{}),'ResultsResponse':object,'json':json,
        'semantic_cluster_claims':semantic_cluster_claims,'materialize_groups':materialize_groups},cls='ClaimPool')
    p=NS(pool=[],client=llm,motion='m',side='for',act='support',pool_size=2,max_claim_groups=10,
         clustering_method='semantic',cluster_audit_file=None,minimax_search=Mock(return_value=({},1)))
    result=fn(p,need_evidence=False)
    assert len(result)==1 and p.minimax_search.call_count==1
    assert len(result[0])==3


def test_negative_member_assessment_requires_issue():
    obj=json.loads(review())
    obj['checks'][0]['member_assessments'][0]['fits_root']=False
    with pytest.raises(ValueError,match='requires a concrete issue'):
        validate_review(obj,[group([0,1])])


def test_repair_prompt_contains_only_affected_sources_and_explicit_original_indices():
    claims=sources(3);claims[2]['claim']='UNRELATED_SOURCE_MUST_NOT_APPEAR'
    original=[group([0,1]),group([2])]
    def reviewed(gs, bad=False):
        return json.dumps({'checks':[{'group_index':i,
            'member_assessments':[{'id':j,'mechanism':'cause','root_connection':'fits','fits_root':True} for j in g['member_ids']],
            'issues':[issue()] if bad and i==0 else []} for i,g in enumerate(gs)]})
    llm=Mock(side_effect=[[reviewed(original,True)],[patch()],[reviewed([original[1],original[0]])]])
    _,audit=semantic_cluster_claims(claims,llm,'m','for',initial_proposal={'groups':original})
    repair_prompt=llm.call_args_list[1].kwargs['prompt']
    assert 'UNRELATED_SOURCE_MUST_NOT_APPEAR' not in repair_prompt
    assert 'Allowed member IDs exactly once: [0, 1]' in repair_prompt
    assert 'ORIGINAL index' in repair_prompt and audit['accepted']


def test_partition_reports_missing_duplicate_and_unknown_ids():
    bad={'groups':[group([0,0]),group([3])]}
    with pytest.raises(ValueError) as exc:
        validate_partition(bad,3,10)
    message=str(exc.value)
    assert 'missing=[1, 2]' in message
    assert 'duplicated IDs and group indices={0: [0]}' in message
    assert 'unknown=[3]' in message
    assert 'update coverage' in message


@pytest.mark.parametrize('target',[None,0,7,True])
def test_invalid_move_identifies_issue_and_valid_targets(target):
    groups=[group([0,1]),group([2])]
    obj={'checks':[{'group_index':gi,'member_assessments':[
        {'id':i,'mechanism':'cause','root_connection':'fit','fits_root':True}
        for i in g['member_ids']], 'issues':[issue(action='move',target_group_index=target)] if gi==0 else []}
        for gi,g in enumerate(groups)]}
    with pytest.raises(ValueError) as exc:validate_review(obj,groups)
    msg=str(exc.value)
    assert 'checks[group_index=0].issues[0]' in msg
    assert 'member_ids=[0]' in msg and 'allowed target indices=[1]' in msg
    assert 'action="split"' in msg and 'never invent a target' in msg


def test_combined_action_is_rejected_with_explicit_alternatives():
    with pytest.raises(ValueError) as exc:
        validate_review(json.loads(review([issue(action='split/rewrite_root')])),[group([0,1])])
    assert 'Choose exactly one action' in str(exc.value)
    assert 'split/rewrite_root' in str(exc.value)


def test_missing_id_feedback_reaches_retry_and_is_audited():
    llm=Mock(side_effect=[[json.dumps(proposal([0]))],[json.dumps(proposal())],[review()]])
    _,audit=semantic_cluster_claims(sources(),llm,'m','for')
    prompt=llm.call_args_list[1].kwargs['prompt']
    assert 'missing=[1]' in prompt and 'COMPLETE corrected JSON' in prompt
    assert audit['calls'][0]['retry_feedback'] in prompt
    assert audit['accepted']


def test_null_move_retry_keeps_review_phase_and_requires_real_repair():
    bad=review([issue(action='move',target_group_index=None)])
    llm=Mock(side_effect=[[json.dumps(proposal())],[bad],[review([issue(action='rewrite_root')])],[patch()],[review()]])
    _,audit=semantic_cluster_claims(sources(),llm,'m','for')
    retry=llm.call_args_list[2].kwargs['prompt']
    assert 'allowed target indices=[]' in retry
    assert 'Keep the supplied partition unchanged' in retry
    assert 'Do not suppress semantic issues' in retry
    assert [c['phase'] for c in audit['calls']]==['proposal','review','review','repair','review']


def test_patch_cap_error_accounts_for_untouched_groups():
    with pytest.raises(ValueError) as exc:
        apply_local_patch([group([0,1]),group([2])],json.loads(patch([group([0]),group([1])])),
                          [dict(issue(action='split'),group_index=0)],sources(3),2)
    assert '1 unaffected groups + 2 replacement groups = 3 > 2' in str(exc.value)
    assert 'at most 1 groups' in str(exc.value)
