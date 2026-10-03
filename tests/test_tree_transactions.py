import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from debate_tree import DebateTree
from streaming.argument_revisions import revise_claim
from streaming.tree_updates import apply_statements
from streaming.tree_grounding import attach_source, tree_targets
from streaming.branch_planning import planning_material, parse_branch_state, branch_prompt


def setup_trees():
    return DebateTree('Transport','for'), DebateTree('Transport','against')


def add(tree, claim):
    node=tree.update_node('propose',new_claim=claim,new_argument=[],target=claim)
    attach_source(node,claim,claim,tree.side)
    return node


def item(quote, action, node=None, **kwargs):
    purpose={'action':action,'target':node.claim if node else 'N/A',
             'target_id':node.node_id if node else None,'targeted_debate_tree':'opponent'}
    return dict(claim=quote,content=quote,arguments=[quote],purpose=[purpose],**kwargs)


def test_retract_then_revise_same_item_preserves_replacement_and_archives_old_dependencies():
    own,other=setup_trees();old=add(own,'Backup for all homes.')
    old.add_node(new_claim='Capacity is insufficient.',new_argument=[],side='against')
    replacement=item('Backup only for the clinic.', 'retract',old)
    replacement['purpose'].append(dict(replacement['purpose'][0],action='revise'))
    events=apply_statements((own,other),[replacement],replacement['content'],'for')
    assert old in own.get_all_nodes() and old.claim==replacement['claim'] and not old.children
    assert old.source_spans==[replacement['content']]
    assert own.revisions[0]['before']['children']
    assert sum(e['action']=='APPLY_CORRECTION' for e in events)==1


def test_later_actual_withdrawal_wins_over_earlier_replacement():
    own,other=setup_trees();old=add(own,'Run a pilot.')
    earlier=item('Only a one-month pilot.', 'revise',old)
    later=item('I withdraw even the pilot.', 'retract',old)
    apply_statements((own,other),[earlier,later],earlier['content']+' '+later['content'],'for')
    assert old not in own.get_all_nodes()


def test_id_correction_does_not_rewrite_same_text_on_an_independent_branch():
    own,other=setup_trees();a=add(own,'Cost matters.')
    parent=add(other,'Build it.');b=parent.add_node(new_claim=a.claim,new_argument=[],side='for')
    assert revise_claim((own,other),target=a.claim,side='for',action='revise',claim='Total cost matters.',
                        arguments=[],source='Total cost matters.',target_id=a.node_id)==1
    assert b.claim=='Cost matters.'


def test_reply_to_opponent_root_claim_creates_speaker_owned_child_not_reinforcement():
    own,other=setup_trees();target=add(other,'No accessible vehicle is provided.')
    statement=item('Every run includes an accessible vehicle.', 'rebut',target)
    apply_statements((own,other),[statement],statement['content'],'for')
    assert target.argument==[] and target.source_spans==[target.claim]
    child=target.children[0]
    assert child.side=='for' and child.relation=='reply' and child.source_spans==[statement['content']]
    assert tree_targets((own,other),'for')[0]['attacks_our_claim']


def test_unknown_link_preserves_statement_but_never_claims_a_relation():
    own,other=setup_trees();statement=item('Only a one-term trial.', 'rebut')
    statement['purpose'][0]['target_id']='missing'
    events=apply_statements((own,other),[statement],statement['content'],'for')
    assert own.root.children[0].claim==statement['claim'] and not other.root.children
    assert any(e['action']=='UNLINKED_CLAIM' for e in events)


def test_wrong_owner_correction_and_unattributed_content_cannot_mutate_graph():
    own,other=setup_trees();target=add(other,'Keep buses.')
    bad=item('Remove buses.', 'revise',target)
    events=apply_statements((own,other),[bad],bad['content'],'for')
    assert target.claim=='Keep buses.' and not own.root.children
    assert events[0]['action']=='REJECT_OWNER'
    apply_statements((own,other),[item('Invented.', 'propose')],'Actually heard.','for')
    assert not own.root.children


def test_same_side_attack_does_not_create_false_opponent_node():
    own,other=setup_trees();target=add(own,'Keep buses.')
    statement=item('Improve buses.', 'attack',target)
    apply_statements((own,other),[statement],statement['content'],'for')
    assert not target.children and len(own.root.children)==2
    assert all(n.side=='for' for n in own.root.children)


def test_tree_transaction_metadata_survives_roundtrip():
    own,other=setup_trees();target=add(other,'No access.')
    statement=item('Provide access.', 'rebut',target)
    apply_statements((own,other),[statement],statement['content'],'for')
    assert DebateTree.from_json(own.get_tree_info()).get_tree_info()==own.get_tree_info()
    restored=DebateTree.from_json(other.get_tree_info())
    assert restored.root.children[0].children[0].relation=='reply'


def materials():
    own,other=setup_trees();proposal=add(own,'Run a one-month trial only.')
    objection=proposal.add_node(new_claim='Wheelchair access is missing.',new_argument=[],side='against')
    attach_source(objection,objection.claim,objection.claim,'against')
    reply=objection.add_node(new_claim='Every run must have an accessible vehicle.',new_argument=[],side='for')
    attach_source(reply,reply.claim,reply.claim,'for')
    targets=tree_targets((own,other),'for')
    return (own,other),reply,targets


def test_branch_briefs_link_prior_objection_reply_and_limits_while_flat_control_omits_edges():
    trees,reply,targets=materials()
    rich=planning_material(targets,trees,'for',topology=True)
    flat=planning_material(targets,trees,'for',topology=False)
    brief=next(b for b in rich['branch_briefs'] if b['node_id']==reply.node_id)
    assert brief['latest_our_objection']['claim']=='Wheelchair access is missing.'
    assert rich['position_limits']==flat['position_limits']
    assert 'branch_briefs' not in flat
    assert all(set(n)=={'node_id','claim','arguments','sources','version'} for n in flat['tree_targets'])
    assert 'Use branch_briefs' in branch_prompt(rich,[],{})
    assert 'Use branch_briefs' not in branch_prompt(flat,[],{})


def test_indexed_choices_are_bound_server_side_and_keep_unselected_constraint_coverage():
    trees,reply,targets=materials();material=planning_material(targets,trees,'for',topology=True)
    index=next(i for i,n in enumerate(targets) if n['node_id']==reply.node_id)
    raw=json.dumps({'claims':[{'target':index}],'limits':[],
                    'rebuttals':[{'target':0,'move':'concede_then_distinguish','point':'Ask about missed pickups.','assumptions':[]}]})
    state=parse_branch_state(raw,'',material)
    assert state['claims'][0]['node_id']==reply.node_id
    assert state['position_limits'] and state['branch_briefs'][0]['latest_our_objection']
    assert state['rebuttals'][0]['target_node_id']==reply.node_id


@pytest.mark.parametrize('index',[True,-1,999,'0'])
def test_indexed_choices_reject_invalid_or_fabricated_targets(index):
    trees,reply,targets=materials();material=planning_material(targets,trees,'for',topology=True)
    with pytest.raises(ValueError):
        parse_branch_state(json.dumps({'claims':[{'target':index}],'limits':[],'rebuttals':[]}),'',material)


def test_empty_target_set_allows_source_context_without_fabricating_a_claim():
    own,other=setup_trees();material=planning_material([], (own,other),'for',topology=True)
    state=parse_branch_state('{"claims":[],"limits":[],"rebuttals":[]}', 'Hello.',material)
    assert state['claims']==[] and state['rebuttals']==[]


@pytest.mark.parametrize('mode',['branch_tree','flat_tree'])
def test_indexed_plan_is_used_by_generation_without_rendered_tree_leakage(mode):
    from ouragents import TreeDebater
    from streaming.planning import IncrementalPlanner, PlanningConfig
    trees,reply,targets=materials()
    p=TreeDebater.__new__(TreeDebater)
    p.motion='Transport';p.side='against';p.oppo_side='for';p.act='oppose';p.counter_act='support';p.status='rebuttal'
    p.oppo_debate_tree,p.debate_tree=trees
    p.use_debate_flow_tree=True;p.conversation=[];p.high_quality_evidence_pool=[]
    p.planner=IncrementalPlanner(PlanningConfig(mode=mode));p.planner.start('for:rebuttal')
    p.planner.chunks=[reply.claim];p.planner.version=p.planner.plan_version=1
    material=p._planning_context();idx=next(i for i,n in enumerate(material['tree_targets']) if n['node_id']==reply.node_id)
    p.planner.state=parse_branch_state(json.dumps({'claims':[{'target':idx}],'limits':[],'rebuttals':[]}),reply.claim,material)
    p.planner.plan=json.dumps(p.planner.state)
    p.debate_tree.print_tree=Mock(return_value='UNWANTED_RENDERED_TREE')
    p.oppo_debate_tree.print_tree=Mock(return_value='UNWANTED_RENDERED_TREE')
    p.listen=Mock();p.speak=Mock(return_value='Delivered.');p._analyze_statement=Mock()
    p.rebuttal_generation([],60)
    prompt=p.speak.call_args.args[0]
    assert 'UNWANTED_RENDERED_TREE' not in prompt
    assert ('"branch_briefs"' in prompt)==(mode=='branch_tree')
    assert 'position_limits' in prompt and 'BRANCH DELIVERY' in prompt
