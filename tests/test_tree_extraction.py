"""Offline regressions for transcript grounding and duplicate main claims."""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


SRC = Path(__file__).resolve().parents[1] / 'src'


def load_function(path, name, scope, cls=None):
    module = ast.parse(path.read_text())
    body = module.body
    if cls:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == cls).body
    function = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


class ExtractionTests(unittest.TestCase):
    def extract(self, statement, items, plan=None):
        module = ast.parse((SRC / 'utils/prompts/others.py').read_text())
        assignment = next(n for n in module.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'extract_statment_with_tree_prompt'
                                  for t in n.targets))
        scope = {'logger': logging.getLogger('extraction-test'), 'log_llm_io': Mock(),
                 'json': __import__('json'), 'StatementsResponse': object,
                 'get_response_with_retry': Mock(return_value=(items, '{}'))}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), 'prompt', 'exec'), scope)
        extract = load_function(SRC / 'utils/helper.py', 'extract_statement', scope)
        result = extract(None, 'Writing still matters', statement, tree=['', ''], side='for', stage='opening', planned_actions=plan)
        return result, scope['get_response_with_retry'].call_args.args[1]

    def test_rejects_invented_missing_quotes_and_motion_restatements(self):
        items = [
            {'claim': 'Writers communicate effectively', 'content': 'Communication is necessary for success.'},
            {'claim': 'Writers communicate effectively', 'content': None},
            {'claim': 'Writing still matters.', 'content': 'For example, first, a good...'},
        ]
        result, prompt = self.extract('For example, first, a good... The writer must have the creative...', items)
        self.assertEqual(result, [])
        self.assertNotIn('at least three', prompt)
        self.assertIn('There is no minimum number', prompt)

    def test_retains_grounded_claim_and_accepts_empty_extraction(self):
        item = {'claim': 'Writing develops creativity', 'content': 'Writing develops creativity.', 'arguments': []}
        result, _ = self.extract('Writing develops creativity.\nIt is useful.', [item])
        self.assertEqual(result, [item])
        self.assertEqual(self.extract('Hello everyone...', [])[0], [])

    def test_spoken_planned_proposal_keeps_attack_and_excludes_omitted_claim(self):
        plan = [
            {'action': 'propose', 'target_claim': 'Businesses prefer AI because it is cheaper and faster'},
            {'action': 'propose', 'target_claim': 'AI writes in multiple languages'},
        ]
        attack = {'action': 'attack', 'targeted_debate_tree': 'opponent', 'target': 'Human writers are essential'}
        item = {'claim': 'AI reduces business writing costs', 'content': 'AI reduces business writing costs.',
                'arguments': [], 'purpose': [attack], 'planned_action_ids': [0]}
        result, prompt = self.extract(item['content'], [item], plan)
        self.assertIn('Businesses prefer AI', prompt)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['purpose'], [attack, {
            'action': 'propose', 'targeted_debate_tree': 'you', 'target': item['claim'],
        }])
        self.assertNotIn('faster', result[0]['claim'])

    def test_plan_cannot_bypass_quote_grounding_or_use_invalid_ids(self):
        plan = [{'action': 'propose', 'target_claim': 'AI is cheaper'}]
        invented = {'claim': 'AI is cheaper', 'content': 'AI is cheaper.', 'planned_action_ids': [0]}
        self.assertEqual(self.extract('Hello everyone.', [invented], plan)[0], [])
        for ids in ([], [-1, 99, True], [0]):
            item = {'claim': 'Writing develops creativity', 'content': 'Writing develops creativity.',
                    'purpose': [], 'planned_action_ids': ids}
            # No plan means no role can be inferred from IDs alone.
            result, _ = self.extract(item['content'], [item], plan if ids != [0] else None)
            self.assertEqual(result[0]['purpose'], [])

    def test_existing_own_claim_is_reinforced_without_duplicate_proposal(self):
        purpose = {'action': 'reinforce', 'targeted_debate_tree': 'you', 'target': 'AI lowers costs'}
        item = {'claim': 'AI is cheaper', 'content': 'AI is cheaper.',
                'purpose': purpose, 'planned_action_ids': [0]}
        result, _ = self.extract(item['content'], [item], [{'action': 'propose', 'target_claim': 'AI lowers costs'}])
        self.assertEqual(result[0]['purpose'], [purpose])


class PlanHandoffTests(unittest.TestCase):
    def test_plan_contains_only_actions_in_the_rendered_speech_prompt(self):
        selected = {'idx': 0, 'action': 'propose', 'target_claim': 'AI lowers costs', 'prepared_materials': ''}
        omitted = {'idx': 1, 'action': 'propose', 'target_claim': 'AI is multilingual', 'prepared_materials': ''}
        battlefield = {'battlefield_importance': 'high', 'battlefield': 'Cost',
                       'battlefield_argument': 'Efficiency', 'actions': [selected, selected]}
        scope = {'json': __import__('json'), 'get_actions_from_tree': Mock(return_value=[selected, omitted]),
                 'get_battlefields_from_actions': Mock(return_value=[battlefield]),
                 'sort_by_importance': lambda value: 1}
        build = load_function(SRC / 'ouragents.py', '_add_additional_info', scope, cls='TreeDebater')
        player = SimpleNamespace(status='opening', use_debate_flow_tree=True, main_claims_content=[],
                                 debate_tree=Mock(), oppo_debate_tree=Mock(), helper_client=Mock(),
                                 motion='Writing still matters', side='against', _retrieve_on_prepared_tree=lambda action: '')
        plan = []
        prompt = build(player, '{tips}', [], planned_actions=plan)
        self.assertEqual(plan, [{'action': 'propose', 'target_claim': 'AI lowers costs', 'targeted_debate_tree': 'you'}])
        self.assertEqual(prompt.count('*propose*'), 1)
        self.assertNotIn('AI is multilingual', prompt)

    def test_all_speech_stages_pass_only_the_current_plan_with_final_text(self):
        for stage in ('opening', 'rebuttal', 'closing'):
            with self.subTest(stage=stage):
                plan = {'action': 'propose', 'target_claim': 'AI lowers costs'}
                def add_info(prompt, history, planned_actions, **kwargs):
                    self.assertEqual(planned_actions, [])
                    planned_actions.append(plan)
                    return prompt
                player = SimpleNamespace(
                    side='against', motion='Writing still matters', act='OPPOSE', counter_act='SUPPORT',
                    config=SimpleNamespace(streaming_tts=False),
                    use_debate_flow_tree=True, main_claims_content=['AI lowers costs'],
                    debate_thoughts=[{'mode': 'choose_main_claims', 'framework': '', 'explanation': ''}],
                    listen=Mock(), claim_selection=Mock(), debate_tree=Mock(), oppo_debate_tree=Mock(),
                    _add_additional_info=add_info, speak=Mock(return_value='AI lowers costs.'), _analyze_statement=Mock(),
                )
                from utils import speech_length
                scope = {'math': __import__('math'), 'time': __import__('time'), 'speech_length': speech_length,
                         f'expert_{stage}_prompt_2': '{act}'}
                from types import MethodType
                for name in ('_generate_stage', '_prepare_stage_prompt'):
                    method = load_function(SRC / 'ouragents.py', name, scope, cls='TreeDebater')
                    setattr(player, name, MethodType(method, player))
                player._listening_prefix_enabled = lambda: False
                tree_context = load_function(SRC / 'ouragents.py', '_generation_tree_context', scope, cls='TreeDebater')
                player._generation_tree_context = lambda: tree_context(player)
                generate = load_function(SRC / 'ouragents.py', stage + '_generation', scope, cls='TreeDebater')
                generate(player, [], 60)
                player._analyze_statement.assert_called_once_with('AI lowers costs.', 'against', planned_actions=[plan])

    def test_opponent_analysis_does_not_receive_own_plan_and_propose_na_is_resolved(self):
        from contextlib import nullcontext
        from debate_tree import DebateTree
        extract = Mock(return_value=[{'claim': 'Writing develops creativity', 'arguments': [], 'content': 'Writing develops creativity.',
                                     'purpose': {'action': 'propose', 'target': 'N/A', 'targeted_debate_tree': 'you'}}])
        scope = {'logger': logging.getLogger('plan-test'), 'timed_phase': lambda *a, **kw: nullcontext(),
                 'extract_statement': extract}
        analyze = load_function(SRC / 'ouragents.py', '_analyze_statement', scope, cls='TreeDebater')
        player = SimpleNamespace(use_debate_flow_tree=True, side='against', status='opening', motion='Writing still matters',
                                 helper_client=Mock(), debate_tree=DebateTree('Writing still matters', 'against'),
                                 oppo_debate_tree=DebateTree('Writing still matters', 'for'), debate_thoughts=[])
        # A proposal must work after an earlier streaming batch already populated the tree.
        player.oppo_debate_tree.update_node('propose', new_claim='Writing preserves culture', new_argument=[])
        analyze(player, 'Writing develops creativity.', 'for', planned_actions=[{'action': 'propose'}])
        self.assertIsNone(extract.call_args.kwargs['planned_actions'])
        self.assertEqual(player.oppo_debate_tree.root.children[-1].claim, 'Writing develops creativity')
        self.assertFalse(player.debate_tree.root.children)
        self.assertEqual(len(extract.call_args.kwargs['relation_targets']), 1)
        self.assertIn('tree_updates', player.debate_thoughts[-1])


class MainClaimTests(unittest.TestCase):
    def test_duplicate_merges_support_and_preserves_subtree_and_status(self):
        update = load_function(SRC / 'debate_tree.py', 'update_node',
                               {'logger': logging.getLogger('tree-test')}, cls='DebateTree')
        attack = object()
        existing = SimpleNamespace(claim='Writing develops creativity.', argument=['Original reason'],
                                   children=[attack], status='attacked')
        root = SimpleNamespace(children=[existing], side='for', add_node=Mock(), parent=None)
        existing.parent = root
        tree = SimpleNamespace(root=root)
        update(tree, 'propose', new_claim='  WRITING develops  creativity ',
               new_argument=['Original reason', 'New reason'], target='Writing develops creativity')
        root.add_node.assert_not_called()
        self.assertEqual(existing.argument, ['Original reason', 'New reason'])
        self.assertEqual(existing.children, [attack])
        self.assertEqual(existing.status, 'attacked')
        update(tree, 'propose', new_claim='Writing preserves culture', new_argument=[], target='Writing preserves culture')
        root.add_node.assert_called_once()


if __name__ == '__main__':
    unittest.main()


class ActionOwnershipTests(unittest.TestCase):
    def test_ownership_follows_level_and_tree_without_labeling_raw_text_as_support(self):
        import pandas as pd
        child = SimpleNamespace(claim='Counterclaim', argument=['Counterargument'])
        node = SimpleNamespace(claim='Claim', argument=['Support.', 'Opposition.'], children=[child])
        own = SimpleNamespace(max_level=2, get_nodes_by_level=lambda level: [
            SimpleNamespace(claim=f'own-{level}', argument=node.argument, children=node.children)])
        other = SimpleNamespace(max_level=2, get_nodes_by_level=lambda level: [
            SimpleNamespace(claim=f'other-{level}', argument=node.argument, children=node.children)])
        scope = {'pd': pd, 'json': __import__('json'), 'logger': Mock(), 'log_llm_io': Mock()}
        build = load_function(SRC / 'utils/helper.py', 'get_actions_from_tree', scope)
        actions = build([], own, other)
        self.assertEqual([(a['action'], a['claim_owner'], a['desired_direction']) for a in actions], [
            ('reinforce', 'us', 'support'), ('rebut', 'opponent', 'challenge'),
            ('attack', 'opponent', 'challenge'), ('reinforce', 'us', 'support')])
        for action in actions:
            self.assertEqual(action['unclassified_arguments'], ['Support.', 'Opposition.'])
            self.assertEqual(action['counterarguments'][0]['claim'], 'Counterclaim')
            self.assertNotIn('supporting_arguments', action)

    def test_new_claim_belongs_to_us(self):
        import pandas as pd
        scope = {'pd': pd, 'json': __import__('json'), 'logger': Mock(), 'log_llm_io': Mock()}
        build = load_function(SRC / 'utils/helper.py', 'get_actions_from_tree', scope)
        actions = build(['Our new claim'], SimpleNamespace(max_level=0), SimpleNamespace(max_level=0))
        self.assertEqual(actions[0]['claim_owner'], 'us')
        self.assertEqual(actions[0]['desired_direction'], 'support')

    def test_planner_renders_opponent_perspective_and_preserves_separated_arguments(self):
        tree = ast.parse((SRC / 'utils/prompts/others.py').read_text())
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'debate_flow_tree_action_eval_prompt'
                                  for t in n.targets))
        item = {'battlefield': 'Cost', 'idx_list': [0], 'importance': 'high',
                'unified_argument': 'Our funding concern stands.',
                'supporting_arguments': ['Limited funds'], 'counterarguments': ['Congestion benefit']}
        scope = {'json': __import__('json'), 'logger': Mock(), 'log_llm_io': Mock(),
                 'BattlefieldResponse': object, 'sort_by_action': lambda _: 1,
                 'get_response_with_retry': Mock(return_value=([item], '{}'))}
        exec(compile(ast.Module(body=[assignment], type_ignores=[]), 'prompt', 'exec'), scope)
        build = load_function(SRC / 'utils/helper.py', 'get_battlefields_from_actions', scope)
        own, other = Mock(), Mock()
        own.print_tree.return_value = 'Our tree'
        other.print_tree.return_value = 'Opponent tree'
        result = build(None, 'Expand transit', 'against', [], [{'idx': 0, 'action': 'reinforce'}], own, other)
        other.print_tree.assert_called_once_with(include_status=True, reverse=True)
        prompt = scope['get_response_with_retry'].call_args.args[1]
        self.assertIn('Assigned position: OPPOSE the motion', prompt)
        self.assertEqual(result[0]['supporting_arguments'], ['Limited funds'])
        self.assertEqual(result[0]['counterarguments'], ['Congestion benefit'])
