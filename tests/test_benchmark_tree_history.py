import json
from types import SimpleNamespace
from unittest.mock import Mock

from scripts.benchmark_incremental_planning import judge, make_player


def case():
    return dict(motion='Transport', side='against', own_claim='Access matters', own_opening='Preserve access.',
                chunks=['Only private cars.'], checks=['Acknowledges scope'], prior_history=[
                    {'side': 'for', 'stage': 'opening', 'content': 'Limit all cars.'},
                    {'side': 'against', 'stage': 'rebuttal', 'content': 'Allow ambulances.'}])


def test_prior_history_is_shared_but_tree_analysis_is_not_hidden(monkeypatch):
    players = []
    def factory(config, motion):
        p = SimpleNamespace(side=config.side, use_debate_flow_tree=False, _add_message=Mock(),
                            _analyze_statement=Mock(), simulated_audience=[],
                            planner=SimpleNamespace(config=SimpleNamespace(corrections=config.planning['mode']=='grounded_tree')))
        players.append(p)
        return p
    monkeypatch.setattr('ouragents.TreeDebater', factory)
    for mode in ('grounded_linear', 'grounded_tree'):
        make_player(case(), mode, Mock())
    assert players[0]._add_message.call_args_list == players[1]._add_message.call_args_list
    assert len(players[0]._add_message.call_args_list) == 3
    assert players[1]._analyze_statement.call_args_list[0].kwargs['allow_corrections'] is True
    assert players[0]._analyze_statement.call_args_list[0].kwargs['allow_corrections'] is False


def test_judge_receives_prior_context_and_uniform_explicit_output_cap():
    client = Mock()
    client.complete.return_value = json.dumps(dict(checks=[{'passed': True, 'reason': 'Only private cars'}],
                                                   relevance=5, rebuttal_strength=3, strawman=False, unsupported_facts=False))
    judge(case(), 'Only private cars are covered.', client, 'gpt-5.6-sol', max_tokens=1600)
    args, kwargs = client.complete.call_args
    assert kwargs['model'] == 'gpt-5.6-sol' and kwargs['max_tokens'] == 1600
    assert 'Allow ambulances.' in args[0][0]['content']


def test_saved_pre_generation_tree_is_not_mutated_by_own_speech_analysis(monkeypatch):
    from agents import Debater
    from debate_tree import DebateTree
    from scripts.benchmark_incremental_planning import run_case
    ours, theirs = DebateTree('Transport', 'against'), DebateTree('Transport', 'for')
    node = theirs.update_node('propose', new_claim='Limit cars', new_argument=['Initial reason'], target='Limit cars')
    planner = SimpleNamespace(plan='', state={'limits': []}, events=[])
    player = SimpleNamespace(side='against', oppo_side='for', debate_tree=ours,
                             oppo_debate_tree=theirs, planner=planner, observe_opponent=Mock())
    monkeypatch.setattr(Debater, 'post_process', lambda *a, **kw: 'Answer')
    def generate(*args, **kwargs):
        result = Debater.post_process(player)
        node.argument.append('Added by subsequent own-speech analysis')
        planner.state['limits'].append('Later state')
        return result
    player.rebuttal_generation = generate
    monkeypatch.setattr('scripts.benchmark_incremental_planning.make_player', lambda *a: player)
    client = Mock(label='offline')
    client.summary.return_value = dict.fromkeys(('calls', 'input_tokens', 'output_tokens',
                                                'reported_usage_estimate_usd', 'reserved_upper_usd'), 0)
    sample = case();sample.update(id='offline', kind='regression')
    result = run_case(sample, 'grounded_tree', 0, client)
    assert result['before_generation']['opponent_tree']['structure']['children'][0]['argument'] == ['Initial reason']
    assert result['before_generation']['state'] == {'limits': []}
    assert len(result['after_generation']['opponent_tree']['structure']['children'][0]['argument']) == 2


def test_evaluation_tree_limits_reach_real_planning_configuration(monkeypatch):
    from streaming.planning import PlanningConfig
    configs = []
    def factory(config, motion):
        configs.append(PlanningConfig(**config.planning))
        return SimpleNamespace(side=config.side, use_debate_flow_tree=False, _add_message=Mock(),
                               _analyze_statement=Mock(), simulated_audience=[],
                               planner=SimpleNamespace(config=configs[-1]))
    monkeypatch.setattr('ouragents.TreeDebater', factory)
    make_player(case(), 'branch_tree', Mock())
    make_player(case(), 'branch_tree', Mock(), {'max_tree_targets': 128, 'max_tree_context_nodes': 256})
    assert (configs[0].max_tree_targets, configs[0].max_tree_context_nodes) == (8, 16)
    assert (configs[1].max_tree_targets, configs[1].max_tree_context_nodes) == (128, 256)
