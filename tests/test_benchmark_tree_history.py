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
