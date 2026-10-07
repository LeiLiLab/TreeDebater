"""Exercise retrieval control flow offline, without model imports or API calls."""
import ast
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SRC = Path(__file__).resolve().parents[1] / 'src'


def load_function(path, name, scope, cls=None):
    module = ast.parse(path.read_text())
    body = module.body
    if cls:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == cls).body
    function = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


@pytest.fixture
def retrieve():
    function = load_function(SRC / 'utils/helper.py', 'get_retrieval_from_rehearsal_tree',
                             {'__package__': 'utils'})
    def run(*args):
        return function(*args, embed=lambda claims: [[1.0] for _ in claims],
                        validate=lambda cs: material_decisions(
                            cs, args[1], 'challenges' if args[0] in {'attack', 'rebut'} else 'supports'))
    return run


def material_decisions(candidates, target, relation='challenges'):
    return [{'id': c['id'], 'materials': [
        {'id': m['id'], 'relation': relation, 'target_part': 'claim', 'scope': 'compatible',
         'target_quote': target, 'material_quote': m['claim'], 'reason': 'Test relation verdict'}
        for m in c['materials']]} for c in candidates]


def node(claim='target', children=(), side='against'):
    return SimpleNamespace(claim=claim, argument=['support'], children=list(children), side=side,
                           get_strength=lambda **kw: 0.9)


def tree(match):
    return SimpleNamespace(get_node_by_side=lambda side: [match] if match.side == side else [])


@pytest.mark.parametrize('own', [None, []])
def test_missing_own_trees_still_search_opponent(retrieve, own):
    opponent = tree(node(children=[node('response', side='for')]))
    info, matches = retrieve('attack', 'target', 'for', 'against', own, [opponent], 1, [1.0])
    assert len(info) == len(matches) == 1
    assert 'response' in info[0]
    assert matches[0][0] == 'Prepared-Opponent-Tree-Retrieval'


@pytest.mark.parametrize('own', [None, []])
def test_both_pools_empty(retrieve, own, caplog):
    with caplog.at_level(logging.DEBUG):
        assert retrieve('attack', 'target', 'for', 'against', own, [], 1, [1.0]) == ([], [])
    assert caplog.text.count('Miss.') == 2


def test_opponent_summary_does_not_inherit_own_hit(retrieve, caplog):
    with caplog.at_level(logging.DEBUG):
        info, matches = retrieve('propose', 'target', 'for', 'against', [tree(node(side='for'))], [], 1, [1.0])
    assert info and matches
    assert '[Prepared-Opponent-Tree-Retrieval-Summary] propose Miss.' in caplog.text


@pytest.mark.parametrize('action', ['attack', 'rebut'])
def test_leaf_match_does_not_report_usable_response(retrieve, action, caplog):
    with caplog.at_level(logging.DEBUG):
        result = retrieve(action, 'target', 'for', 'against', [tree(node())], [tree(node())], 1, [1.0])
    assert result == ([], [])
    assert caplog.text.count('Miss.') == 2


def test_filtered_nodes_and_scores_stay_aligned():
    nodes = [node(str(i)) for i in range(3)]
    hits = [{'corpus_id': i, 'score': score} for i, score in enumerate([0.95, 0.85, 0.4])]
    search = load_function(SRC / 'debate_tree.py', 'get_most_similar_node', {
        'torch': SimpleNamespace(tensor=lambda x: x),
        'semantic_search': lambda *a, **kw: [hits], 'dot_score': None,
        'logger': logging.getLogger('retrieval-test'),
    }, cls='Tree')
    subject = SimpleNamespace(get_all_nodes=lambda: nodes,
                              get_embedding_from_cache=lambda _: [[1.0]])
    found, scores = search(subject, 'target', top_k=3, threshold=0.8)
    assert found == nodes[:2]
    assert scores == [0.95, 0.85]
