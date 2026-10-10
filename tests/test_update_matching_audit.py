"""Guard historical audit ownership and ensure it uses pre-extraction snapshots."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('matching_audit', ROOT / 'experiments/tree_update_matching/validate.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_printed_root_and_first_claim_share_speaker_then_alternate():
    text = '''Level-0 Motion: Transport, Side: for
    Level-1 Your Main Claim (Visit: 1, Status: proposed): {"claim": "Keep buses.", "argument": []}
        Level-2 Opponent's Attack (Visit: 1, Status: proposed): {"claim": "Costs rise.", "argument": []}
            Level-3 Your Rebuttal (Visit: 1, Status: proposed): {"claim": "Revenue covers costs.", "argument": []}'''
    tree = audit.parse_tree(text, 'Transport', 'for', 'fixture')
    assert [n.side for n in tree.get_all_nodes()] == ['for', 'for', 'against', 'for']
    assert len({n.node_id for n in tree.get_all_nodes()}) == 4
    assert tree.root.children[0].children[0].claim == 'Costs rise.'


def test_snapshot_ids_are_stable_and_scope_distinct_trees():
    text = 'Level-1 Your Main Claim: {"claim": "Keep buses.", "argument": []}'
    a = audit.parse_tree(text, 'Transport', 'for', 'source:own')
    b = audit.parse_tree(text, 'Transport', 'for', 'source:own')
    c = audit.parse_tree(text, 'Transport', 'for', 'source:other')
    assert a.root.children[0].node_id == b.root.children[0].node_id
    assert a.root.children[0].node_id != c.root.children[0].node_id
