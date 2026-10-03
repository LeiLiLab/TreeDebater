import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from streaming.planning import IncrementalPlanner, PlanningConfig
from streaming.argument_revisions import revise_claim


def planner(mode, **kw):
    p = IncrementalPlanner(PlanningConfig(mode=mode, **kw))
    p.start("for:opening")
    return p


def callbacks(llm=None):
    return dict(llm=llm or Mock(return_value="Rebut the stated policy only."), analyze=Mock(),
                context=lambda: {"motion": "Limit cars", "our_side": "against"})


def test_linear_is_causal_and_never_builds_a_tree():
    p = planner("linear")
    c = callbacks()
    p.observe("Ban cars.", **c)
    first_prompt = c["llm"].call_args.args[0]
    assert "rush hour" not in first_prompt
    assert "against" in first_prompt
    p.observe("Only downtown during rush hour.", **c)
    assert "rush hour" in c["llm"].call_args.args[0]
    c["analyze"].assert_not_called()
    assert p.plan_version == 2


def test_adaptive_wait_keeps_tail_and_forces_endpoint_reconciliation():
    p = planner("adaptive_tree")
    c = callbacks(Mock(side_effect=["Initial plan", '{"action":"WAIT"}', "Narrowed plan"]))
    p.observe("Ban cars.", **c)
    p.observe("Only downtown.", **c)
    assert p.instructions() == ""  # stale version cannot be used as current
    assert c["analyze"].call_count == 1
    p.finalize("Ban cars. Only downtown.", reset_tree=Mock(), **c)
    assert c["analyze"].call_args.args == ("Only downtown.", True)
    assert "Narrowed plan" in p.instructions()
    assert p.finished


def test_update_cap_does_not_drop_final_qualifier():
    p = planner("linear", max_updates=1)
    c = callbacks()
    p.observe("All cars.", **c)
    p.observe("Correction: only private cars.", **c)
    assert c["llm"].call_count == 1
    p.finalize("All cars. Correction: only private cars.", reset_tree=Mock(), **c)
    assert c["llm"].call_count == 2
    assert p.processed == 2


def test_final_asr_replacement_restores_turn_snapshot_before_replay():
    p = planner("corrected_tree")
    c = callbacks()
    reset = Mock()
    p.observe("Allow all cars.", **c)
    p.finalize("Do not allow all cars.", reset_tree=reset, **c)
    reset.assert_called_once()
    assert c["analyze"].call_args.args == ("Do not allow all cars.", True)
    assert p.chunks == ["Do not allow all cars."]
    p.finalize("Do not allow all cars.", reset_tree=reset, **c)
    assert reset.call_count == 1


def test_malformed_gate_updates_and_failed_plan_never_publishes_stale_state():
    p = planner("adaptive_linear")
    c = callbacks(Mock(side_effect=["Old plan", "not JSON", RuntimeError("timeout")]))
    p.observe("Ban all cars.", **c)
    with pytest.raises(RuntimeError):
        p.observe("Actually only private cars.", **c)
    assert p.instructions() == ""
    assert p.processed == 1


def test_end_of_turn_does_no_speculative_work():
    p = planner("end_of_turn")
    c = callbacks()
    p.observe("First claim.", **c)
    p.observe("Second claim.", **c)
    c["llm"].assert_not_called()
    c["analyze"].assert_not_called()
    p.finalize("First claim. Second claim.", reset_tree=Mock(), **c)
    c["analyze"].assert_called_once_with("First claim. Second claim.", False)


def test_checkpoint_is_independent_and_new_turn_discards_old_notes():
    p = planner("linear")
    c = callbacks()
    p.observe("First claim.", **c)
    saved = copy.deepcopy(p)
    p.observe("Withdraw it.", **c)
    assert saved.version == 1
    p.start("for:rebuttal")
    assert not p.plan and not p.chunks


class Node:
    def __init__(self, claim, side="for", parent=None):
        self.node_id = str(id(self))
        self.position_status = 'current'
        self.claim, self.side, self.parent = claim, side, parent
        self.argument, self.evidence, self.children = ["old support"], ["old evidence"], []
        self.scores = {"support": 1}
        self.status = "attacked"
        if parent: parent.children.append(self)

    def get_node_info(self):
        return {"claim": self.claim, "children": [n.get_node_info() for n in self.children]}

    def update_status(self, value):
        self.status = value

    def add_node(self, *, new_claim, new_argument, side):
        child = Node(new_claim, side, self)
        child.argument, child.evidence, child.scores = list(new_argument), [], None
        return child


def fake_tree():
    root = Node("motion")
    def nodes(n):
        return [n] + [child for c in n.children for child in nodes(c)]
    return SimpleNamespace(root=root, side='for', get_all_nodes=lambda: nodes(root))


def test_revision_and_retraction_retain_original_nodes_and_response_history():
    t = fake_tree()
    claim = Node("Ban all cars", parent=t.root)
    Node("Ambulances need roads", side="against", parent=claim)
    assert revise_claim([t], target="Ban all cars", side="for", action="revise",
                        claim="Limit private cars downtown", arguments=["Local congestion"],
                        source="I mean private cars downtown.") == 1
    assert claim.children and claim.evidence == ['old evidence'] and claim.scores == {'support': 1}
    assert claim.claim == 'Ban all cars' and claim.position_status == 'superseded'
    replacement = t.root.children[-1]
    assert replacement.claim == 'Limit private cars downtown' and replacement.supersedes == claim.node_id
    assert t.revisions[0]["before"]["children"][0]["claim"] == "Ambulances need roads"
    assert revise_claim([t], target=replacement.claim, side="against", action="retract",
                        claim="withdrawn", arguments=[], source="I disagree.") == 0
    assert revise_claim([t], target=replacement.claim, side="for", action="retract",
                        claim="withdrawn", arguments=[], source="I withdraw my proposal.") == 1
    assert t.root.children == [claim, replacement] and replacement.position_status == 'withdrawn'


@pytest.mark.parametrize("settings", [{"mode": "unknown"}, {"max_updates": 0}, {"max_wait_chunks": True},
                                     {"max_tree_targets": True}, {"max_tree_context_nodes": 0}])
def test_invalid_configuration(settings):
    with pytest.raises(ValueError):
        PlanningConfig(**settings)


def test_correction_id_tolerates_paraphrased_target_but_cannot_change_other_speaker():
    t = fake_tree()
    claim = Node("Phone restrictions improve classroom focus", parent=t.root)
    claim.node_id = "claim-1"
    args = dict(target="A classroom concentration improves concentration", target_id="claim-1",
                action="revise", claim="Noneducational phone use is restricted", arguments=[],
                source="Educational use is exempt.")
    assert revise_claim([t], side="against", **args) == 0
    assert revise_claim([t], side="for", **args) == 1
    assert claim.claim == 'Phone restrictions improve classroom focus'
    assert t.root.children[-1].claim == "Noneducational phone use is restricted"
    assert t.root.children[-1].supersedes == claim.node_id
    args["target_id"] = "invented-id"
    assert revise_claim([t], side="for", **args) == 0
