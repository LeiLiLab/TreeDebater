from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ouragents import TreeDebater
from streaming.planning import IncrementalPlanner, PlanningConfig


@pytest.mark.parametrize("mode,grounded", [("linear", False), ("structured_linear", False),
                                           ("grounded_linear", True), ("light_linear", True),
                                           ("grounded_tree", True), ("light_tree", True)])
def test_existing_feedback_and_revision_calls_receive_grounding_without_extra_calls(mode, grounded):
    p = TreeDebater.__new__(TreeDebater)
    p.motion, p.side, p.oppo_side, p.status = "Restrict cars", "against", "for", "rebuttal"
    from debate_tree import DebateTree
    p.debate_tree, p.oppo_debate_tree = DebateTree(p.motion, p.side), DebateTree(p.motion, p.oppo_side)
    p.conversation, p.high_quality_evidence_pool = [], []
    p.add_retrieval_feedback = False
    p.debate_thoughts = []
    p.planner = IncrementalPlanner(PlanningConfig(mode=mode))
    p.planner.start("for:opening")
    p.planner.chunks = ["Only private cars; ambulances are exempt."]
    p.planner.plan = "Current position with an ambulance exception."
    p.planner.version = p.planner.plan_version = 1
    au = SimpleNamespace(feedback=Mock(return_value="Critical Issues and Minimal Revision Suggestions: retain the exception"))
    p.simulated_audience = [au]
    history = [{"side": "for", "stage": "opening", "content": p.planner.chunks[0]}]
    feedback, _ = p._get_feedback_from_audience("Check eligibility.", history)
    p.helper_client = Mock(return_value=["Although ambulances are exempt, eligibility needs a clear appeal process."])
    result = p._length_adjust("Check eligibility.", feedback, [], "", 60, max_retry=1)
    assert "ambulances are exempt" in result
    assert au.feedback.call_count == p.helper_client.call_count == 1
    assert ("GROUNDING CHECK" in au.feedback.call_args.args[0]) == grounded
    assert ("GROUNDING CHECK" in p.helper_client.call_args.kwargs["prompt"]) == grounded
