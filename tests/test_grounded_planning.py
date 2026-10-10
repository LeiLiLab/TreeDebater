import json
from unittest.mock import Mock

import pytest

from streaming.grounding import parse_state
from streaming.planning import IncrementalPlanner, PlanningConfig


def snapshot(quote="Restrict cars."):
    return {"claims": [{"text": "Restrict cars", "quote": quote}], "limits": [],
            "rebuttals": [{"target": 0, "point": "Ask how access will be preserved", "assumptions": []}]}


def policy(mode="flat_tree", **kw):
    p = IncrementalPlanner(PlanningConfig(mode=mode, **kw))
    p.start("for:opening")
    cb = dict(llm=Mock(return_value='{"claims": [], "limits": [], "rebuttals": []}'), analyze=Mock(),
              context=lambda: {"tree_targets": [], "constraints": [], "position_limits": [],
                               "correction_history": [], "use_topology": False})
    return p, cb


@pytest.mark.parametrize("damage", ["quote", "target", "assumptions", "kind"])
def test_unattributed_or_mislinked_state_is_rejected(damage):
    data = snapshot()
    if damage == "quote": data["claims"][0]["quote"] = "Ban ambulances."
    if damage == "target": data["rebuttals"][0]["target"] = 1
    if damage == "assumptions": data["rebuttals"][0]["assumptions"] = "proven"
    if damage == "kind": data["limits"] = [{"kind": "opinion", "quote": "Restrict cars."}]
    with pytest.raises(ValueError):
        parse_state(json.dumps(data), "Restrict cars.")


def test_invalid_state_never_publishes_untrusted_model_notes_and_keeps_source():
    p, cb = policy()
    cb["llm"].return_value = '{"made_up": "Ban ambulances."}'
    p.observe("Restrict cars.", **cb)
    assert p.state == {}
    assert "Ban ambulances" not in p.instructions()
    assert "Restrict cars." in p.instructions()
    assert p.events[-2]["action"] == "INVALID_STATE"
    assert p.plan_version == p.version  # safe raw-input fallback, not a stale plan


def test_observed_phase_in_label_preserves_attributed_limit_without_accepting_fabrication():
    data = snapshot()
    data["limits"] = [{"kind": "phase-in", "quote": "phase this in over two years"}]
    result = parse_state(json.dumps(data), "Restrict cars. We phase this in over two years.")
    assert result["limits"] == [{"kind": "scope", "quote": "phase this in over two years"}]
    with pytest.raises(ValueError, match="Source quote"):
        parse_state(json.dumps(data), "Restrict cars. Phase this in immediately.")


def test_budget_wait_and_identical_pending_input_are_still_drained():
    p, cb = policy(max_updates=1)
    p.observe("Restrict cars.", **cb)
    p.observe("Except ambulances.", **cb)
    p.observe("Except ambulances.", **cb)
    assert p.processed == 1
    p.finalize("Restrict cars. Except ambulances. Except ambulances.", reset_tree=Mock(), **cb)
    assert p.processed == 3 and cb["llm"].call_count == 2
