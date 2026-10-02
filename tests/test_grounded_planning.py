import copy
import json
from unittest.mock import Mock

import pytest

from streaming.grounding import parse_state, needs_semantic_gate
from streaming.planning import IncrementalPlanner, PlanningConfig


def snapshot(quote="Restrict cars."):
    return {"claims": [{"text": "Restrict cars", "quote": quote}], "limits": [],
            "rebuttals": [{"target": 0, "point": "Ask how access will be preserved", "assumptions": []}]}


def policy(mode="light_linear", **kw):
    p = IncrementalPlanner(PlanningConfig(mode=mode, **kw))
    p.start("for:opening")
    cb = dict(llm=Mock(return_value=json.dumps(snapshot())), analyze=Mock(), context=lambda: {})
    return p, cb


def test_structured_state_links_sources_without_leaking_future_input():
    p, cb = policy("structured_linear")
    p.observe("Restrict cars.", **cb)
    assert "Ambulances" not in cb["llm"].call_args.args[0]
    assert p.state == snapshot()
    assert "UNVERIFIED" in p.instructions()
    cb["analyze"].assert_not_called()
    assert not p.grounding_instructions()


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


def test_exact_adjacent_repetition_skips_model_and_endpoint_work():
    p, cb = policy()
    p.observe("Restrict cars.", **cb)
    p.observe("Restrict   cars.", **cb)
    assert cb["llm"].call_count == 1
    assert p.events[-1]["action"] == "SKIP_DUPLICATE"
    assert p.processed == 2 and p.plan_version == 2
    p.finalize("Restrict cars. Restrict cars.", reset_tree=Mock(), **cb)
    assert cb["llm"].call_count == 1


def test_old_claim_repeated_after_a_correction_is_not_a_duplicate():
    p, cb = policy()
    for text in ("Restrict cars.", "Actually allow cars.", "Restrict cars."):
        p.observe(text, **cb)
    assert cb["llm"].call_count == 3
    assert all(e["action"] != "SKIP_DUPLICATE" for e in p.events)


def test_split_exception_waits_without_guessing_and_drains_at_endpoint():
    p, cb = policy()
    p.observe("Restrict cars except", **cb)
    cb["llm"].assert_not_called()
    assert p.instructions() == ""
    cb["llm"].return_value = json.dumps(snapshot("Restrict cars except ambulances."))
    p.observe("ambulances.", **cb)
    assert cb["llm"].call_count == 1
    assert p.processed == 2
    p.observe("Unless", **cb)
    assert p.instructions() == ""
    p.finalize("Restrict cars except ambulances. Unless", reset_tree=Mock(), **cb)
    assert p.processed == 3 and cb["llm"].call_count == 2


def test_budget_wait_and_identical_pending_input_are_still_drained():
    p, cb = policy(max_updates=1)
    p.observe("Restrict cars.", **cb)
    p.observe("Except ambulances.", **cb)
    p.observe("Except ambulances.", **cb)
    assert p.processed == 1
    p.finalize("Restrict cars. Except ambulances. Except ambulances.", reset_tree=Mock(), **cb)
    assert p.processed == 3 and cb["llm"].call_count == 2


def test_model_gate_only_for_near_repetition_and_wait_remains_pending():
    p, cb = policy()
    p.observe("Restrict cars.", **cb)
    cb["llm"].side_effect = ['{"action":"WAIT"}', json.dumps(snapshot())]
    p.observe("Restrict cars now.", **cb)
    assert p.processed == 1 and p.instructions() == ""
    p.finalize("Restrict cars. Restrict cars now.", reset_tree=Mock(), **cb)
    assert p.processed == 2 and cb["llm"].call_count == 3


def test_changes_in_numbers_negation_and_new_claims_bypass_gate():
    assert not needs_semantic_gate("The fee is 10 dollars.", ["The fee is 20 dollars."])
    assert not needs_semantic_gate("Restrict all private cars.", ["Do not restrict all private cars."])
    assert not needs_semantic_gate("Restrict cars.", ["Public transport needs additional capacity."])


def test_grounding_tracks_replacement_and_checkpoint_without_old_claims():
    p, cb = policy("grounded_linear")
    p.observe("Restrict cars.", **cb)
    saved = copy.deepcopy(p)
    cb["llm"].return_value = json.dumps(snapshot("Allow ambulances."))
    reset = Mock()
    p.finalize("Allow ambulances.", reset_tree=reset, **cb)
    reset.assert_called_once()
    assert "Allow ambulances." in p.grounding_instructions()
    assert "Restrict cars." not in p.grounding_instructions()
    assert saved.state["claims"][0]["quote"] == "Restrict cars."
    p.start("for:rebuttal")
    assert p.state == {} and p.grounding_instructions() == ""
