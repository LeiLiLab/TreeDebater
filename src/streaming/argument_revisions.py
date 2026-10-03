"""Speaker-owned amendments retain old nodes and response paths in the tree."""
import copy

from .tree_selection import is_current


def claim_key(text):
    return " ".join(text.split()).casefold().rstrip(".!?")


def revise_claim(trees, *, target, side, action, claim, arguments, source, target_id=None):
    if action not in ("revise", "retract") or not source.strip() or not target.strip():
        return 0
    if target_id:
        identified = [node for tree in trees for node in tree.get_all_nodes()
                      if node.parent is not None and node.side == side
                      and getattr(node, "node_id", None) == target_id]
        if not identified:
            return 0
        target = identified[0].claim
    matches = []
    for tree in trees:
        for node in tree.get_all_nodes():
            if (node.parent is not None and node.side == side
                    and (node.node_id == target_id if target_id else claim_key(node.claim) == claim_key(target))):
                matches.append((tree, node))
    applied = 0
    for tree, node in matches:
        # Repeated extraction of a change must not generate another replacement.
        if getattr(node, "position_status", "current") != "current":
            continue
        applied += 1
        if not hasattr(tree, "revisions"):
            tree.revisions = []
        event = {"action": action, "side": side, "source": source,
                 "before": copy.deepcopy(node.get_node_info()),
                 "replacement": claim if action == "revise" else None,
                 "replacement_id": None}
        tree.revisions.append(event)
        node.change_source = source
        if action == "retract":
            node.position_status = "withdrawn"
        else:
            # A revised statement gets its own identity. Its old replies still
            # refer to the original wording, so never move them to the new node.
            parent = node.parent
            if not is_current(parent):
                # An earlier correction in this batch changed the old context.
                # Preserve the newly sourced claim without inventing a new edge.
                parent = next(t.root for t in trees if t.side == side)
            replacement = parent.add_node(new_claim=claim, new_argument=list(arguments), side=side)
            replacement.relation = getattr(node, "relation", None) if parent is node.parent else 'propose'
            replacement.source_spans = [source]
            replacement.update_order = 1 + max(getattr(n, 'update_order', 0)
                                              for t in trees for n in t.get_all_nodes())
            replacement.supersedes = node.node_id
            replacement.update_status("proposed")
            node.position_status = "superseded"
            node.superseded_by = replacement.node_id
            event['replacement_id'] = replacement.node_id
    return applied


CORRECTION_INSTRUCTIONS = """
Additional allowed purposes: revise and retract. These apply ONLY when the speaker
explicitly corrects, narrows, replaces, or withdraws a claim they previously made.
New exceptions or permissions can narrow an earlier claim even without the word
'correct'. Preserve ALL relevant limits/exemptions in the revised claim/arguments.
Use the EXACT existing claim text as target. The speaker may not revise/retract
the other side's claim: disagreeing with an opponent is attack/rebut, not correction.
For revise, claim is the corrected current claim and arguments contain ONLY support
valid for that corrected claim. For retract, claim describes the withdrawn claim.
content must quote the current speech that establishes the correction verbatim.
Do not infer a correction from silence, topic changes, a low score, or an opponent's
attack. A qualification need not contain the word 'withdraw', but it must actually
change the speaker's prior position. If that meaning is uncertain, preserve the
new statement separately instead of treating the earlier claim as abandoned.
Do not also propose/reinforce the same replacement as a separate purpose.
If the correction's target is unavailable, extract a grounded new claim instead.
"""
