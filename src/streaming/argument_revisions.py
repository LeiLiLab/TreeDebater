"""Explicit speaker-owned argument amendments with archived dependent subtrees."""
import copy


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
        # A parent's revision can already have detached a matched descendant.
        if node not in tree.get_all_nodes():
            continue
        applied += 1
        if not hasattr(tree, "revisions"):
            tree.revisions = []
        tree.revisions.append({"action": action, "side": side, "source": source,
                               "before": copy.deepcopy(node.get_node_info()),
                               "replacement": claim if action == "revise" else None})
        if action == "retract":
            node.parent.children.remove(node)
            if node.parent.parent is not None and not node.parent.children:
                node.parent.update_status("proposed")
        else:
            node.claim = claim
            node.argument = list(arguments)
            node.evidence = []
            if hasattr(node, "source_spans"):
                node.source_spans = [source]
            # Old attacks/replies depend on the old premise. Archive them above;
            # require revalidation before reusing them against the narrower claim.
            node.children = []
            node.scores = None
            node.update_status("proposed")
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
Do not also propose/reinforce the same replacement as a separate purpose.
If the correction's target is unavailable, extract a grounded new claim instead.
"""
