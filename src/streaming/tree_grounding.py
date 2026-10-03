"""Grounded, active targets from the existing debate trees; no model calls."""
import hashlib
import json

from .grounding import normalize


def attach_source(node, quote, transcript, side):
    """Attach only a speaker-owned, observed excerpt to the actual updated node."""
    if (node is None or node.side != side or not isinstance(quote, str)
            or not quote.strip() or normalize(quote) not in normalize(transcript)):
        return False
    if quote not in node.source_spans:
        node.source_spans.append(quote)
    return True


def tree_targets(trees, opponent_side):
    """Rank unanswered branches and attacks on our claims as response candidates.

    Ranking describes tree topology, not argumentative quality. Archived/detached
    nodes are absent. A version binds claim, supporting material, ancestry and
    direct responses so an old plan cannot silently attach to a changed branch.
    """
    targets = []
    seen = set()
    for tree in trees:
        for node in tree.get_all_nodes():
            if (node.parent is None or node.side != opponent_side or node.node_id in seen
                    or not getattr(node, "source_spans", [])):
                continue
            seen.add(node.node_id)
            ancestry = []
            parent = node.parent
            while parent is not None and parent.parent is not None:
                ancestry.append({"node_id": parent.node_id, "side": parent.side,
                                 "claim": parent.claim, "arguments": list(parent.argument),
                                 "sources": list(getattr(parent, "source_spans", []))})
                parent = parent.parent
            responses = [{"node_id": c.node_id, "claim": c.claim, "side": c.side,
                          "arguments": list(c.argument), "sources": list(c.source_spans),
                          "relation": getattr(c, "relation", None)}
                         for c in node.children]
            item = {"node_id": node.node_id, "claim": node.claim,
                    "arguments": list(node.argument), "sources": list(node.source_spans),
                    "ancestors": ancestry, "responses": responses,
                    "unanswered": not any(c.side != opponent_side for c in node.children),
                    "attacks_our_claim": bool(ancestry and ancestry[0]["side"] != opponent_side)}
            item["version"] = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()[:16]
            targets.append(item)
    targets.sort(key=lambda n: (not n["unanswered"], not n["attacks_our_claim"]))
    return targets
