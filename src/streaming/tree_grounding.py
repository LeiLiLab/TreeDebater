"""Grounded, active targets from the existing debate trees; no model calls."""
import hashlib
import json

from .grounding import normalize
from .tree_selection import DEFAULT_MAX_TARGETS, DEFAULT_MAX_CONTEXT_NODES, is_current, select_nodes


def attach_source(node, quote, transcript, side):
    """Attach only a speaker-owned, observed excerpt to the actual updated node."""
    if (node is None or node.side != side or not isinstance(quote, str)
            or not quote.strip() or normalize(quote) not in normalize(transcript)):
        return False
    if quote not in node.source_spans:
        node.source_spans.append(quote)
    return True


def tree_targets(trees, opponent_side, *, max_targets=DEFAULT_MAX_TARGETS,
                 max_context_nodes=DEFAULT_MAX_CONTEXT_NODES):
    """Rank unanswered branches and attacks on our claims as response candidates.

    Ranking describes tree topology, not argumentative quality. Withdrawn,
    superseded and dependent historical nodes remain stored but are not selected.
    A version binds claim, supporting material, ancestry and
    direct responses so an old plan cannot silently attach to a changed branch.
    """
    def response_info(node):
        return {"node_id": node.node_id, "claim": node.claim, "side": node.side,
                "arguments": list(node.argument), "sources": list(node.source_spans),
                "relation": getattr(node, "relation", None)}
    chosen, context = select_nodes(trees, opponent_side, max_targets=max_targets,
                                   max_context_nodes=max_context_nodes)
    included = {n.node_id for n in chosen + context}
    targets = []
    for node in chosen:
        ancestry = []
        parent = node.parent
        while parent is not None and parent.parent is not None and parent.node_id in included:
            ancestry.append({"node_id": parent.node_id, "side": parent.side,
                             "claim": parent.claim, "arguments": list(parent.argument),
                             "sources": list(getattr(parent, "source_spans", [])),
                             "omitted_response_count": sum(is_current(c) and c.node_id not in included
                                                           for c in parent.children),
                             "responses": [response_info(c) for c in parent.children if c.node_id in included]})
            parent = parent.parent
        responses = [response_info(c) for c in node.children if c.node_id in included]
        item = {"node_id": node.node_id, "claim": node.claim,
                "arguments": list(node.argument), "sources": list(node.source_spans),
                "ancestors": ancestry, "responses": responses,
                "omitted_response_count": sum(is_current(c) and c.node_id not in included for c in node.children),
                "concession": getattr(node, "relation", None) == "concede",
                "unanswered": not any(c.side != opponent_side and is_current(c) for c in node.children),
                "attacks_our_claim": node.parent.parent is not None and node.parent.side != opponent_side}
        item["version"] = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()[:16]
        targets.append(item)
    return targets
