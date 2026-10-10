"""Small exchange records derived from retained, source-bearing debate branches.

No model calls or second mutable ledger: replay, corrections and checkpoints use
the existing trees as the source of truth. Edges are extracted relationships,
not proof that an objection was answered adequately.
"""
from .tree_selection import selection_status
from .claim_constraints import exported_constraints


RECORD_GUIDANCE = (
    'Use clash_records to continue each main exchange: acknowledge the latest reply '
    'before explaining the remaining disagreement. Entries contain observed excerpts, '
    'not private draft arguments. A reply edge does not establish that its reasoning '
    'succeeds; awaiting_response means no current reply is recorded, not that the '
    'opponent said nothing. Assess the remaining gap against the full latest speech. '
    'Withdrawn/superseded entries are history, not current commitments. Records and '
    'excerpts are bounded; the complete latest transcript overrides them, including '
    'qualifications outside an excerpt. Do not invent a concession or a resolution. '
    'response_chains preserve the selected target, our connected objection and the latest '
    'opponent reply on that path, with full source excerpts and claim-owned conditions. '
    'Acknowledge that reply before explaining the remaining gap; do not keep an old '
    'premise merely after mentioning its replacement. Other replies to the same objection '
    'may supply safeguards. The remaining question is a task, not a recorded concession. '
)


def _brief(text, limit=180):
    return text if len(text) <= limit else text[:limit] + '…'


def exchange_records(trees, our_side, *, max_entries=4, target_ids=(), max_chains=4):
    """One stable record per main branch, including earlier versions of its root."""
    if max_entries < 2:
        raise ValueError('Exchange records need at least two entries')
    trees = tuple(trees)
    nodes = {n.node_id: n for tree in trees for n in tree.get_all_nodes()
             if n.parent is not None}
    original_orders = {}
    for tree in trees:
        for event in getattr(tree, 'revisions', []):
            before = event.get('before', {})
            if before.get('node_id') in nodes:
                original_orders.setdefault(before['node_id'], before.get('update_order', 0))

    def root_id(node):
        while node.parent is not None and node.parent.parent is not None:
            node = node.parent
        seen = set()
        while getattr(node, 'supersedes', None) in nodes and node.node_id not in seen:
            seen.add(node.node_id)
            old = nodes[node.supersedes]
            # An independently reasserted subclaim must not merge whole branches.
            if old.parent.parent is not None:
                break
            node = old
        return node.node_id

    groups = {}
    for node in nodes.values():
        if node.source_spans:
            groups.setdefault(root_id(node), []).append(node)
    def current(node):
        return bool(node.source_spans) and selection_status(node) == 'current'

    def source(node):
        if node is None:
            return None
        return dict(node_id=node.node_id, side=node.side, relation=getattr(node, 'relation', None),
                    responds_to=node.parent.node_id if node.parent.parent is not None else None,
                    status=selection_status(node), quote=node.source_spans[-1],
                    constraints=exported_constraints(node))

    # Bound the number of connected paths, never cut a source qualification in half.
    candidates = [nodes[i] for i in dict.fromkeys(target_ids) if i in nodes
                  and nodes[i].side != our_side and current(nodes[i])]
    if not target_ids:
        candidates = sorted((n for n in nodes.values() if n.side != our_side and current(n)),
                            key=lambda n: -n.update_order)
    chains = {}
    for target in candidates[:max_chains]:
        descendants = []
        def visit(node):
            if not current(node):
                return
            if node.side != our_side:
                descendants.append(node)
            for child in node.children:
                visit(child)
        visit(target)
        reply = max(descendants, key=lambda n: n.update_order)
        ancestor = reply.parent
        while ancestor.parent is not None and ancestor.side != our_side:
            ancestor = ancestor.parent
        prior = ancestor if ancestor.parent is not None and current(ancestor) else None
        objection = prior if prior and getattr(prior, 'relation', None) in ('attack', 'reply') else None
        siblings = sorted((n for n in objection.children if n.side != our_side
                           and n.node_id != reply.node_id and current(n)),
                          key=lambda n: -n.update_order) if objection else []
        own_responses = sorted((n for n in reply.children if n.side == our_side and current(n)),
                               key=lambda n: -n.update_order)
        chains.setdefault(root_id(target), []).append(dict(target_node_id=target.node_id,
            target=source(target), our_objection=source(objection), our_prior_position=source(prior),
            latest_opponent_reply=source(reply),
            other_replies=[source(n) for n in siblings[:2]], omitted_other_replies=max(0, len(siblings)-2),
            our_latest_response=source(own_responses[0]) if own_responses else None,
            remaining_question='After accounting for the latest reply and its conditions, what specific inferential or comparative gap remains?'))
    records = []
    for key, members in groups.items():
        ordered = sorted(members, key=lambda n: original_orders.get(n.node_id, n.update_order))
        roots = [n for n in members if n.parent.parent is None]
        current_roots = [n for n in roots if selection_status(n) == 'current']
        latest_root = (current_roots or roots or ordered)[-1]
        selected = ordered if len(ordered) <= max_entries else [ordered[0], *ordered[-(max_entries - 1):]]
        entries = []
        for node in selected:
            quote = node.source_spans[-1]
            entries.append(dict(node_id=node.node_id, side=node.side,
                relation=getattr(node, 'relation', None),
                responds_to=node.parent.node_id if node.parent.parent is not None else None,
                status=selection_status(node), excerpt=_brief(quote),
                excerpt_truncated=len(quote) > 180))
        pending = [n for n in ordered if selection_status(n) == 'current'
                   and getattr(n, 'relation', None) != 'concede'
                   and not any(c.side != n.side and c.source_spans
                               and selection_status(c) == 'current' for c in n.children)]
        changes = sorted((n for n in ordered if getattr(n, 'change_source', None)),
                         key=lambda n: n.update_order)
        records.append(dict(clash_id=key, topic=_brief(latest_root.claim, 100),
            our_side=our_side, status=selection_status(latest_root), entries=entries,
            omitted_entries=max(0, len(ordered) - len(selected)),
            latest_change=(_brief(changes[-1].change_source) if changes else None),
            awaiting_response=[dict(node_id=n.node_id, from_side='against' if n.side == 'for' else 'for',
                                    excerpt=_brief(n.source_spans[-1])) for n in pending[-2:]],
            omitted_pending=max(0, len(pending) - 2),
            last_update=max(n.update_order for n in members),
            response_chains=chains.get(key, [])))
    return records


def prompt_records(records, limit=3, *, target_ids=()):
    """Bound prompt size; all main branches remain available in the retained tree."""
    priorities = {node_id: i for i, node_id in enumerate(target_ids)}
    def rank(record):
        priority = min((priorities.get(c['target_node_id'], len(priorities))
                        for c in record.get('response_chains', [])), default=len(priorities))
        return record['status'] != 'current', priority, -record['last_update']
    return sorted(records, key=rank)[:limit]
