"""Select a bounded current view without removing debate history from the trees."""

DEFAULT_MAX_TARGETS = 8
DEFAULT_MAX_CONTEXT_NODES = 16


def selection_status(node):
    """A response to a changed premise needs review, not deletion or a truth label."""
    status = getattr(node, 'position_status', 'current')
    if status != 'current':
        return status
    ancestor = node.parent
    while ancestor is not None:
        if getattr(ancestor, 'position_status', 'current') != 'current':
            return 'needs_review'
        ancestor = ancestor.parent
    return 'current'


def is_current(node):
    return selection_status(node) == 'current'


def select_nodes(trees, side, *, max_targets=DEFAULT_MAX_TARGETS,
                 max_context_nodes=DEFAULT_MAX_CONTEXT_NODES, require_sources=True):
    """Rank current targets, then keep nearby objections, concessions and replies.

    The two caps count distinct nodes. They bound the tree view, not transcript
    length or tokens. Unselected nodes remain in storage and may be used later.
    Silence, a low score or an attack never changes a node's position status.
    """
    for value in (max_targets, max_context_nodes):
        if type(value) is not int or value <= 0:
            raise ValueError('Tree selection limits must be positive integers')
    candidates = []
    seen = set()
    for tree in trees:
        for node in tree.get_all_nodes():
            if (node.parent is None or node.side != side or node.node_id in seen
                    or not is_current(node)
                    or (require_sources and not getattr(node, 'source_spans', []))):
                continue
            seen.add(node.node_id)
            candidates.append(node)

    def priority(node):
        answered = any(c.side != side and is_current(c) for c in node.children)
        responds_to_other = node.parent.parent is not None and node.parent.side != side
        return (getattr(node, 'relation', None) == 'concede', -getattr(node, 'update_order', 0),
                answered, not responds_to_other)

    # Stable ties follow tree traversal; random IDs do not define relevance.
    targets = sorted(candidates, key=priority)[:max_targets]
    selected = {n.node_id for n in targets}
    context = []

    def include(node):
        if (node is not None and node.parent is not None and is_current(node)
                and node.node_id not in selected and len(context) < max_context_nodes):
            selected.add(node.node_id)
            context.append(node)

    # Give each target its immediate question before expanding any long path.
    for node in targets:
        include(node.parent)
    for node in targets:
        if node.parent.node_id in selected:
            for sibling in node.parent.children:
                if getattr(sibling, 'relation', None) == 'concede':
                    include(sibling)
        for child in node.children:
            include(child)
    for node in targets:
        parent = node.parent
        while parent is not None and parent.parent is not None:
            include(parent)
            if parent.node_id not in selected:
                break
            for sibling in parent.children:
                include(sibling)
            parent = parent.parent
    return targets, context
