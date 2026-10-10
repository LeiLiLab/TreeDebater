"""Resolve update targets by identity or unique text, never semantic proximity."""
import unicodedata

from .tree_selection import is_current


def target_key(text):
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFC', text).translate(str.maketrans({'’': "'", '‘': "'", '“': '"', '”': '"'}))
    return ' '.join(text.split()).casefold().rstrip('.!?。！？').strip()


def resolve_target(nodes, purpose, *, expected_side=None, preferred_tree=None):
    """Return (tree/node pair or None, auditable resolution metadata).

    An explicit ID is authoritative, including an invalid ID. Text recovery
    preserves negation, quantities and qualifiers. Repeated text in independent
    branches is ambiguous unless the supplied tree uniquely disambiguates it.
    Inactive/wrong-owner unique matches are returned for the caller to reject;
    callers MUST check ``reason`` before mutating them.
    """
    nodes = [(t, n) for t, n in nodes if n.parent is not None]
    identity = purpose.get('target_id')
    key = target_key(purpose.get('target'))
    method = 'id' if identity else 'normalized_text'
    if identity:
        matches = [(t, n) for t, n in nodes if n.node_id == identity]
    elif key and key != 'n/a':
        matches = [(t, n) for t, n in nodes if target_key(n.claim) == key]
        eligible = [(t, n) for t, n in matches if is_current(n) and (expected_side is None or n.side == expected_side)]
        if eligible:
            matches = eligible
        if len(matches) > 1 and preferred_tree is not None:
            preferred = [(t, n) for t, n in matches if t is preferred_tree]
            if len(preferred) == 1:
                matches = preferred
                method = 'normalized_text_in_tree'
    else:
        matches = []
    info = dict(method=method, target=purpose.get('target'), requested_target_id=identity,
                candidate_ids=[n.node_id for _, n in matches])
    if not matches:
        info['reason'] = 'unknown_id' if identity else ('missing_target' if not key or key == 'n/a' else 'text_not_found')
        return None, info
    if len(matches) != 1:
        info['reason'] = 'ambiguous_target'
        return None, info
    tree, node = matches[0]
    info['target_id'] = node.node_id
    info['reason'] = ('owner_mismatch' if expected_side is not None and node.side != expected_side else
                      'historical_target' if not is_current(node) else None)
    return (tree, node), info
