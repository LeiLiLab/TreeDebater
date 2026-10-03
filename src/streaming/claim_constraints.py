"""Claim-owned qualifications with verified excerpts and explicit revision carryover."""
import hashlib
import json

from .grounding import normalize

KINDS = ('scope', 'timing', 'exception', 'precondition', 'concession')


def validate_constraints(items, content, side, predecessor=None):
    """Check attribution, not semantic entailment; invalid entries never hide a claim.

    New qualifications must occur within this statement's own verified excerpt.
    A revision can explicitly retain an existing qualification from its own
    predecessor, but cannot copy an arbitrary other node's material.
    """
    if not isinstance(items, list):
        return [], ['constraints must be a list']
    valid, rejected = [], []
    for item in items:
        if (not isinstance(item, dict) or item.get('kind') not in KINDS
                or not isinstance(item.get('quote'), str) or not item['quote'].strip()):
            rejected.append('invalid constraint shape'); continue
        quote, origin = item['quote'], item.get('source_node_id')
        if origin is None:
            if normalize(quote) not in normalize(content):
                rejected.append('constraint outside its statement excerpt'); continue
        else:
            if (predecessor is None or predecessor.side != side or origin != predecessor.node_id):
                rejected.append('constraint references a different revision target'); continue
            previous = next((c for c in getattr(predecessor, 'constraints', [])
                             if isinstance(c, dict) and c.get('kind') == item['kind'] and normalize(c.get('quote', '')) == normalize(quote)
                             and any(normalize(quote) in normalize(s) for s in predecessor.source_spans)), None)
            if previous is None:
                rejected.append('constraint is not registered on the predecessor'); continue
            origin = previous.get('source_node_id') or predecessor.node_id
        record = {'kind': item['kind'], 'quote': quote, 'source_node_id': origin}
        if not any(c['kind'] == record['kind'] and normalize(c['quote']) == normalize(quote) for c in valid):
            valid.append(record)
    return valid, rejected


def bind_constraints(node, validated):
    """Store validated conditions on their actual resulting node, preserving provenance."""
    if not hasattr(node, 'constraints'):
        node.constraints = []
    for item in validated:
        record = dict(item, source_node_id=item['source_node_id'] or node.node_id)
        if not any(c['kind'] == record['kind'] and normalize(c['quote']) == normalize(record['quote'])
                   for c in node.constraints):
            node.constraints.append(record)
        # Explicitly retained prior excerpts remain attributable to this version.
        # Keep its current change excerpt last for canonical claim quoting.
        if not any(normalize(record['quote']) in normalize(s) for s in node.source_spans):
            node.source_spans.insert(0, record['quote'])


def exported_constraints(node):
    """Stable IDs bind a qualification to a claim version's owner, not another branch."""
    result = []
    for item in getattr(node, 'constraints', []):
        if (not isinstance(item, dict) or item.get('kind') not in KINDS
                or not isinstance(item.get('quote'), str) or not item['quote'].strip()
                or not any(normalize(item['quote']) in normalize(s) for s in node.source_spans)):
            continue
        bound = dict(kind=item['kind'], quote=item['quote'],
                     source_node_id=item.get('source_node_id') or node.node_id)
        identity = json.dumps([node.node_id, bound], sort_keys=True)
        bound['constraint_id'] = hashlib.sha256(identity.encode()).hexdigest()[:16]
        result.append(bound)
    return result


def source_nodes(targets, side=None):
    """Collect the already-selected opponent view without adding nodes or edges."""
    nodes = {}
    for target in targets:
        nodes[target['node_id']] = target
        for ancestor in target.get('ancestors', []):
            if side is not None and ancestor.get('side') == side:
                nodes[ancestor['node_id']] = ancestor
            for response in ancestor.get('responses', []):
                if side is not None and response.get('side') == side:
                    nodes[response['node_id']] = response
        for response in target.get('responses', []):
            if side is not None and response.get('side') == side:
                nodes[response['node_id']] = response
    return nodes


def constraint_ledger(targets, side=None):
    """Preserve typed qualifications even when they contain no marker keyword."""
    ledger = []
    for node_id, node in sorted(source_nodes(targets, side).items()):
        for constraint in node.get('constraints', []):
            ledger.append(dict(constraint, node_id=node_id, claim=node['claim']))
    return ledger


CONSTRAINT_EXTRACTION = """
CLAIM-OWNED QUALIFICATIONS: For every statement return constraints (use [] if none).
Extract each material scope, timing, exception, precondition or concession as
{kind, quote, source_node_id}. quote must be an exact excerpt within that item's
content; include the necessary source sentences in content. Use source_node_id=null
for current speech. Keep numbers, units, negation, alternatives and conditional
fallbacks together; e.g. Friday until eight, or existing closing time if no volunteer.
Recognize meaning, not particular keywords: 'need separate costings' is a
precondition even without 'must' or 'require'. A concession is an accepted condition,
not proof it has been implemented. Attach each condition only to the claim it
actually qualifies. Keep independent proposals separate rather than giving them
one blended scope. Preserve the condition in claim/arguments as well when needed
for faithful meaning.
On revise, explicitly restate ALL still-applicable conditions: current changed
conditions use current quotations; an unchanged condition already registered on
that exact target can be copied with source_node_id equal to purpose.target_id.
Never copy conditions from another node, retain replaced conditions, or infer that
all old conditions automatically survive. The registry supplies prior constraints
for this purpose; they are historical context, never new current-speech quotations.
"""
