"""Speaker-owned, source-checked tree transactions; no similarity or model calls."""
from .argument_revisions import claim_key, revise_claim
from .tree_grounding import attach_source
from .grounding import normalize
from .tree_selection import is_current, selection_status


def target_registry(trees):
    return [{"node_id": n.node_id, "side": n.side, "claim": n.claim,
             "parent_id": n.parent.node_id if n.parent.parent is not None else None,
             "selection_status": selection_status(n),
             "superseded_by": getattr(n, 'superseded_by', None)}
            for tree in trees for n in tree.get_all_nodes() if n.parent is not None]


def apply_statements(trees, statements, transcript, side):
    """Resolve against the pre-update graph, then apply one correction per target.

    The latest quoted correction wins. At the same source position a replacement
    (revise) supersedes retract, rather than deleting its own target first.
    Missing relations preserve a sourced standalone claim, never mutate another
    speaker's node. An unmatched withdrawal never becomes a current claim.
    """
    own = next(t for t in trees if t.side == side)
    initial = [(t, n) for t in trees for n in t.get_all_nodes() if n.parent is not None]
    update_order = 1 + max((getattr(n, 'update_order', 0) for _, n in initial), default=0)
    events = []

    def resolve(p):
        if p.get('target_id'):
            matches = [(t,n) for t,n in initial if n.node_id == p['target_id']]
        else:
            matches = [(t,n) for t,n in initial if is_current(n)
                       and claim_key(n.claim) == claim_key(p.get('target',''))]
        return matches[0] if len(matches) == 1 else None

    def record(action, **fields):
        events.append(dict(action=action, side=side, **fields))

    def propose(item, reason=None):
        node = own.update_node('propose', new_claim=item['claim'], new_argument=list(item['arguments']), target=item['claim'])
        node.relation = 'propose'
        node.update_order = max(getattr(node, 'update_order', 0), update_order)
        attach_source(node, item['content'], transcript, side)
        if reason:
            record('UNLINKED_CLAIM', node_id=node.node_id, reason=reason)
        return node

    valid = []
    for item in statements:
        quote = item.get('content')
        if (not isinstance(quote,str) or not quote.strip() or normalize(quote) not in normalize(transcript)
                or not item.get('claim','').strip()):
            record('REJECT_SOURCE');continue
        purposes = item.get('purpose') or []
        if isinstance(purposes,dict):purposes=[purposes]
        valid.append((item,purposes))

    corrections = {}
    corrected_items = set()
    for index,(item,purposes) in enumerate(valid):
        for p in purposes:
            if p['action'] not in ('revise','retract'):continue
            corrected_items.add(index)
            target = resolve(p)
            if target is None:
                record('UNMATCHED_CORRECTION', requested=p['action'], target=p.get('target'), target_id=p.get('target_id'))
                # A revise explicitly supplies a current replacement. Keep it as
                # unlinked source material, without claiming an old node matched.
                if p['action']=='revise':propose(item, 'unmatched revision')
                continue
            tree,node=target
            if node.side != side:
                record('REJECT_OWNER', requested=p['action'], node_id=node.node_id);continue
            rank=(normalize(transcript).rfind(normalize(item['content'])), index, p['action']=='revise')
            previous=corrections.get(node.node_id)
            if previous:
                record('COALESCE_CORRECTION', node_id=node.node_id)
            if previous is None or rank > previous[0]:
                corrections[node.node_id]=(rank,tree,node,item,p)

    for _,tree,node,item,p in sorted(corrections.values(),key=lambda x:x[0]):
        if getattr(node, 'position_status', 'current') != 'current':
            record('INACTIVE_CORRECTION', requested=p['action'], node_id=node.node_id)
            if p['action']=='revise':propose(item,'historical correction target')
            continue
        count=revise_claim(trees,target=node.claim,side=side,action=p['action'],claim=item['claim'],
                           arguments=item['arguments'],source=item['content'],target_id=node.node_id)
        record('APPLY_CORRECTION', requested=p['action'],node_id=node.node_id,matches=count)

    for index,(item,purposes) in enumerate(valid):
        if index in corrected_items:continue
        if not purposes:
            propose(item,'missing relation');continue
        for p in purposes:
            action=p['action']
            if action=='propose':propose(item);continue
            target=resolve(p)
            if target is None or not is_current(target[1]):
                propose(item,'missing or historical relation target');continue
            tree,node=target
            if action=='reinforce' and node.side==side:
                for arg in item['arguments']:
                    if arg not in node.argument:node.argument.append(arg)
                attach_source(node,item['content'],transcript,side)
                node.update_order = max(getattr(node, 'update_order', 0), update_order)
                record('REINFORCE',node_id=node.node_id)
            elif action in ('attack','rebut','concede') and node.side!=side:
                # Ownership determines the child speaker; root parity does not.
                child=next((c for c in node.children if is_current(c) and c.side==side
                            and claim_key(c.claim)==claim_key(item['claim'])),None)
                if child is None:
                    child=node.add_node(new_claim=item['claim'],new_argument=list(item['arguments']),side=side)
                else:
                    for arg in item['arguments']:
                        if arg not in child.argument:child.argument.append(arg)
                child.relation={'rebut':'reply','attack':'attack','concede':'concede'}[action]
                child.update_order = max(getattr(child, 'update_order', 0), update_order)
                attach_source(child,item['content'],transcript,side)
                if action!='concede':node.update_status('attacked')
                child.update_status('proposed')
                record('LINK_RESPONSE',node_id=child.node_id,target_id=node.node_id,relation=child.relation)
            else:
                # Do not reinterpret opponent support as an attack or vice versa.
                propose(item,'action conflicts with target speaker')
                record('REJECT_RELATION_OWNER',requested=action,target_id=node.node_id)
    own.update_events=getattr(own,'update_events',[])+events
    return events


RELATION_INSTRUCTIONS = """
SOURCE-OWNED NODE UPDATES: The registry below lists actual node IDs and speakers.
Historical nodes remain in storage. Only selection_status=current nodes may be
targets for new relations or corrections. If a claim is reasserted after withdrawal,
propose it as a new current claim; do not silently reactivate its old response path.
Silence or changing the topic is not withdrawal. When a change in position is
uncertain, preserve the new source as a separate claim instead of retiring the old.
For EVERY attack/rebut/reinforce/revise/retract/concede, copy its node_id into target_id;
choose by meaning, not paraphrased target text or tree-root parity. For propose,
use target_id=null and target=N/A. attack/rebut connects the current speaker's new
claim to a node owned by the OTHER speaker. reinforce/revise/retract targets ONLY
the current speaker's own node. A reply to the other side is never reinforce.
If the speaker narrows/replaces an old claim, emit ONE revise with the replacement
and all still-applicable conditions; never retract and then revise the same ID.
Use retract ONLY for a withdrawal without a replacement. If a relation is unclear,
preserve the current sourced claim as propose, rather than invent a link.
Keep independently qualified claims separate. Preserve timing, exemptions,
conditions and explicitly unanswered implementation questions in claim/arguments.
All content fields must be exact current-speech excerpts. Prior context is for
linking only; do not manufacture new quotations from it.
Additional action concede: when the speaker explicitly accepts the OTHER speaker's
objection or commits to their requested safeguard, link this acceptance to that
other-speaker node with concede and a verbatim current quote. Do not discard an
explicit acceptance as empty/filler, and do not label it an attack or reinforce.
Only the accepted part is conceded; a conditional promise is not fulfilled work.
"""
