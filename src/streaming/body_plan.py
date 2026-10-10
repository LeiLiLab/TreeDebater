"""Bounded issue/action plans carried by existing planning and speech calls."""
import copy
import json

ACTIONS = ('develop_case', 'challenge_support', 'challenge_inference',
           'answer_objection', 'concede_then_distinguish', 'weigh')

PLAN_INSTRUCTIONS = (
    'Also return body_plan: an array of at most4 items, each with '
    '{"axis":0,"target":null,"issue":"short issue","action":"answer_objection",'
    '"point":"specific response direction","weight":2}. '
    'axis is an index in overview.response_axes, or null for an additional issue. '
    'target is an index in context.tree_targets, or null for our own case or general weighing. '
    'Cover every promised response axis. Choose actions from develop_case, challenge_support, '
    'challenge_inference, answer_objection, concede_then_distinguish, weigh. '
    'Use relative integer weights1–5 to allocate body words by importance. '
    'Keep issue under8 words and point under12 words. Preserve the previous body plan '
    'when it still fits; update changed targets, missing replies or new qualifications. '
    'These are private tactics, never evidence or opponent commitments. '
    'Use an empty list if the overview is not ready. No additional model call is needed. '
)

DELIVERY_INSTRUCTIONS = (
    'Use body_plan to organize the body by issue, response action and relative importance. '
    'Address every promised overview direction substantively, even if briefly; mentioning '
    'an issue in the overview alone does not cover it. Explain the response mechanism and '
    'compare impacts under the judging criteria. Combine related issues within paragraphs. '
    'Reconcile tactics with the complete latest transcript; correct stale assumptions. '
    'Use response_chain and clash_records.response_chains to answer the latest_opponent_reply with its conditions, '
    'accounting for other replies to the same objection. State what remains disputed; '
    'acknowledging a safeguard does not permit continuing to assume it was absent. '
    'The plan is private guidance, not evidence. A target quote identifies a position, '
    'not proof it is true. Word allocations guide emphasis, not rigid paragraph lengths. '
)


def parse_body_plan(items, framework, targets):
    if not isinstance(items, list) or len(items) > 4:
        raise ValueError('Invalid body plan size')
    axes = framework['response_axes']
    result = []
    for item in items:
        if not isinstance(item, dict) or set(item) != {'axis', 'target', 'issue', 'action', 'point', 'weight'}:
            raise ValueError('Invalid body plan fields')
        axis, target, weight = item['axis'], item['target'], item['weight']
        if axis is not None and (type(axis) is not int or not 0 <= axis < len(axes)):
            raise ValueError('Invalid body plan axis')
        if target is not None and (type(target) is not int or not 0 <= target < len(targets)):
            raise ValueError('Invalid body plan target')
        if type(weight) is not int or not 1 <= weight <= 5:
            raise ValueError('Invalid body plan weight')
        if not isinstance(item['action'], str) or item['action'] not in ACTIONS:
            raise ValueError('Invalid body plan action')
        if any(not isinstance(item[k], str) or not item[k].strip() or len(item[k]) > 300
               for k in ('issue', 'point')):
            raise ValueError('Invalid body plan text')
        source = targets[target] if target is not None else None
        result.append(dict(issue=item['issue'], action=item['action'], point=item['point'], weight=weight,
            covers=[axes[axis]] if axis is not None else [],
            target=(dict(node_id=source['node_id'], version=source['version'],
                         quote=source['sources'][-1]) if source else None)))
    return result


def resolve_body_plan(data, framework):
    """Keep current bindings and make every frozen overview promise explicit."""
    axes = list(dict.fromkeys(framework.get('response_axes', [])))
    targets = {t['node_id']: t for t in data.get('current_targets', [])}
    retained = []
    for item in data.get('body_plan', []):
        target = item['target']
        if target:
            current = targets.get(target['node_id'])
            if (current is None or current['version'] != target['version']
                    or target['quote'] not in current.get('sources', [])):
                continue
        # A plan tied to a replaced overview cannot supply its old tactics.
        if any(axis not in axes for axis in item['covers']):
            continue
        bound = copy.deepcopy(item)
        bound.pop('response_chain', None)
        if target:
            chain = next((chain for record in data.get('clash_records', [])
                          for chain in record.get('response_chains', [])
                          if chain['target_node_id'] == target['node_id']), None)
            if chain is not None:
                bound['response_chain'] = copy.deepcopy(chain)
        retained.append(bound)
    covered = {axis for item in retained for axis in item['covers']}
    missing = [dict(issue=axis, action='answer_objection', weight=1, covers=[axis], target=None,
        point='Address this promised issue using the latest input; explain the mechanism and tradeoff.')
        for axis in axes if axis not in covered]
    # Preserve one entry per promised axis before optional additional issues.
    required, extra, seen = [], [], set()
    for item in retained:
        if set(item['covers']) - seen:
            required.append(item)
            seen.update(item['covers'])
        else:
            extra.append(item)
    result = (required + missing + extra)[:4]
    if not result:
        result = [dict(issue=claim, action='develop_case', weight=1, covers=[], target=None,
            point='Develop the case with a mechanism and impact; answer relevant latest objections.')
            for claim in data.get('our_main_claims', [])[:3]]
    return result


def allocate_words(plan, n_words):
    """Largest-remainder allocation sums to the decoded-audio body budget."""
    if not plan:
        return []
    total = sum(item['weight'] for item in plan)
    words = [n_words * item['weight'] // total for item in plan]
    order = sorted(range(len(plan)), key=lambda i: -(n_words * plan[i]['weight'] % total))
    for i in order[:n_words - sum(words)]:
        words[i] += 1
    return [dict(copy.deepcopy(item), words=count) for item, count in zip(plan, words)]


def allocation_words(allocation, n_words):
    try:
        value = json.loads(allocation)
    except (TypeError, ValueError):
        return []
    return allocate_words(value.get('body_plan', []), n_words) if isinstance(value, dict) else []
