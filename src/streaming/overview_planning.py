"""A provisional overview carried by the existing incremental planning call."""
import hashlib
import json


OVERVIEW_PLAN = (
    'Also return overview:{ready:boolean,position:string,core_dispute:string,'
    'response_axes:[string],prefix_action:"wait|keep|replace",reason:string}. '
    'This is a provisional speech framework, NOT a complete speech plan. Mark ready once '
    'the heard input establishes the opponent position and a defensible direction for our '
    'overview; detailed rebuttals, evidence, ordering and word allocations may remain incomplete. '
    'Use context.overview_preparation for our upcoming stage and existing overview. '
    'If the framework is unclear return ready:false and prefix_action:wait. '
    'When an overview exists, use keep for ordinary examples, evidence, elaboration or new '
    'independent points that can enter the body. Preserve its position, core_dispute and '
    'response_axes verbatim when they remain accurate. Use replace when newly heard '
    'scope, qualifications, withdrawal or stance changes make that overview inaccurate or '
    'its promised response directions untenable. Also use replace when a newly heard '
    'central opponent point lets a generic overview become a specific response to the '
    'actual clash, especially in rebuttal or closing; update core_dispute or response_axes '
    'to identify that point and explain the substantive change in reason. '
    'Do not replace merely to add another example or mention every opponent argument. '
    'A changed private rebuttal tactic alone does not invalidate a broad overview. '
    'Also preserve a still-accurate previous_framework while waiting for a candidate; '
    'paraphrasing the framework is not progress and must not trigger another draft. '
    'The response_axes need not be exhaustive; leave them empty rather than promise unsettled '
    'directions. Do not invent what the opponent will say. Keep position and core_dispute '
    'under20 words each, axes under8 words each, and reason under12 words. The reason '
    'may simply say "unchanged" when keeping a still-valid framework. '
)


def parse_overview(value):
    keys = {'ready', 'position', 'core_dispute', 'response_axes', 'prefix_action', 'reason'}
    if not isinstance(value, dict):
        raise ValueError('framework must be a JSON object')
    errors = [f'framework.{key} is required' for key in sorted(keys - value.keys())]
    errors += [f'framework.{key} is not an allowed field' for key in sorted(value.keys() - keys)]
    if errors:
        raise ValueError('; '.join(errors))
    if type(value['ready']) is not bool:
        errors.append('framework.ready must be a boolean')
    if value['prefix_action'] not in ('wait', 'keep', 'replace'):
        errors.append('framework.prefix_action must be wait, keep or replace')
    for key in ('position', 'core_dispute', 'reason'):
        if not isinstance(value[key], str) or len(value[key]) > 800:
            errors.append(f'framework.{key} must be a string of at most 800 characters')
    axes = value['response_axes']
    if not isinstance(axes, list) or len(axes) > 3:
        errors.append('framework.response_axes must be an array of at most 3 strings')
    else:
        for index, axis in enumerate(axes):
            if not isinstance(axis, str) or not axis.strip() or len(axis) > 400:
                errors.append(f'framework.response_axes[{index}] must be a nonempty string of at most 400 characters')
    for key in ('reason', 'position', 'core_dispute'):
        if (key == 'reason' or value['ready'] is True) and isinstance(value[key], str) and not value[key].strip():
            errors.append(f'framework.{key} must not be empty')
    if value['ready'] is True and value['prefix_action'] == 'wait':
        errors.append('framework.prefix_action cannot be wait when framework.ready is true')
    if value['ready'] is False and value['prefix_action'] != 'wait':
        errors.append('framework.prefix_action must be wait when framework.ready is false')
    if errors:
        raise ValueError('; '.join(errors))
    return dict(value)


def framework_stamp(framework):
    return hashlib.sha256(json.dumps({k: framework[k] for k in (
        'position', 'core_dispute', 'response_axes')}, sort_keys=True).encode()).hexdigest()
