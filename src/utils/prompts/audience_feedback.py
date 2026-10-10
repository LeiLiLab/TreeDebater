"""Full and compact output contracts over the same audience evaluation rules."""
from .others import audience_feedback_rules, audience_feedback_prompt


COMPACT_AUDIENCE_PROMPT = audience_feedback_rules.replace(
    'provide comprehensive feedback', 'provide concise feedback') + (
    '### Output Format\n'
    'Evaluate all four dimensions, but output ONLY the following section:\n'
    '[Critical Issues and Minimal Revision Suggestions]\n'
    'List at most THREE high-priority issues that require a concrete body edit. '
    'For each, identify the affected claim and give the specific minimal change. '
    'Do not output Comprehensive Analysis, separate dimension summaries, praise, '
    'scores, or points that need no change. Keep the entire response at most '
    '180 English words; aim for 120–180 only when needed, and use fewer words '
    'for fewer issues. Do not invent issues or pad the response to meet a minimum. '
    'If no edit is needed, output exactly No changes under the section heading.\n')


def feedback_template(mode):
    if mode not in ('full', 'compact'):
        raise ValueError('Audience feedback mode must be full or compact')
    return COMPACT_AUDIENCE_PROMPT if mode == 'compact' else audience_feedback_prompt
