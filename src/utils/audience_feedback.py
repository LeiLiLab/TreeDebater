"""Native Audience reviews on frozen inputs, with the spoken prefix immutable."""
import json

from utils.prompts.audience_feedback import COMPACT_AUDIENCE_PROMPT, feedback_template
from utils.speech_context import SpeechRejected as SegmentRejected


def review_whole_speech(helper, *, motion, side, stage, statement, history, prefix,
                       clash_records=(), body_plan=(), audiences=None, provisional=False,
                       feedback_context=None):
    if stage == 'closing':
        return 'No changes', []  # Match TreeDebater's original closing-feedback policy.
    from agents import Audience, AudienceConfig

    panel = tuple(audiences) if audiences is not None else (Audience(AudienceConfig()),)
    if not panel:
        raise SegmentRejected('Audience panel is empty')
    opponent = 'against' if side == 'for' else 'for'
    history_text = ''
    for entry in history:
        owner = entry.get('side') or (side if entry.get('role') == 'assistant' else opponent)
        label = f'Opponent ({opponent})' if owner == opponent else f'You ({side})'
        history_text += (f"*{label}'s {entry.get('stage', 'previous').title()} Statement*\t"
                         + entry['content'].replace('\n', ' ') + '\n\n')
    marker = 'LISTENING BODY FEEDBACK:' if provisional else 'LISTENING WHOLE SPEECH FEEDBACK:'
    feedback_context = feedback_context or {}
    prompt = marker + '\n' + feedback_template(feedback_context.get('mode', 'compact')).format(
        motion=motion, side=side, stage=stage.title(), statement=statement,
        history=history_text, retrieval=feedback_context.get('retrieval', ''))
    if prefix:
        prompt += ('\nThe opening below is fixed for audio. Review the complete supplied speech, '
                   'but direct edits to the remaining body only. Never rewrite, repeat or contradict '
                   'the opening. If no edit is needed, write exactly No changes in the '
                   'Critical Issues and Minimal Revision Suggestions section.\n')
    if provisional:
        prompt += ('This is a provisional draft based only on the speech heard so far. '
                   'Do not assume future input. Feedback is for revision, not publication approval.\n')
    # Only supplemental data: full history and statement already appear above.
    prompt += json.dumps(dict(motion=motion, side=side, stage=stage, fixed_prefix=prefix,
                              body_plan=list(body_plan)), ensure_ascii=False)
    details, corrections = [], []
    for audience in panel:
        feedback = audience.feedback(prompt, isolated=True, completion=helper)
        if not isinstance(feedback, str) or not feedback.strip():
            raise SegmentRejected('Audience returned empty feedback')
        details.append(feedback)
        heading = 'Critical Issues and Minimal Revision Suggestions'
        key = feedback.split(heading, 1)[-1].strip().lstrip(']:*# \n').strip()
        if key.rstrip('.').casefold() != 'no changes':
            corrections.append(f'Audience {len(details)} Feedback:\n{heading}\n{key}')
    return ('\n\n'.join(corrections) if corrections else 'No changes'), details
