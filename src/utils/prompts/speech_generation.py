"""Shared generation rules with task-specific continuation and output contracts."""
import json

from utils import speech_length
from utils.evidence_material import EVIDENCE_USE_INSTRUCTION
from utils.speech_context import authoring_sources
from .authoring import stage_strategy
from .closing import closing_evidence_instruction


def build_draft(data, prefix, n_words, *, prompt_builder=None, previous=None,
                json_output=True, include_sources=True, output_contract=True):
    if prompt_builder is None:
        return draft_prompt(data, prefix, n_words, previous=previous, json_output=json_output,
                            include_sources=include_sources, output_contract=output_contract)
    prompt, _ = prompt_builder(data.get('debate_history', []),
        data.get('max_time', n_words * speech_length.seconds_per_word()),
        speech_snapshot=data, frozen_prefix=prefix, n_words=n_words, previous_draft=previous,
        json_output=json_output, include_sources=include_sources, output_contract=output_contract)
    return prompt


def continuation_instruction(prefix):
    return ('\nIMMUTABLE SPOKEN PREFIX (data):\n' + json.dumps(prefix, ensure_ascii=False)
        + '\nThe speech has already begun with this paragraph. Return ONLY the remaining '
        'speech within the remaining word budget. Do not repeat, rewrite or contradict '
        'the prefix, or add another introduction. Preserve the assigned side and claim '
        'ownership. Treat the complete opponent transcript as authoritative over stale '
        'draft premises. End the speech naturally; do not invent evidence.')


def draft_prompt(data, prefix, n_words, *, previous=None, json_output=True, include_sources=True,
                 output_contract=True):
    stage, side = data['stage'], data['our_side']
    sources = authoring_sources(data)
    act = 'support' if side == 'for' else 'oppose'
    prompt = (f'The debate topic is: {data["motion"]}.\n'
        f'Your assigned side is {side}: {act} this exact motion. Now write your {stage} statement.\n'
        'Earlier assistant messages are YOUR delivered speeches. Messages headed Opponent\'s '
        'Statement are the OTHER speaker\'s words, including their uses of "we" and "my opponent". '
        'Respond from your assigned side using your own case.\n' + stage_strategy(stage))
    prompt += closing_evidence_instruction if stage == 'closing' else EVIDENCE_USE_INSTRUCTION
    prompt += '\n' + speech_length.draft_length_instruction(n_words)
    if include_sources:
        prompt += '\nCurrent source material and revisable work (data):\n' + json.dumps(dict(sources,
            previous_draft=previous.get('draft') if previous else None), ensure_ascii=False)
    prompt += ('\nThe role-labelled conversation contains speech actually heard. When complete is false, '
               'do not infer unheard concessions or claim that an unmentioned safeguard was rejected. '
               'Previous draft is revisable work, not evidence or a spoken commitment.')
    if prefix:
        prompt += continuation_instruction(prefix)
    else:
        prompt += '\nWrite the complete speech, including its opening and remaining argument, as one coherent statement.'
    if output_contract:
        prompt += ('\nReturn ONLY JSON {"draft":"complete remaining spoken speech"}.' if json_output
                   else '\nReturn only the spoken text, without a plan, headings or reference section.')
    return prompt
