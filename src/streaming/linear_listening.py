"""Tree-free planning metadata and references to heard speech."""
import hashlib
import json
import re

from utils.speech_context import speech_history
from .overview_planning import OVERVIEW_PLAN, parse_overview


PLAN_CONTRACT = (
    '\nReturn ONLY JSON {"notes":"private rebuttal notes","overview":{...}}. '
    'Keep the notes concise and separate from quoted speech and factual evidence. '
    + OVERVIEW_PLAN.replace('position:string,', '').replace('its position, core_dispute', 'its core_dispute')
    + 'Do not output overview.position; the server supplies our assigned side.\n'
)


def parse_plan(raw, side):
    value = json.loads(raw.strip().removeprefix('```json').removesuffix('```').strip())
    if not isinstance(value, dict) or set(value) != {'notes', 'overview'}:
        raise ValueError('Expected linear notes and overview')
    if not isinstance(value['notes'], str) or not value['notes'].strip():
        raise ValueError('Linear notes must be a nonempty string')
    overview = value['overview']
    if not isinstance(overview, dict) or 'position' in overview:
        raise ValueError('Overview side is supplied by the server')
    return value['notes'].strip(), parse_overview(dict(overview, position=side))


def source_targets(history, side, turn, chunks, max_targets, *, transcript=None):
    """Index verbatim batches, without inferring claims, edges or conditions.

    Corrected ASR replaces the current turn's batches. IDs identify a span's
    location; versions identify its text, invalidating bindings after correction.
    Full role-labelled speech history remains available even beyond this bound.
    """
    opponent = 'against' if side == 'for' else 'for'
    owner, _, stage = (turn or '').partition(':')
    current_stage = stage if owner == opponent else None
    heard = ' '.join(chunks)
    transcript = heard if transcript is None else transcript
    messages = speech_history(history, side, opponent_statement=transcript,
                              opponent_stage=current_stage)
    targets = []
    for message in messages:
        if message['role'] != 'user':
            continue
        header = re.match(r"\*\*Opponent's (\w+) Statement\*\*\n", message['content'])
        speech_stage = header.group(1)
        text = message['content'][header.end():]
        spans = (chunks if speech_stage == current_stage and chunks
                 and ' '.join(text.split()) == ' '.join(heard.split()) else [text])
        for index, span in enumerate(spans):
            if not span.strip():
                continue
            targets.append(dict(node_id=f'heard-{opponent}-{speech_stage}-{index}',
                version=hashlib.sha256(span.encode()).hexdigest(), side=opponent,
                claim=span, arguments=[span], sources=[span], constraints=[],
                source_kind='transcript_span'))
    return targets[-max_targets:]
