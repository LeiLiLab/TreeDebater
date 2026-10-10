"""Decode a spoken revision before counting words or removing a fixed prefix."""
import json
import re


def spoken_revision(value):
    """Accept plain text and the observed {speech: text} model response envelope."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Speech revision must be nonempty text')
    text = value.strip()
    # Models sometimes fence a JSON response despite being asked for plain text.
    if text.startswith('```'):
        lines = text.splitlines()
        if lines[0].strip().lower() not in ('```', '```json') or lines[-1].strip() != '```':
            raise ValueError('Invalid speech revision envelope')
        text = '\n'.join(lines[1:-1]).strip()
    if text.startswith(('{', '[')):
        try:
            parsed = json.loads(text)
        except ValueError as exc:
            raise ValueError('Malformed speech revision JSON') from exc
        speech = parsed.get('speech') if isinstance(parsed, dict) else None
        if not isinstance(speech, str) or not speech.strip():
            raise ValueError('Speech revision JSON requires a nonempty speech string')
        return speech.strip()
    if not text:
        raise ValueError('Speech revision must be nonempty text')
    return text


def clean_spoken_revision(value):
    """Apply the native speech-revision cleanup before synthesis or reuse."""
    text = spoken_revision(value).replace("Revised Statement:\n", "")
    text = text.replace("et al.,", "").replace("[X]", "")
    return re.sub(r" [X-Z][ \%]", "", text)
