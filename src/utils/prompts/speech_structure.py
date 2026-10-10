"""Shared speech structure for full statements and listening-time preparation."""

OVERVIEW = (
    'Write a natural first paragraph suited to this stage and the actual exchange. '
    'Choose its purpose from what the speech needs: clarify our definition or scope, '
    'propose a judging principle, state our own substantive point, pose a focused '
    'question, draw a relevant contrast, or respond directly to something already heard. '
    'These are equally available approaches, not a checklist or a fixed rotation. '
    'A brief greeting or acknowledgment may accompany any approach; it need not appear '
    'in every speech and should lead directly into substance in the same paragraph. '
    'Keep the whole paragraph within the existing word budget. Make our stance clear '
    'through its substance; no separate declaration or roadmap is required. '
    'Do not default to an opponent paraphrase followed by a promise to demonstrate '
    'our case. The paragraph can make its point directly, without naming the opponent '
    'or announcing what we will say later. A brief roadmap is optional when useful. '
    'Use supplied previous speeches from both sides to avoid repeating their opening '
    'purpose and sentence construction merely with different nouns. Choose a different '
    'natural entry point when it serves this speech; do not invent facts or arguments '
    'for variety. You may directly explain one concrete idea in this first paragraph; '
    'the following paragraphs continue its reasoning or develop the remaining points. '
)
OVERVIEW_STAGES = {
    'opening': 'For opening, establish the case using the elements it needs. If a definition '
               'needs clarification, start there; if not, a judging principle, concrete '
               'problem, own claim or response may be more useful. FOR can explain its '
               'own definition; AGAINST can accept the scope and proceed, or challenge an '
               'actual problematic definition. Do not require an opponent reference. ',
    'rebuttal': 'For rebuttal, advance the live disagreement. An own claim, focused question, '
                'conceptual clarification, comparison or direct answer may open it. '
                'Responding does not require first restating the opponent. If you choose '
                'an attribution, respect the supplied sources and their argumentative role. ',
    'closing': 'For closing, make a final assessment of the established case. An impact '
               'comparison, deciding question, established judging principle or verdict '
               'may lead. Do not introduce a new definition, argument or evidence, or '
               'promise what we will argue later. ',
}
POINTS = (
    'Develop distinct points in a clear order, using verbal signposts such as first, second '
    'and finally. Each point should have a purpose, reasoning and relevant support. '
    'Conclude by weighing the central issue and reaffirming our position. '
)
STAGES = {
    'opening': 'Plan definitions and judging criteria where needed, then main claims and '
               'their support in order of importance. Introduce the case without attributing '
               'unspoken positions to the opponent. ',
    'rebuttal': 'Open naturally before developing the individual battlefields. For each point, '
                'introduce and explain the actual opponent argument, respond, then explain '
                'the impact for our side. ',
    'closing': 'Frame the core conflicts, compare established impacts against the judging '
               'criteria, and finish with a persuasive conclusion and grounded value appeal. '
               'Do not introduce new arguments or evidence. ',
}


def speech_structure(stage):
    return 'SPEECH STRUCTURE: ' + OVERVIEW + OVERVIEW_STAGES[stage] + STAGES[stage] + POINTS


def body_structure(stage):
    """Continuation contract: never ask a body writer to open the speech again."""
    stage_rule = STAGES[stage].replace('Open naturally before developing the individual battlefields. ', '')
    return ('SPEECH STRUCTURE: The overview has already been spoken; do not add a second overview. '
            + stage_rule + POINTS)
