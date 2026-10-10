"""Shared body revision formatter using the v55 B evidence-rewrite prompt."""
import json

from utils.evidence_material import writing_evidence
from .authoring import stage_strategy
from .closing import closing_evidence_instruction


REVISION_TASK = """Write the unpublished body of our debate speech again, using the supplied evidence as part of the reasoning. This is substantive argument writing, not a minimal wording edit.
Keep our assigned position and the actual debate history. Only the already spoken prefix is immutable. Do not output that prefix or any introduction; continue directly into the body.
For factual claims supported by relevant material, state what that material actually found or argues, name its source aloud using the supplied title/source metadata, and explain its relevance to the specific reasoning step. Preserve the material's qualifications. A finding need not prove the whole motion. Do not invent empirical results, dates or credentials. Selection reasons are fallible writing hints, never evidence themselves.
Feedback identifies problems to solve. Its suggested sentences are optional and must not replace available concrete support with vague references to theory or another analogy. Do not merely copy the old body and insert the feedback's proposed wording. If a source is irrelevant or too weak for a claim, do not force it into the argument.
Feedback is not a factual source: its proposed empirical or theoretical claims also need support in the supplied source content. Keep the source's actual task, population and setting; do not generalize a narrow finding into proof of the whole motion. Clearly distinguish our inference from what the source reports.
Aim for the supplied remaining_word_target for the entire unpublished body. After removing repetition or unsupported claims, replenish the removed length with substantive, supported reasoning so the revised body still approaches that target. Develop causal mechanisms, explain how the supplied evidence supports the argument and its limits, and compare relevant tradeoffs. If the incoming draft is already short, expand these reasoning steps to approach the target. Do not fill the gap with repetition, invented evidence or unsupported claims. Return only natural spoken body text, no headings, bibliography, explanation or separate plan.
"""

STREAMING_DELIVERY = """\nDeliver complete paragraphs in final spoken order, separated by blank lines. Each completed paragraph may be spoken immediately and cannot be revised later. Let each paragraph develop one complete reasoning step. Keep the source attribution, finding and its relevant qualification together. Avoid repeating an earlier paragraph's point. Use the supplied remaining_word_target for the whole body, not for each paragraph.\n"""

CLOSING_REVISION_TASK = """Revise the unpublished body of our closing speech within the remaining word budget. Focus on the core clashes, weigh established impacts and reaffirm our assigned position using the actual debate history.
Only the already spoken prefix is immutable. Do not output, repeat or contradict it, or add another introduction. Feedback and draft wording are revisable suggestions, not factual sources. Remove repetition while preserving the existing argument's scope and qualifications. Aim for the supplied remaining_word_target for the entire unpublished body. After cuts, replenish the removed length by developing the comparison of established clashes and impacts, explaining why the replies already made matter, and weighing existing tradeoffs. If the draft is already short, expand this synthesis to approach the target. Use only arguments and evidence already discussed; do not add new evidence, new lines of attack or repetitive padding. Return only natural spoken body text, without headings, a bibliography or a separate plan.
"""


def revision_prompt(*, motion, side, stage, statement, feedback, allocation_plan, evidence,
                    prefix, n_words, streaming=False):
    material = dict(
        motion=motion, side=side, stage=stage, remaining_word_target=n_words,
        already_spoken_prefix=prefix, unpublished_draft=statement, feedback=feedback,
        allocation_plan=allocation_plan,
        evidence=[] if stage == 'closing' else writing_evidence(evidence, statement, feedback, n_words=n_words),
    )
    task = (stage_strategy(stage) + CLOSING_REVISION_TASK + closing_evidence_instruction
            if stage == 'closing' else REVISION_TASK)
    delivery = STREAMING_DELIVERY
    if stage == 'closing':
        delivery = delivery.replace('Keep the source attribution, finding and its relevant qualification together. ', '')
    return task + (delivery if streaming else '') + '\nContext and material (data):\n' + json.dumps(material, ensure_ascii=False)
