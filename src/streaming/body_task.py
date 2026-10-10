"""Value-only body task shared by early and committed speech work."""
from dataclasses import dataclass
import json

from utils.audience_feedback import review_whole_speech
from utils.prompts.speech_revision import revision_prompt
from utils.speech_context import speech_history
from .full_speech import remaining_text
from utils import speech_length


def no_changes(feedback):
    return feedback.strip().removeprefix('Revision Guidance:').strip().rstrip('.').casefold() == 'no changes'


@dataclass(frozen=True)
class BodyTask:
    # JSON owns a deep, immutable snapshot, including nested history/records.
    context: str
    tail: str
    allocation: str
    # None means preparatory review is unfinished, never an approval verdict.
    prepared_feedback: str | None
    needs_fit: bool

    @classmethod
    def create(cls, *, motion, side, stage, history, prefix, framework, draft,
               preparation=None, clash_records=(), body_plan=(), feedback_context=None):
        preparation = preparation or {}
        allocation = json.dumps(dict(framework=framework, body_plan=list(body_plan),
                                     clash_records=list(clash_records)),
                                ensure_ascii=False, sort_keys=True)
        draft = draft.replace('**Statement:**', '**Statement**').replace('**Statement**:', '**Statement**')
        if '**Statement**' in draft:
            _, draft = draft.split('**Statement**', 1)
        context = json.dumps(dict(motion=motion, side=side, stage=stage,
            history=history, prefix=prefix, clash_records=list(clash_records), body_plan=list(body_plan),
            feedback_context=feedback_context or {'mode': 'compact', 'retrieval': ''}),
            ensure_ascii=False, sort_keys=True)
        return cls(context, remaining_text(draft.strip(), prefix), allocation,
                   preparation.get('feedback', 'No changes'), bool(preparation.get('needs_fit')))

    @property
    def statement(self):
        return json.loads(self.context)['prefix'] + '\n\n' + self.tail

    def same_input(self, other):
        """Derived plans may finish later; complete speech input must still match.

        Callers may keep this immutable task for the current turn when only its
        private plan/record projection changed. Publication still checks the final source version. A changed transcript or draft is never reusable.
        """
        def source(task):
            context = json.loads(task.context)
            context.pop('body_plan')
            context.pop('clash_records')
            return (context, task.tail, task.prepared_feedback, task.needs_fit,
                    json.loads(task.allocation)['framework'])
        return source(self) == source(other)

    def review(self, helper, audiences=None):
        return review_whole_speech(helper, audiences=audiences, statement=self.statement, **json.loads(self.context))

    def needs_revision(self, feedback, estimated_seconds, remaining, *, include_prepared=True, evidence=()):
        return (bool(evidence) or not no_changes(feedback)
                or (include_prepared and (self.needs_fit or self.has_prepared_corrections))
                or not speech_length.duration_fits(estimated_seconds, remaining))

    @property
    def has_prepared_corrections(self):
        return self.prepared_feedback is not None and not no_changes(self.prepared_feedback)

    def guidance(self, feedback, *, include_prepared=True):
        if not feedback.startswith('Revision Guidance:'):
            feedback = 'Revision Guidance:\n' + feedback
        if include_prepared and self.has_prepared_corrections:
            feedback += ('\nUnresolved preparatory corrections: apply each that remains valid '
                'against the final input. Do not preserve a flagged assertion merely to meet '
                'the word target.\n' + self.prepared_feedback)
        return feedback

    def revision_prompt(self, feedback, remaining, evidence=(), *, streaming=False):
        context = json.loads(self.context)
        return revision_prompt(motion=context['motion'], side=context['side'], stage=context['stage'],
            statement=self.tail, feedback=self.guidance(feedback), allocation_plan=self.allocation,
            evidence=evidence,
            prefix=context['prefix'], n_words=speech_length.draft_word_budget(remaining), streaming=streaming)

    def authoring_history(self):
        context = json.loads(self.context)
        return speech_history(context['history'], context['side'])
