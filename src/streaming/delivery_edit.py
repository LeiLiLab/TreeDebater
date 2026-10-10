"""The immutable source and adjacency of one optional spoken length edit.

Source text belongs to this segment. Neighbours explain references and transitions;
they do not authorize moving their propositions into this segment. Changing either
neighbour invalidates a contextual approval, even if the candidate text is identical.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class EditScope:
    source: str
    preceding: tuple[str, ...]
    following: str
    revision: int = 0

    def payload(self):
        return dict(source=self.source, preceding=list(self.preceding),
                    following=self.following, revision=self.revision)
