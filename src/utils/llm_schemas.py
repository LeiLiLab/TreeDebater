from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SchemaBase(BaseModel):
    model_config = ConfigDict(extra="ignore")


class PurposeItem(SchemaBase):
    action: Literal["propose", "rebut", "reinforce", "attack", "revise", "retract"]
    target: str
    target_id: str | None = None
    targeted_debate_tree: Literal["you", "opponent"]


class StatementItem(SchemaBase):
    claim: str
    arguments: list[str] = Field(default_factory=list)
    content: str | None = None
    type: Literal["common", "definition", "criteria"] | None = None
    purpose: list[PurposeItem] | PurposeItem | None = None
    planned_action_ids: list[int] = Field(default_factory=list)


class StatementsResponse(SchemaBase):
    statements: list[StatementItem]


class LinkedPurposeItem(PurposeItem):
    action: Literal["propose", "rebut", "reinforce", "attack", "revise", "retract", "concede"]


class ClaimConstraint(SchemaBase):
    kind: Literal["scope", "timing", "exception", "precondition", "concession"]
    quote: str
    source_node_id: str | None = None


class LinkedStatementItem(StatementItem):
    constraints: list[ClaimConstraint] = Field(default_factory=list)
    purpose: list[LinkedPurposeItem] | LinkedPurposeItem | None = None


class LinkedStatementsResponse(SchemaBase):
    statements: list[LinkedStatementItem]


class BranchChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: int


class BranchMove(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: int
    move: Literal["challenge_support", "challenge_inference", "answer_objection", "concede_then_distinguish"]
    point: str
    assumptions: list[str]


class BranchPlanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claims: list[BranchChoice]
    limits: list[int]
    rebuttals: list[BranchMove]


class SelectionClaimsOnlyResponse(SchemaBase):
    selection: list[str]


class SelectionFramework(SchemaBase):
    claims: list[str]
    framework: str
    explanation: str


class SelectionFrameworkResponse(SchemaBase):
    selection: SelectionFramework


class ActionItem(SchemaBase):
    action: str
    target_claim: str
    target_argument: str | None = None
    prepared_materials: str | None = None
    targeted_debate_tree: Literal["you", "opponent"] | None = None
    idx: int | None = None
    argument: str | None = None
    importance: Literal["high", "medium", "low"] | None = None


class ActionListResponse(SchemaBase):
    response: list[ActionItem]


class BattlefieldEvalItem(SchemaBase):
    battlefield: str
    idx_list: list[int] = Field(default_factory=list)
    supporting_arguments: list[str] = Field(default_factory=list)
    counterarguments: list[str] = Field(default_factory=list)
    unified_argument: str = ""
    importance: Literal["high", "medium", "low"] = "medium"


class BattlefieldResponse(SchemaBase):
    response: list[BattlefieldEvalItem]


class ResultsItem(SchemaBase):
    claim: str
    explanation: str | None = None
    perspective: str | None = None
    concepts: list[str] | None = None
    strength: int | float | None = None


class ResultsResponse(SchemaBase):
    results: list[ResultsItem]


class AuthorItem(SchemaBase):
    id: str | int | None = None
    author: str | None = None
    author_info: str | None = None
    publication: str | None = None


class AuthorsResponse(SchemaBase):
    authors: list[AuthorItem]


class SelectedIdsResponse(SchemaBase):
    selected_ids: list[int | str]


class QueryResponse(SchemaBase):
    query: list[str]
