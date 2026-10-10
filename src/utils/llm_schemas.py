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


class OverviewFramework(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ready: bool
    position: str
    core_dispute: str
    response_axes: list[str]
    prefix_action: Literal['wait', 'keep', 'replace']
    reason: str


class BodyPlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    axis: int | None
    target: int | None
    issue: str
    action: Literal['develop_case', 'challenge_support', 'challenge_inference',
                    'answer_objection', 'concede_then_distinguish', 'weigh']
    point: str
    weight: int


class ListeningBranchPlanResponse(BranchPlanResponse):
    overview: OverviewFramework
    body_plan: list[BodyPlanItem] = Field(default_factory=list)


class ListeningSelectionOverview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ready: bool
    core_dispute: str
    response_axes: list[str]
    prefix_action: Literal['wait', 'keep', 'replace']
    reason: str


class ListeningSelectionResponse(BaseModel):
    """The listener selects source material; the speech writer owns arguments."""
    model_config = ConfigDict(extra="forbid")
    claims: list[BranchChoice]
    limits: list[int]
    overview: ListeningSelectionOverview


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
    target_node_id: str | None = None
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
    analysis: dict[str, str] = Field(default_factory=dict)


class QueryResponse(SchemaBase):
    query: list[str]


class RehearsalMaterialDecision(SchemaBase):
    id: int = Field(strict=True)
    relation: Literal["supports", "challenges", "answers", "related", "unrelated", "uncertain"]
    target_part: Literal["claim", "premise", "objection"]
    scope: Literal["compatible", "incompatible", "uncertain"]
    target_quote: str
    material_quote: str
    reason: str


class RehearsalRelationDecision(SchemaBase):
    id: int = Field(strict=True)
    materials: list[RehearsalMaterialDecision]


class RehearsalRelationResponse(SchemaBase):
    decisions: list[RehearsalRelationDecision]
