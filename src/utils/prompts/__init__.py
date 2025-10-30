import glob
import os

from .closing import default_closing_prompt, expert_closing_prompt_2
from .evaluation import (
    Impactful_finegrained,
    audience_system_prompt_post,
    audience_system_prompt_pre,
    judge_system_prompt,
    tree_data_list,
)
from .opening import (
    claim_propose_prompt,
    default_opening_prompt,
    expert_opening_prompt_2,
    main_claim_selection,
    propose_definition_prompt,
)
from .others import (
    audience_feedback_prompt,
    debate_flow_tree_action_eval_prompt,
    debate_flow_tree_action_prompt,
    debater_system_prompt,
    evidence_selection_prompt,
    expert_debater_system_prompt,
    extract_statment_by_claim_prompt,
    extract_statment_prompt,
    extract_statment_with_tree_prompt,
    post_process_prompt,
    rhetorical_techniques_prompt,
    select_query_prompt,
)
from .rebuttal import default_rebuttal_prompt, expert_rebuttal_prompt_2
from .search import extract_author_prompt, iterative_search_prompt, search_prompt, summarize_result_prompt
