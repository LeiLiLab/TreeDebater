import os
import glob

from .opening import (
        propose_definition_prompt,
        claim_propose_prompt,
        main_claim_selection,
        default_opening_prompt, 
        expert_opening_prompt_2,
        )
from .rebuttal import (
        default_rebuttal_prompt, 
        expert_rebuttal_prompt_2
        )
from .closing import (
        default_closing_prompt, 
        expert_closing_prompt_2
        )

from .evaluation import (
        judge_system_prompt,
        audience_system_prompt_pre,
        audience_system_prompt_post,
        Impactful_finegrained,
        tree_data_list
        )


from .others import (
        debater_system_prompt,
        expert_debater_system_prompt,
        extract_statment_prompt,
        extract_statment_by_claim_prompt,
        audience_feedback_prompt, 
        post_process_prompt,
        evidence_selection_prompt,
        select_query_prompt,
        extract_statment_with_tree_prompt,
        rhetorical_techniques_prompt,
        debate_flow_tree_action_prompt,
        debate_flow_tree_action_eval_prompt
        )

from .search import (
    search_prompt,
    iterative_search_prompt,
    summarize_result_prompt,
    extract_author_prompt
)