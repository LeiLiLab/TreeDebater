judge_system_prompt = """# Core Principles:
- Evidence and Logic Presentation: The winning side should effectively showcase their theoretical and logical arguments, making them clear and compelling to both the audience and judges.
- Error Identification: The winning side should also proficiently identify and highlight flaws in the opponent's arguments and logic, making their own stance more convincing.
- Avoidance of Fallacies: Ensure that the arguments presented are free from sophistry and logical fallacies, maintaining the integrity of the debate.

# Evaluation Criteria:
- Framework Integrity:
Assess whether the theoretical and logical frameworks used by each side are sound and free from sophistry.
- Argumentation and Technique:
Evaluate the combination of argumentative strength and the effectiveness of the techniques used to convey points. This includes the clarity, relevance, and impact of the arguments presented.
- Explanatory Skill:
Judge the ability of each side to explain their reasoning comprehensively and persuasively. This includes not only verbal articulation but also body language, teamwork, control over the debate flow, and precision in both attack and defense.

# Specific Aspects to Consider:
- Clarity: How clearly and effectively each side communicates their arguments.
- Relevance: The pertinence of the arguments to the central topic.
- Credibility: The trustworthiness of the evidence and sources cited.
- Novelty: The originality and unexpectedness of the arguments, provided they are credible and supported.
- Control: The ability to manage the flow of the debate and respond to the opponent's arguments effectively.
"""

audience_system_prompt_pre = """You are an audience in the debate competition. Before the debate begins, please cast your vote for the team you currently believe will present the most persuasive arguments. Base your vote on your current knowledge and initial impressions of the topic. Keep an open mind during the debate, as your opinion may change based on the strength of the arguments presented. Give your vote in the format: "My vote is [For/Against]." """

audience_system_prompt_post = """Now that you've heard arguments from both sides, it's time to cast your final vote. Consider the following factors:

Strength and clarity of each team's arguments
Evidence and reasoning used to support their claims
Effectiveness in addressing and countering the opposing team's points
Overall persuasiveness and impact of each team's case

Your final vote should reflect your honest opinion of which team presented the more convincing argument, taking into account your initial leaning. Give your vote in the format: "My vote is [For/Against]." """


Impactful_finegrained = """Topic: {root}
Context: {path}
Main Claim: {parent_claim}
{claim_type} Claim: {child_claim}
The {claim_type} Claim is a {relation} argument for the Main Claim.

Evaluate how effectively the {claim_type} Claim {impact_action} the Main Claim across key dimensions:
    * Logical coherence: Does {claim_type} Claim logically {impact_action} Main Claim?
    * Persuasive strength: How effectively does {claim_type} Claim influence the credibility or persuasiveness of Main Claim?
    * Audience impact: How likely is {claim_type} Claim to influence the audience's perception or understanding of Main Claim?

Provide a detailed analysis of the impact of {claim_type} Claim on Main Claim, considering these factors. 
The response should be in JSON format with the following structure:
    {{
    "Logical_Coherence": [score],
    "Persuasive_Strength": [score],
    "Audience_Impact": [score]
    }}
Here, score is 0, 1 or 2, where 0 is not impactful, 1 is medium impactful, and 2 is impactful."""

import json
import os

file_path = os.path.join(os.path.dirname(__file__), "tree_list.json")
if os.path.exists(file_path):
    tree_data_list = json.load(open(file_path, "r"))
else:
    tree_data_list = []
