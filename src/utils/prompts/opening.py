context = (
    "The debate topic  is: {motion}. \n"
    "You side is to {act} this topic .\n"
    "Now it comes the opening phase. A complete opening statement should include definitions, judging criteria, and arguments. \n"
)

default_opening_prompt = context + (
    "Please give an opening statement using three claims with {{n_words}} words, do not output other things.\n"
)

expert_opening_prompt = context + (
    "\n## Rules\n"
    "- Ensure your language is fluent and natural, indistinguishable from human writing. Make sure your debate script is complete, including definitions, judging criteria, and arguments.\n"
    "- When citing data and academic research, provide sources within the context and avoid using information not present in the provided materials. Ensure your arguments are supported by data and academic evidence.\n"
    "- When citing data, **using specific figures** instead of just descriptive language will make your argument more persuasive.\n"
    "- When citing data and academic research, **don't just** list the information, **but also explain** how it supports your point.\n"
    "- When choosing evidence, numeric data is preferred over qualitative descriptions.\n"
    "- List your references in a standard format in a Reference section, making sure each source has a clear number and url information.\n"
    "- Renumber the original citations to [1],[2], ... and etc. List full references in **Reference** section. This section should start with **Reference**. Use Chicago style for the reference list and DO NOT include the web link or url information.\n\n"
)

expert_opening_prompt_2 = context + (
    "\n\n## Workflow \n"
    "1. Create a opening plan based on the current debate flow trees. It should include the definition, judging criteria, and the battlefields to discuss. \n"
    "- If the position is to support the topic, discuss the given definition with necessary details and address uncertainties. "
    "- If the position is to oppose the topic, discuss the opponent's EXISTING definition in their debate flow tree if unclear or problematic. Make only necessary clarifications. \n"
    "- If the definition concerns a specific subset of a broader category, clearly distinguish it by highlighting unique characteristics, mechanisms, or impacts. "
    "2. Select among the definition, judging criteria, and battlefields and allocate your word budget based on the importance of each point. You should allocate more words to the more important points and can ignore the less important points. \n"
    "\t- If the definition is selected to discuss, discuss it first. \n"
    "\t- If the judging criteria is selected to discuss, discuss it next. \n"
    "\t- Among the selected battlefields, discuss the battlefield in the order of importance. \n"
    "3. Follow the opening plan to generate conversational arguments. Write as you would speak, with VARIED sentence lengths. Use short and simple words and sentences that are easy to understand.\n"
    "4. Deliver a {{n_words}}-word opening statement. Present only the final text in clear, flowing prose without bullet points, asterisks, or numbered lists. \n"
    "Note that it's possible that the debate flow tree is not provided, in this case, you can just generate a opening statement without following the debate flow tree.\n"
    "## Debate Flow Tree Structure\n"
    "You are given two debate trees that model the back-and-forth between you and your opponent. Each node contains:\n"
    "* Data: The specific claims and arguments\n"
    "* Visit Count: Number of times addressed in debate\n"
    "* Status: 'proposed' (new), 'attacked' (challenged), or 'solved' (resolved)\n\n"
    "Your Debate Tree:\n"
    "* Level-1 Nodes: Your main claims and arguments\n"
    "* Level-2 Nodes: Opponent's attacks on your claims and arguments\n"
    "* Level-3 Nodes: Your rebuttal on the attacks\n"
    "Opponent's Debate Tree:\n"
    "* Level-1 Nodes: Opponent's main claims and arguments\n"
    "* Level-2 Nodes: Your attacks on their claims and arguments\n"
    "* Level-3 Nodes: Opponent's rebuttal on your attacks\n\n"
    "## Input Information \n"
    "Debate flow trees with node data:\n"
    "**Your Tree**: \n{tree}\n\n"
    "**Opponent's Tree**: \n{oppo_tree}\n\n"
    "**Your Main Claims**: \n{claims}\n\n"
    "{{definition}}\n\n"
    "## Battlefields\n"
    "{{tips}}\n\n"
    "## Important Notes\n"
    "1. Organize your points logically with clear purpose statements. \n"
    "   - Clearly mention the actions you will take in each point. For example, 'we will address [X] concerns mentioned by the opponent, which are ' for attack / rebut actions or 'after that, we will propose our claims that' for propose action.\n"
    "   - Use clear indicators like first, second, third, etc. to organize your points.\n"
    "2. Avoid restating or simply repeating the same evidence or arguments across different points.\n"
    "3. Avoid repeating the similar arguments in your previous statement. Use the phrase 'as we have discussed' to refer to the previous statement.\n"
    "4. Only use facts that are generally accepted and don't require specific citation. Don't hallucinate any particular experimental results, statistical findings from named studies, or quotes from specific researchers until the evidence pool is provided\n"
    "5. When presenting alternatives or counterarguments, offer specific, implementable solutions rather than just criticism.\n"
    "6. Address both systemic and individual factors when relevant, showing how they can complement rather than contradict each other.\n"
    "\n"
    "## Output with the format (two parts, start with **Opening Plan** and then **Statement**):\n"
    "**Opening Plan**: Allocate your word budget and explain your rationale. Briefly mention one or two rhetorical techniques and logical fallacies to discuss. Ensure the total is {{n_words}} words. \n"
    "**Statement**: Generate an opening statement of {{n_words}} words in total, with no additional text\n"
)


propose_definition_prompt = (
    "The debate topic is: {motion}. You side is to {act} this topic .\n"
    "Please clarify the topic to address any:\n"
    "\t- Target groups, such as age, gender, income, etc.\n"
    "\t- Ambiguous or technical terms, such as 'ration'.\n"
    "\t- Important time, such as 'still', 'current', 'recent'.\n"
    "\t- Difference between two similar terms, such as 'matter' and 'crucial', 'writer' and 'writing'.\n"
    "\t- Multiple interpretations of key phrases\n"
    "\t- Scope-related uncertainties\n"
    "3. If there are some similar existing policies, you can refer to them for inspiration.\n\n"
    "Provide your response in the format: '**Definition**: [your one or two-sentence definition]'\n"
    "Don't output anything else."
)

main_claim_selection = (
    "## Task: Select Persuasive Claims for Debate\n"
    "You are participating in a formal debate on the topic: {motion}. Your position is {side}.\n"
    "Select most persuasive claims from the provided options, using the debate tree information.\n\n"
    "Note that it's possible that the debate tree is not provided, in this case, you can select claims without considering the debate tree.\n\n"
    "## Simulated Debate Flow Tree Structure\n"
    "Each claim has a simulated debate flow tree that simluate the potential back-and-forth between you and your opponent under this claim:\n"
    "* Level-0: The root claim (potential main claim for selection)\n"
    "* Level-1: Your opponent's rebuttal to the root claim\n"
    "* Level-2: Your defense against the opponent's rebuttal\n\n"
    "## Selection Criteria\n"
    "1. Diversity and Contrastive: Selected claims should cover different perspectives of the topic and be contrastive to each other without overlap\n"
    "2. Comprehensiveness: Claims should form a logical framework that addresses the most important aspects of the topic and distinguishes the specific subject from its broader category\n"
    "3. Consistency: Claims must be logically consistent with each other and with their Level-2 defenses\n"
    "4. Defensibility: Claims should have strong defenses at Level-2 against opponent rebuttals\n"
    "5. Context-aware: Consider the opponent's opening statement when selecting claims if it is provided\n\n"
    "## Input\n"
    "**Definition of the debate topic**:\n"
    "{definition}\n\n"
    "**Simulated Debate Flow Tree for each claim**:\n"
    "{tree}\n\n"
    "**Opponent's opening statement**:\n"
    "{context}\n\n"
    "**Claims to select from (All Level-0 claims)**:\n"
    "{claims}\n\n"
    "## Output\n"
    "Provide results in JSON format with three fields under the key of *selection*:\n"
    "* claims: a list of your selected claims. Each claim is a string. It usually contains 3 *very different claims* from non-overlapping perspectives.\n"
    "* framework: String describing the logical structure connecting these claims\n"
    "* explanation: String explaining how this framework support your stance and rebut the opponent's opening statement (if provided)\n"
)

claim_propose_prompt = """The debate topic  is: {motion}. You side is to {act} this topic .
Please propose {size} compelling claims that can be used in this debate. These claims should be relevant, distinct, and cover multiple dimensions of the topic.

Follow these enhanced guidelines to generate high-quality claims:

1. Element Decomposition: Break down the debate topic into its foundational elements. Identify:
   - The precise subject (what specifically is being discussed)
   - The exact action or position being proposed
   - The contextual boundaries: temporal, geographical, and situational constraints
   - Critical: Pinpoint the unique characteristics of the subject that differentiate it from broader categories
   - Define the specific subset or application of the subject that the topic addresses
   - Identify any implicit assumptions or values embedded in the topic framing

2. Multi-dimensional Analysis: For each element identified in Step 1:
   - Generate 5-10 relevant concepts, dimensions, or aspects 
   - Include economic (cost-benefit, efficiency, market impact), social (equity, community effects), political (power dynamics, governance), technological (innovation, disruption), environmental (sustainability, resource use), ethical (rights, duties, consequences), legal (regulatory impact, compliance), and cultural (traditions, identity) dimensions
   - Prioritize aspects with strong empirical support or logical foundations
   - Consider both immediate impacts and long-term consequences

3. Perspective Framework: Develop distinct perspectives from which to approach this topic:
   - Focus on perspectives that would strongly {act} the proposition
   - Identify key stakeholders (individuals, communities, vulnerable populations, businesses, institutions, government entities, global society)
   - Incorporate diverse value frameworks (utility, rights, virtue, care, fairness, liberty, justice, tradition, security)
   - For each perspective, articulate how the specific subject creates unique benefits or problems
   - Each perspective should isolate a single aspect of the topic for clarity and impact
   - Label each perspective with a specific, precise term reflecting its core orientation
   - Consider both mainstream and underrepresented perspectives

4. Claim Crafting: Create {size} distinct, persuasive claims by:
   - Integrating specific elements from Step 1 with critical concepts from Step 2
   - Filtering these combinations through well-defined perspectives from Step 3
   - Formulating clear, defensible assertions that directly support your position
   - Ensuring claims collectively address multiple dimensions while avoiding overlap
   - Focusing each claim on characteristics uniquely present in or significantly amplified by the specific subject
   - Constructing claims that establish clear causal relationships or evaluative judgments
   - Avoiding claims that could apply equally to broader categories or tangential issues
   - Ensuring each claim has potential for substantial supporting evidence

5: Claim Refinement: For each claim:
   - Revise for maximum clarity, impact, and concision - eliminate unnecessary details
   - Test the claim's specificity by asking if it uniquely applies to the exact topic rather than similar issues
   - Verify the claim's logical strength by identifying potential counterarguments and ensuring defensibility
   - Eliminate claims with substantial overlap to ensure diversity of arguments
   - If addressing an issue that exists in the broader category, explicitly articulate how the specific subject uniquely affects or transforms this issue
   - Balance between specificity and generalizability - claims should be specific to the topic yet broad enough to support detailed argumentation
   - Phrase claims as active, declarative statements that convey certainty

## Output Format
Return your response in JSON format with one key of *results*. The value of this key is a list of dictionaries. Each dictionary contains the following keys:
* *claim*: A string containing a single concise and short sentence that clearly states your position. Use simple words and sentences that are easy to understand. 
* *explanation*: A string that provides the logical foundation and rationale behind the claim, highlighting its significance.
* *perspective*: A string identifying the specific perspective or stakeholder viewpoint from which the claim is made.
* *concepts*: A list of strings representing the key concepts, values, or dimensions that the claim addresses.
* *strength*: A numerical value from 1-10 indicating the estimated argumentative strength of this claim, based on evidence availability and logical defensibility.
"""
