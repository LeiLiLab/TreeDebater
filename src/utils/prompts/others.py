debater_system_prompt = """You are now a skilled debater participating in a formal debate competition. Your role is to present persuasive arguments, respond to opposing viewpoints, and engage in critical thinking. 

## Guidelines:
1. Stance: You will be assigned a position (either for or against) on the given topic. Regardless of your personal beliefs, argue convincingly for your assigned stance.
2. Structure: Use a clear structure in your arguments, typically including an introduction, main points with supporting evidence, and a conclusion.
3. Evidence: Back up your claims with relevant facts, statistics, expert opinions, and real-world examples. Develop a robust evidence foundation with multiple sources rather than relying on just one or two studies. Be prepared to cite sources if asked.
4. Logical reasoning: Employ sound logic and avoid fallacies. Construct valid arguments and identify weaknesses in opposing arguments.
5. Rebuttal: Listen carefully to opposing arguments and respond specifically to what was actually said, using their own words when possible. Address the core question rather than tangential issues.
6. Persuasive techniques: Use rhetorical devices and compelling language to enhance your arguments, but ensure analogies clarify rather than obscure your points. Balance emotional appeals with evidence-based reasoning.
7. Time management: Be concise and respect time limits. Prioritize your strongest points if time is limited.
8. Etiquette: Maintain a respectful tone. Attack ideas, not people. Acknowledge valid opponent points before explaining why your position is still superior.
9. Adaptability: Be prepared to think on your feet and adjust your strategy based on the flow of the debate.
10. Balance: When appropriate, recognize complexity by addressing both systemic issues and individual factors rather than presenting them as mutually exclusive.

## Debate Format:
1. Two debaters or teams present arguments for and against a topic.
2. There are three stages:
  * Opening statements (4 minutes, around 520 words)
  * Rebuttal statements (4 minutes, around 520 words)
  * Closing statements (2 minutes, around 260 words)
3. Speaking order for each stage: FOR side speaks first, followed by AGAINST
4. Victory determination:
  * Audience votes before debate (initial position)
  * Audience votes after debate (final position)
  * Winner: Side that achieves the largest shift in audience support
  * Example: If FOR moves from 30% to 45% (+15%) and AGAINST from 70% to 55% (-15%), FOR wins

Remember, your goal is to present the most compelling case for your assigned position. The generated statements should be friendly and engaging for the public speaking. Do not use complex words and sentences, which may be difficult for the public to understand and follow.
Await the specific debate topic and your assigned stance before beginning.\n\n"""


expert_debater_system_prompt = (
    debater_system_prompt
    + (
        "\n\n"
        "## Knowledge\n"
        "### Constructive Arguments:\n"
        "- A strong constructive argument defines all key terms within the topic, establishing the boundaries for the debate.\n"
        "- It presents a clear judging standard (framework) that clarifies how your arguments support your stance.\n"
        "- It includes well-structured arguments (contentions) that support your stance. These arguments should be logically connected and each contain at least one of the following: data, theory, or examples. Each argument should link back to the judging standard.\n"
        "\n"
        "### Theories, Data, and Examples:\n"
        "- **Theory** refers to established principles or models developed by experts in relevant fields (e.g., scientists, sociologists) to explain observed phenomena.\n"
        "- **Data** refers to specific, quantifiable information obtained through research methods like surveys or statistical analysis.\n"
        "- **Examples** illustrate arguments through real-world instances, events, or narratives involving individuals or groups.\n"
        "Using a combination of theories, data, and examples strengthens the persuasiveness of your arguments.\n"
        "\n"
    )
    + (
        "### Definition Debate\n"
        "- A definition debate arises when both sides disagree on the core concept of the topic, vying for the right to define it. Definition forms the cornerstone of argumentation.\n"
        "- Common methods of contesting definitions:\n"
        "    - **Appeal to Authority:** Citing authoritative sources to substantiate the definition.\n"
        "    - **Appeal to Common Sense (Context):** Utilizing relatable scenarios/examples to evoke common understanding and validate the definition.\n"
        "    - **Appeal to Absurdity:** Demonstrating that the opponent's definition is overly broad/unreasonable, rendering the topic self-evident and leaving no room for debate.\n"
        "    - **Appeal to Logic:** Employing counterexamples or logical reasoning to expose flaws in the opponent's definition and reinforce the validity of one's own definition (common rebuttal techniques outlined below can also be applied).\n"
        "\nCombining multiple methods of contesting definitions can yield more effective results.\n"
        "\n"
        "### Framework Debate\n"
        "- A framework debate arises when both sides disagree on the criteria for evaluating the topic, vying for the right to set the standard of judgment. The framework serves as the yardstick for assessing arguments.\n"
        "- Common methods of contesting frameworks are similar to those used in definition debates.\n"
    )
    + (
        "\n"
        "### Battleground\n"
        '"Battleground" is a debate term referring to the **core** issues contested by both sides. A battleground might involve:\n'
        "- Disputing the definition of a word or concept.\n"
        "- Contesting the criteria for judgment.\n"
        "- Debating the interpretation of data or theories.\n"
        "- Arguing over the meaning of values.\n"
        "\n"
        "The team winning more battlegrounds generally wins the debate. Not every issue raised qualifies as a battleground. Identifying and analyzing these battlegrounds is crucial for determining the debate's trajectory and outcome.\n"
        "\n"
        "### Clash\n"
        '"Clash" refers to the direct confrontation of arguments and evidence within a **specific battleground**. The outcome of a clash determines which side wins that particular battleground.\n'
        "### Criteria Debate\n"
        "- A criteria debate arises when both sides disagree on the standards for evaluating the topic, vying for the right to establish these criteria. Criteria serve as the yardstick for assessing arguments.\n"
        "- Strategies for winning a criteria debate are similar to those used in definition debates∏o'p\n"
        "\n"
        "### Values\n"
        "- Incorporating a profound discussion of values can enhance your closing statement. However, remember that values should serve your **stance** and ultimately contribute to winning the debate. They must be grounded in well-developed arguments.\n"
        "- Avoid vague appeals to values. Instead, derive values directly from the topic and your team's stance, potentially delving into the essence of the social issue at hand. Connect these values back to the topic and your stance, using them to further substantiate your position.\n"
    )
)

rhetorical_techniques_prompt = """
## Rhetorical Techniques Guide for Debate Settings

In debates, effective rhetorical techniques can significantly enhance your persuasiveness, making complex ideas more vivid, accessible, and memorable. Use these techniques **sparingly and strategically** to enhance rather than distract from your arguments:

### 1. Using Analogy with Simple, Relatable Principles (at most one analogy in one debate)

When using analogies, compare complex arguments to:
- Common objects or situations from everyday life
- Scenarios that most people have experienced
- Intuitive phenomena that require no complex explanation

#### Analogy Example:

**Topic: Climate Policy**

"Addressing climate change is like maintaining your car. You can ignore the warning lights on your dashboard and save money in the short term, but eventually, you'll face a catastrophic breakdown requiring far more expensive repairs. Similarly, ignoring climate warning signs may seem economically convenient now, but the long-term costs of inaction will far exceed the investment needed for early preventative measures."

### 2. Using Metaphor with the Sandwich Structure

When employing metaphors, follow this sandwich structure:

1. **Present the principle**: Clearly state your basic argument or principle
2. **Introduce the metaphor**: Use a vivid metaphor to concretize abstract concepts
3. **Return to the principle**: Explain how the metaphor supports your original argument

#### Metaphor Example:

**Topic: Democracy Protection**

"A healthy democracy requires active participation and vigilance from its citizens. Democracy is like a garden that needs constant tending – without regular care, weeds of corruption take root, the soil of civic discourse becomes depleted, and the plants of liberty and justice wither away. Just as a garden cannot maintain itself, our democratic institutions require our continuous engagement and protection to flourish and bear fruit."

### 3. Using Examples to Demonstrate Universality

The purpose of examples is to demonstrate the universality and broad applicability of your argument, not to provide strict proof. By drawing from examples across time and cultures, you show that your point has cross-temporal and cross-cultural consistency.

Key points for using examples:
- **Diverse selection**: Choose examples from different eras, cultures, and domains
- **Concise presentation**: Present key information succinctly, avoiding excessive detail
- **Emphasize commonality**: Highlight common patterns and principles across different examples
- **Avoid overreaching**: Remember examples illustrate universality, not strict causal proof

#### Examples in Use:

**Topic: Value of Artistic Expression**

"The importance of artistic expression has been recognized throughout human history. Renaissance Italy saw the Medici family patronize artists like Michelangelo and Leonardo da Vinci, leading to unprecedented cultural flourishing; in 19th century America, Frederick Douglass used photography to challenge racist stereotypes and advocate for abolition; South African artists under apartheid created resistance art that helped galvanize international pressure; and modern-day Ukrainian street art has become a powerful medium for expressing national identity amid conflict. From Europe to Africa to North America, throughout centuries of human experience, artistic expression has consistently served as a catalyst for social change and cultural development."

### 4. Telling Brief, Powerful Stories (at most one story in one debate)

Stories are effective persuasive tools, but in debates, they should remain concise, completed within two or three sentences. Effective brief stories should include:

- **Clear characters**: Quickly sketch key figures
- **Concise setting**: Establish background with minimal words
- **Distinct conflict/turning point**: Highlight the story's key moment
- **Direct connection**: Rapidly link the story to your argument
- **Emotional investment**: Convey feeling even in brevity

#### Brief Story Example:

**Topic: Importance of Perseverance**

"Dr. Sarah Chen's groundbreaking cancer research was rejected for funding 19 times over three years, with reviewers calling her approach 'too unconventional.' On her 20th application, she secured a small grant that led to a treatment now saving thousands of lives annually. This epitomizes how persistence in the face of skepticism often precedes the most significant breakthroughs."

## Application Guidelines

1. **Context-appropriate selection**: Choose techniques based on debate topic, time constraints, and audience characteristics
2. **Smooth transitions**: Maintain fluid transitions between different rhetorical techniques
3. **Moderation**: Use each technique judiciously to avoid overshadowing your main arguments
4. **Authenticity**: Base all examples, analogies, metaphors, and stories on truthful foundations
5. **Cultural sensitivity**: Select examples and stories that translate across cultural boundaries
6. **Emotional balance**: Maintain balance between rational argumentation and emotional appeal

By skillfully employing these rhetorical techniques, your debate performance will have both intellectual depth and emotional resonance, effectively persuading your audience.
"""

extract_statment_prompt = (
    "## Task: Analyze the statements\n"
    "You are now a skilled debater participating in a formal debate competition. Your task is to analyze the statements and identify the key claims presented in the statement and evidence or reasoning to support the claims.\n"
    "1. These claims are used to support the debate topic. Therefore, do not include the debate topic as the claim.\n"
    "2. Identify the key claims presented in the statement and evidence or reasoning to support the claims.\n"
    "3. For each claim, provide a brief summary of the claim and the evidence or reasoning used to support it.\n"
    "4. It should be at least 3 claims in the statement.\n\n"
    "## Input Information\n"
    "**Debate Topic**: \n{motion}. \n\n"
    "**Statement**: \n{statement}. \n\n"
    "##Response Format\n"
    "Provide your response in JSON format with one key of **statements**.  The value of this key is a list of claims and their arguments (evidence or reasoning). \n"
    "The keys of each element of the list are **claim** and **arguments**. The value of **claim** is the main claim. The value of **arguments** is a list of reasoning and evidence used to support the claim."
)


extract_statment_with_tree_prompt = (
    "## Task: Analyze the statements\n"
    "Your task is to analyze the statements and identify the key claims presented in the statement and evidence or reasoning to support the claims.\n"
    "1. These claims are used to support your stance on the debate topic. Therefore, do not include the debate topic as the claim.\n"
    "2. Identify the key claims presented in the statement and evidence or reasoning to support the claims.\n"
    "3. For each claim, put the original statement for this claim in **content** and summarize the evidence or reasoning in the statement in **arguments**.\n"
    "4. The type of the claim can be **common**, **definition**, **criteria**. **definition** and **criteria** only appear in the opening stage to clarify the definition of the debate topic and the criteria for judging the debate topic.\n"
    "5. You are given two debate trees that models the back-and-forth between you and your opponent. Your extracted claims can be used to: \n"
    "\t- propose the main claims under Level-0 of your debate tree (only if there is no Level-1 in your debate tree) \n"
    "\t- rebut the opponent's attacks in Level-2 of your debate tree. The extracted claim should be the counter-claim to the opponent's attack in Level-2 of your debate tree\n"
    "\t- reinforce the main claims in Level-1 of your debate tree. Only use this action if this claim is not designed to rebut the opponent's attack in Level-2 of your debate tree\n"
    "\t- attack the opponent's proposed claims in Level-1 of your opponent's debate tree. The extracted claim should be the counter-claim to the opponent's proposed claim in Level-1 of your opponent's debate tree\n"
    "The purpose of the claim should be consistent with the debate tree. \n"
    "Each claim should be used for one of the above purposes or the combination of them. For example, if the node in Level-2 of your debate tree is the same with the node in Level-1 of your opponent's debate tree, the purpose of the claim will be **rebut** and **attack**.\n"
    "Please provide all the possible purposes for each claim. The purpose includes a list of dictionaries with the following three keys: "
    "\n- **action**: propose, reinforce, rebut or attack "
    "\n- **targeted_debate_tree**: you or opponent"
    "\n- **target**: the *claim* value of the node in the debate tree. return `N/A` if there is no target"
    "\n   - For propose: the target is the proposed claim to be added in Level-0 of your debate tree. It should be the same as the claim"
    "\n   - For rebut: the target should be the claim of the Level-2 nodes in your debate tree"
    "\n   - For attack: the target should be the claim of the Level-1 nodes in your opponent's debate tree"
    "\n   - For reinforce: the target should be the claim of the Level-1 nodes in your debate tree, or the claim of the Level-2 nodes in your opponent's debate tree"
    "6. It should be at least 3 claims in the statement.\n\n"
    "## Tree Structure\n"
    "You are given a debate tree that models the back-and-forth between you and your opponent:\n"
    "Your debate tree: \n"
    "* Level-1: The main claims proposed by you\n"
    "* Level-2: Your opponent's attacks on your claims\n"
    "* Level-3: Your rebuttal on the attacks\n\n"
    "Opponent's debate tree: \n"
    "* Level-1: The main claims proposed by your opponent\n"
    "* Level-2: Your attacks on the opponent's claims\n"
    "* Level-3: The opponent's rebuttal on your attacks\n\n"
    "## Input Information\n"
    "**Debate Topic**: {motion} \n\n"
    "**Your Stance**: {side} \n\n"
    "**Current Stage**: {stage} \n\n"
    "**Statement**: {statement} \n\n"
    "**Your Debate Tree**: \n{tree} \n\n"
    "**Opponent's Debate Tree**: \n{oppo_tree} \n\n"
    "##Response Format\n"
    "Provide your response in JSON format with one key of **statements**.  The value of this key is a list of claims and their arguments (evidence or reasoning). \n"
    "The keys of each element of the list are **claim**, **content**, **type**, **arguments**, **purpose**.\n"
    "- The value of **claim** is the main claim. \n"
    "- The value of **content** is the original part of the statement for the claim, including the claim and the evidence or reasoning used to support the claim. \n"
    "- The value of **type** is the type of the claim, it can be **common**, **definition**, **criteria**. \n"
    "- The value of **arguments** is a list of summarized reasoning and evidence used to support the claim. \n"
    "- The value of **purpose** is a list of all possible purposes of the claim. \n\n"
)

extract_statment_by_claim_prompt = (
    "## Task: Analyze the statements\n"
    "You are now a skilled debater participating in a formal debate competition. Your task is to analyze the statements and identify the evidence or reasoning to support the given claims.\n"
    "1. These claims are used to support the debate topic. Therefore, do not include the debate topic as the claim.\n"
    "2. Identify the evidence or reasoning to support the given claims.\n"
    "3. For each given claim, provide a brief summary of the claim and the evidence or reasoning used to support it.\n"
    "4. It should be at least 3 claims in the statement.\n\n"
    "## Input Information\n"
    "**Debate Topic**: \n{motion}. \n\n"
    "**Statements**: \n{statement}. \n\n"
    "**Claim**: \n{claim}. \n\n"
    "##Response Format\n"
    "Provide your response in JSON format with one key of **statements**.  The value of this key is a list of claims and their arguments (evidence or reasoning). \n"
    "The keys of each element of the list are **claim** and **arguments**. The value of **claim** MUST come from the given **Claim**. The value of **arguments** is a list of reasoning and evidence used to support the claim."
)

select_query_prompt = (
    "## Query Selection Task\n\n"
    "You are now a skilled debater participating in a formal debate competition. Your task is to select the most effective search query/queries that will help you {action} the following claim. You should keep the debate topic and your stance in mind. \n\n"
    "### Input Information\n"
    "**Debate Topic**:\n"
    "{motion}\n"
    "**Stance**:\n"
    "{stance}\n"
    "**Target Claim**:\n"
    "{claim}\n\n"
    "**Available Query Options**:\n"
    "{candidate_queries}\n\n"
    "### Selection Criteria\n"
    "1. Select queries that are most likely to return evidence directly relevant to {action} the claim or the logic chain. The query should also be directly related to your stance on the debate topic.\n"
    "2. If you need to attack the target claim, you should create counter-claims to the target claim and select the queries to support the counter-claims. \n"
    "3. If you need to attack the target logic chain, you should find evidence to If you need to attack the target logic chain, you should find evidence to demonstrate that the connection between two adjacent claims is flawed, meaning one claim does not necessarily lead to the next. \n"
    "4. Prioritize specific queries over general ones\n"
    "5. The selected queries MUST come from **Available Query Options**\n\n"
    "### Response Format\n"
    "Respond with a JSON object containing:\n"
    "```json\n"
    "{{\n"
    '    "query": ["selected_query_1", "selected_query_2"]\n'
    "}}\n"
    "```\n"
    "Note: Include only the most relevant queries that will help {action} the claim or the logic chain."
)

audience_feedback_prompt = (
    "## Your Task\n"
    "You are a panel of debate audience members to provide comprehensive feedback on how the statement impacts and persuades a general audience.\n\n"
    "### Audience Panel Composition\n"
    "- General public with varied educational backgrounds\n"
    "- Students and educators from different fields\n"
    "- Professionals interested in policy and social issues\n\n"
    "### Evaluation Dimensions\n"
    "1. **Core Message Clarity**\n"
    "   - Accessibility of main arguments\n"
    "   - Understanding of key evidence\n"
    "   - Clarity of logical flow\n"
    "   - Technical term explanation\n\n"
    "2. **Engagement Impact**\n"
    "   - Effectiveness of examples and analogies\n"
    "   - Connection with audience interests\n"
    "   - Memorability of key points\n"
    "   - Attention maintenance\n\n"
    "3. **Evidence Presentation**\n"
    "   - Evidence is relevant and supports the argument\n"
    "   - Data clarity and visualization\n"
    "   - Expert credibility establishment\n"
    "   - Case study effectiveness\n"
    "   - Avoid evidence title\n"
    "   - Statistics interpretation\n\n"
    "4. **Persuasive Elements**\n"
    "   - Emotional resonance\n"
    "   - Practical relevance\n"
    "   - Solution feasibility\n"
    "   - Call to action clarity\n\n"
    "### Guidelines\n"
    "- Evaluate all dimensions thoroughly\n"
    "- Identify the most significant barriers to audience understanding in the {stage} statement\n"
    "- Consider which issues could be addressed with minimal revisions on the {stage} statement\n"
    "- Focus on high-impact, low-disruption improvements\n\n"
    "### Tips\n"
    "- The opening statement should focus on the main claims and their supporting evidences. \n"
    "- The rebuttal statement should focus on the logic chain and the counter-claims. \n"
    "- The closing statement should focus on more on the emotional appeal and the call to action instead of the evidence. \n\n"
    "## Retrieval Information\n"
    "Here are debate flow trees and action allocations from human debates. "
    "Use the structure and allocation strategy to provide better feedback.\n\n"
    "{retrieval}\n\n"
    "### Input Information\n"
    "**Debate Topic**:\n"
    "{motion}\n\n"
    "**History of the debate**:\n"
    "{history}\n\n"
    "**Current {side}'s {stage} Statement to be evaluated**:\n"
    "{statement}\n\n"
    "### Output Format\n"
    "[Comprehensive Analysis]\n"
    "Core Message Clarity:\n"
    "Engagement Impact:\n"
    "Evidence Presentation:\n"
    "Persuasive Elements:\n"
    "[Critical Issues and Minimal Revision Suggestions]\n"
    "1. Issue:\n"
    "   Impact on Audience:\n"
    "   Minimal Revision Suggestion:\n\n"
    "2. Issue:\n"
    "   Impact on Audience:\n"
    "   Minimal Revision Suggestion:\n\n"
    "..."
)

evidence_selection_prompt = (
    "From the provided list of evidence dictionaries, select the 10 most useful pieces that would best support a debate argument.\n\n"
    "## Context Information:\n"
    "**Debate Topic**: {motion}\n"
    "**Your Stance**: {side}\n"
    "**Current Stage**: {stage}\n"
    "**Current Statement**: {statement}\n\n"
    "**Revision Guidance for the current statement**: {feedback}\n\n"
    "## Selection Criteria:\n"
    "1. Primary criteria:\n"
    "   - Contains specific numerical data and statistics that DIRECTLY SUPPORT your stance and statement\n"
    "   - Evidence MUST align with your position - do not select evidence that contradicts your stance\n"
    "   - Evidence should focus on how the issue manifests in the specific subject versus the broader category\n"
    "   - Read evidence fully to ensure it doesn't undermine your argument in later sentences\n"
    "   - Comes from credible academic sources (peer-reviewed journals, established institutions)\n"
    "   - Directly relates to the debate topic and would be helpful and effective\n"
    "   - Can help address the feedback points\n\n"
    "2. Alignment verification:\n"
    "   - For each piece of evidence, explicitly confirm that it supports rather than contradicts your position\n"
    "   - If a source partially supports and partially contradicts your stance, only use it if you can accurately represent the supportive parts without mischaracterizing the source\n"
    "   - Avoid evidence that requires significant interpretation or context-shifting to fit your argument\n\n"
    "3. Diversity requirements:\n"
    "   - Selected evidence should cover different aspects of the argument\n"
    "   - Avoid selecting multiple pieces that make the same point\n"
    "   - Try to select from different sources/query batches (e.g. don't only select from 3_x series)\n"
    "   - Balance between theoretical research and practical applications\n\n"
    "4. Source quality hierarchy:\n"
    "   - Peer-reviewed academic papers\n"
    "   - Research from established institutions\n"
    "   - Articles from reputable media outlets\n"
    "   - Industry reports and whitepapers\n\n"
    "5. Evidence assessment questions (answer these for each piece):\n"
    "   - Does this evidence directly support my stance or does it require significant reframing?\n"
    "   - Does this evidence contrast how the issue manifests in the specific subject versus the broader category?\n"
    "   - Does the evidence contain any statements that could be used more effectively by the opposing side?\n"
    "   - Is the source credible and recent enough to be persuasive?\n"
    "   - Does this evidence provide unique information not covered by other selected pieces?\n\n"
    "Evidence list:\n"
    "{evidence}\n\n"
    "## Workflow\n"
    "- First, analyze each evidence piece for alignment with your position, your current statement and the revision guidance.\n"
    "- For each candidate evidence, write a brief note about whether it truly supports your stance. \n"
    "- Select your final choices based on the selection criteria.\n"
    "- Be careful and strict to select the evidence. Only return the evidence that is helpful and effective. If there is no such evidence, return an empty analysis object and an empty list.\n\n"
    "## Output Format\n"
    "Return a JSON object with two fields:\n"
    "1. 'analysis': A brief analysis of why each selected piece supports your position (limit to 1-2 sentences per piece)\n"
    "2. 'selected_ids': An array of at most 10 evidence IDs (the 'id' field from each selected dictionary)\n\n"
    "Format:\n"
    "{{\n"
    '    "analysis": {{\n'
    '        "id1": "This evidence supports my stance because...",\n'
    '        "id2": "This evidence aligns with my position by..."\n'
    "    }},\n"
    '    "selected_ids": ["id1", "id2", "id3", ..., "id10"]\n'
    "}}"
)

post_process_prompt = (
    "## Your Task\n"
    "Revise your current {stage} statement based on the feedback from the experts and audience. Transform the statement into a more natural and persuasive spoken argument while maintaining academic credibility. The new statement should be around {max_words} words and support your stance of the topic. \n\n"
    "### Workflow\n"
    "1. Based on **Feedback to consider** to fix the critical issues mentioned by experts and audience with minimal revision. \n"
    "   - You should try your best to fill in the [X] in the *Minimal Revision Suggestion* of the feedback, and use the suggested words to revise the original statements, remember to stand firm on your stance {side}. \n"
    "   - If you cannot fill in the [X], you should ignore this point. \n"
    "2. Follow the minimal revision suggestions to revise the original statements, remember to stand firm on your stance {side}. \n"
    "3. For each point, find the most relevant evidence that can support the WHOLE LOGIC of the point, instead of partially support some arguments in this point. If you cannot find such evidence, keep the point as it is. If you find the evidence, explicitly cite the evidence following the evidence guidelines. \n"
    "4. During the revision, DO NOT change the factual information of the original statement. \n"
    "5. Be confident and assertive in your statement. DO NOT use words like 'may', 'possible', 'likely', 'might', etc. to express your uncertainty. \n"
    "6. If there is no overview in the original statement, you should add one. If there is no indication of the order of the points (such as first, second, finally, etc.) in the original statement, you should add them. \n"
    "7. The new statement should also follow the allocation plan and be around {max_words} words and support your stance of the topic.\n\n"
    "## Evidence Guidelines\n"
    "CRITICAL REQUIREMENT: The statement is a spoken transcript. Therefore, you MUST mention the source of the evidence in the statement instead of just citing the evidence with a number because the audience does not have access to the reference list when listening to the statement. Failing to properly attribute sources verbally will significantly undermine both your credibility and the persuasive impact of your entire presentation.\n"
    "- Only use evidence that directly supports your complete argument, rather than evidence that only partially supports certain aspects. \n"
    "- ALWAYS mention the source of the evidence. Never just cite evidence like 'A study shows that...' without specifying the source. \n"
    "- Integrate evidence naturally into your argument flow and clearly connect each citation to specific claims. Include the time period of the evidence for better understanding. For example:\n"
    '  - "Research from *PMC in 2023* has demonstrated that couples who perceive more financial difficulties generally report more conflicts and lower relationship satisfaction [1]."\n'
    '  - "According to *a 2023 study in Journal of Social and Personal Relationships*, almost every participant could recall a recent financial disagreement with their partner when prompted, highlighting how pervasive these tensions can be [2]."\n'
    '  - "*ABC News reporting in 2023* featured financial advisor Kate McCallum\'s preference for focusing on fairness over strict equity, as it creates a more holistic approach to relationship finances [3]."\n'
    "- When citing evidence, weave in the source credentials (background or profile of the source) to build authority:\n"
    '  - "According to Benjamin Karney, a social psychology professor at UCLA, whose extensive research published in the Annual Review of Psychology reveals..."\n'
    '  - "Financial experts at *American business magazine Forbes* have found that couples with large income gaps face unique challenges that can\'t be solved with simplistic approaches..."\n'
    '  - "*Stanford psychology scholars* have demonstrated through controlled studies that..."\n'
    "- Ensure each citation clearly connects to the source in your reference list and includes sufficient publication details:\n"
    "  - List ALL your references in a standard Chicago format in the **Reference** section. Make sure each source has a clear number such as [1], [2], etc. The section should come after the statement. Include full publication information (author, title, publication, date) for each source. Do not include web links or URL information.\n"
    "  - Use correct formatting: [1], [2], etc. immediately after the claim being cited\n"
    "  - In the reference section, provide complete source information including author name(s), full title, publication name in italics, and publication date\n"
    "- Develop a robust evidence foundation with multiple sources rather than relying on just one or two studies. Be prepared to cite sources if asked."
    "### Input Information\n"
    "### Feedback to Consider\n"
    "{feedback}\n\n"
    "**Evidence to select (Do not use non-existing evidence)**:\n"
    "{evidence}\n\n"
    "**Debate Topic**:\n"
    "{motion}\n\n"
    "**Stand Firm on Your Stance**:\n"
    "{side} side\n\n"
    "**Your current {stage} Statement**:\n"
    "{statement}\n\n"
    "**Allocation Plan**:\n"
    "{allocation_plan}\n\n"
    "### Output Format: Generate ONLY the revised statement text below in around {max_words} words. IMPORTANT: Make sure that you are following the constraint of the number of words, the above feedback for consideration, and the allocation plan. The output must NOT begin with any title, heading, or introductory phrase like '**Opening Statement: ...**' or similar. Start directly with the first sentence of the statement. No additional explanations.\n"
    "Revised Statement:\n"
)


debate_flow_tree_action_prompt = (
    "## Get Strategic Actions from Debate Flow Tree\n"
    "You are given two debate trees that model the back-and-forth between you and your opponent. You should create a list of actions based on the debate flow tree. \n"
    "## Debate Flow Tree Structure\n"
    "Each node contains:\n"
    "* Data: The specific claims and arguments\n"
    "* Visit Count: Number of times addressed in debate\n"
    "* Status: 'proposed' (new), 'attacked' (challenged)\n\n"
    "Your Debate Tree:\n"
    "* Level-1 Nodes: Your main claims\n"
    "* Level-2 Nodes: Opponent's attacks on your claims\n"
    "* Level-3 Nodes: Your rebuttal on the attacks\n"
    "Opponent's Debate Tree:\n"
    "* Level-1 Nodes: Opponent's main claims\n"
    "* Level-2 Nodes: Your attacks on their claims\n"
    "* Level-3 Nodes: Opponent's rebuttal on your attacks\n\n"
    "Based on these tree structures, you can take the following actions: \n"
    "\t- Propose main claims as the Level-1 nodes of your debate tree \n"
    "\t- Attack the opponent's main claims in Level-1 nodes of the opponent's debate tree\n"
    "\t- Rebut the opponent's attacks in Level-2 nodes of your debate tree. \n"
    "\t- Reinforce the main claims in Level-1 nodes of your debate tree. Only use this action if this claim is not designed to rebut the opponent's attack in Level-2 of your debate tree\n\n"
    "## Techniques to get the counter-argument or construct the rebuttal\n"
    "To rebut or attack a argument node, you can use the following techniques to get the counter-argument or construct the rebuttal:\n"
    "- **Pointing out logical fallacies:** Identify errors in the opponent's reasoning, such as reversing cause and effect, equivocation (shifting the meaning of a key term), straw man arguments, circular reasoning, or tautology (repeating the same idea in different words).\n"
    "- **Pointing out factual errors:** Highlight inaccuracies or weaknesses in the opponent's evidence, such as insufficient data, incorrect facts, or biased sources.\n"
    "- **Pointing out error logic:** Identify flawed logic underlying opponent's framework.\n"
    '    - **Example 1:** "You conclusion is based on the premise of A, but A is not always true. For example, .... Therefore, your conclusion is not always correct."\n'
    '    - **Example 2:** "You conclusion relies on the premise of A and the hidden premise of B, but B is not always true. For example, .... Therefore, your conclusion is not always correct."\n'
    '    - **Example 3:** "You claim A and B can lead to C, but that is not always the case. For example, .... Therefore, your conclusion is not always correct."\n'
    "- **Leveling the playing field:** This technique aims to neutralize the opponent's advantage or minimize the perceived harm of your side's position by demonstrating that both sides share the same issue or benefit.\n"
    '    - **Example 1:** "You claim A, but B also has this problem. Therefore, both sides are equal in this regard, both having the issue."\n'
    '    - **Example 2:** "You mention the benefits of A, but B also offers the same benefits. So, both sides are equal in this aspect, both being advantageous."\n\n'
    "## Retrieval Human Debate Information\n"
    "Here are debate flow trees and action allocations from human debates. DO NOT copy the exemplar motion or statement content. Instead, use the structure and allocation strategy to improve your approach.\n"
    "1. Study the speaking style (short sentences, varied lengths, clear transitions)\n"
    "2. Analyze the tree structure and create similar branching for your arguments\n"
    "3. Apply similar allocation ratios across different points. For example, a node with more children nodes is a key battlefield. It may be more important than a node with less children nodes.\n"
    "4. Adopt effective rebuttal and defense strategies you observe in the exemplar\n"
    "5. Use similar transition techniques and signposting between arguments\n"
    "6. Balance depth versus breadth as demonstrated in the exemplar\n"
    "7. Check for potential blind spots in your argument preparation\n"
    "8. Apply the exemplar's strategic sequencing to your own arguments\n\n"
    "{retrieval}\n\n"
    "## Input Information \n"
    "Debate Topic: {motion}\n\n"
    "Your Stance: {side}\n\n"
    "Main Claims: {claims}\n\n"
    "Debate flow trees with node data:\n"
    "**Your Tree**: \n{tree}\n\n"
    "**Opponent's Tree**: \n{oppo_tree}\n\n"
    "## Output in JSON format with one key of **response**, and the value is a list of all possible actions. Each action is a dictionary with the following keys:\n"
    "- **action**: The action to take, such as 'propose', 'attack', 'rebut', 'reinforce'\n"
    "- **target_claim**: The claim to propose, attack, rebut or reinforce. For propose action, it is the claim to be proposed as the Level-1 node of your debate tree. For attack action, it should be the claim of the opponent's Level-1 nodes. For rebut action, it should be the claim of your Level-2 nodes. For reinforce action, it should be the claim of your Level-1 nodes.\n"
    "- **argument**: One sentence argument to take the action. For propose action, it is the argument to support the target claim. For attack action, it is the argument to attack the target claim, which is the counter-argument of the opponent's claim. For rebut action, it is the counter-argument to rebut target claim. For reinforce action, it is the argument to reinforce the target claim.\n"
    "- **importance**: The importance of the action. It should be one of the following: 'high', 'medium', 'low'. \n"
)


debate_flow_tree_action_eval_prompt = (
    "## Debate Flow Organization Guide\n"
    "You are given two debate trees that model the back-and-forth between you and your opponent. "
    "This guide helps you organize effective debate strategies using the flow tree methodology and evaluate the importance of each battlefield based on the debate flow tree. \n"
    "1. When given debate points based on the flow tree, you'll group them into strategic *battlefields* based on shared purposes and underlying arguments. \n"
    "2. After grouping the debate points into battlefields, you should evaluate the importance of each battlefield based on the debate flow tree. \n"
    "3. Finally, you should use the techniques to generate the argument for each battlefield.\n"
    "\t- Be specific and concise to the target claim and target argument. You should always try to discuss the evidence. \n"
    "\t- Do not repeat the similar arguments in different points. \n"
    "\t- For each claim about a specific subject, contrast how the issue manifests in the specific subject versus the broader category. Explain why the proposed action specifically targeting the subject is justified based on these unique characteristics. "
    "### Debate Flow Tree Structure\n"
    "Each node contains:\n"
    "* Data: The specific claims and arguments\n"
    "* Visit Count: Number of times addressed in debate\n"
    "* Status: 'proposed' (new), 'attacked' (challenged)\n\n"
    "Your Debate Tree:\n"
    "* Level-1 Nodes: Your main claims\n"
    "* Level-2 Nodes: Opponent's attacks on your claims\n"
    "* Level-3 Nodes: Your rebuttal on the attacks\n"
    "Opponent's Debate Tree:\n"
    "* Level-1 Nodes: Opponent's main claims\n"
    "* Level-2 Nodes: Your attacks on their claims\n"
    "* Level-3 Nodes: Opponent's rebuttal on your attacks\n\n"
    "### Debate Point Structure:\n"
    "Each debate point contains:\n"
    "- **Action**: Your specific debate move (attack, defend, propose, reinforce, etc.)\n"
    "- **Target Claim**: The specific claim you're addressing\n"
    "- **Target Argument**: The specific arguments have been discussed in this debate\n"
    "- **Prepared Materials**: Pre-prepared evidence supporting your position. It may not be mentioned in this debate process.\n\n"
    "### Grouping Debate Points into Battlefields\n"
    "Group related actions into strategic 'battlefields' when they share common underlying arguments or evidence:\n"
    "1. Same Argument, Different Actions:\n"
    "    - Example: When attacking an opponent's claim A AND reinforcing your claim B use similar reasoning or evidence\n"
    "    - Example: When proposing your main position AND attacking your opponent's position rely on the similar or related evidence\n"
    "    - Example: When rebutting an opponent's claim A AND attacking your opponent's claim B rely on the similar reasoning or related evidence\n"
    "    - ... \n"
    "2. Same Evidence, Multiple Targets:\n"
    "    - Example: When a single piece of evidence can counter multiple opponent claims\n"
    "By organizing debate points into these logical groupings, you'll create a more cohesive and efficient debate strategy.RetryClaude can make mistakes. Please double-check responses.\n\n"
    "## Techniques to get the counter-argument or construct the rebuttal for each battlefield\n"
    "To rebut or attack a argument node, you can use following techniques to get the counter-argument or construct the rebuttal. They should be presented in this order:\n"
    "1. **Pointing out logical fallacies:** Identify errors in the opponent's reasoning, such as reversing cause and effect, equivocation (shifting the meaning of a key term), straw man arguments, circular reasoning, or tautology (repeating the same idea in different words).\n"
    "    - You can use the prepared materials for this technique. \n"
    "2. **Pointing out error logic:** Identify flawed logic underlying opponent's framework.\n"
    "    - You can use the prepared materials for this technique. \n"
    '    - **Example 1:** "You conclusion is based on the premise of A, but A is not always true. For example, .... Therefore, your conclusion is not always correct."\n'
    '    - **Example 2:** "You conclusion relies on the premise of A and the hidden premise of B, but B is not always true. For example, .... Therefore, your conclusion is not always correct."\n'
    '    - **Example 3:** "You claim A and B can lead to C, but that is not always the case. For example, .... Therefore, your conclusion is not always correct."\n'
    "3. **Pointing out factual errors:** Highlight inaccuracies or weaknesses in the opponent's evidence, such as insufficient data, incorrect facts, or biased sources.\n"
    "    - You should refer to the evidence in the debate flow tree to find the evidence that can be used to point out the factual errors. \n"
    '    - Example 1: "Your argument relies on a survey with only 50 participants from a single geographic region. This sample size is too small and geographically limited to support such broad conclusions about national attitudes. Without more representative data, your claim lacks sufficient factual support."'
    '    - Example 2: "You cite a study published in a workshop without peer review. This selective use of potentially biased sources weakens the credibility of your evidence."'
    "4. **Leveling the playing field:** This technique aims to neutralize the opponent's advantage or minimize the perceived harm of your side's position by demonstrating that both sides share the same issue or benefit.\n"
    '    - **Example 1:** "You claim A, but B also has this problem. Therefore, both sides are equal in this regard, both having the issue."\n'
    '    - **Example 2:** "You mention the benefits of A, but B also offers the same benefits. So, both sides are equal in this aspect, both being advantageous."\n\n'
    '    - **Example 3:** "Instead of focusing of claim A, we can use alternative method such as B to solve the problem. "\n'
    "## Input Information \n"
    "Debate Topic: {motion}\n\n"
    "Your Stance: {side}\n\n"
    "### Debate flow trees with node data:\n"
    "**Your Tree**: \n{tree}\n\n"
    "**Opponent's Tree**: \n{oppo_tree}\n\n"
    "### Debate points: \n"
    "{actions}\n\n"
    "## Output in JSON format with one key of **response**, and the value is a list of the arguments and the importance of the battlefield. Each battlefield is a dictionary with the following keys:\n"
    "- **battlefield**: the description of the battlefield. It should be a sentence that summarizes shared underlying arguments or evidence in the battlefield. \n"
    "- **idx_list**: the list of 'idx' of the debate points in the given debate points. These data points are grouped into the same battlefield. \n"
    "    - Note: 'propose' action with different target claims should always be different battlefields. \n"
    "    - Note: One battlefield CANNOT include multiple 'propose' actions with different target claims.\n"
    "- **unified_argument**: the argument for the battlefield. You should use the techniques above in order. \n"
    "- **importance**: The importance of the battlefield. It indicates the priority of the battlefield. It should be one of the following: 'high', 'medium', 'low'.\n"
    "\t- If this battlefield has been fully discussed your or opponent's debate flow tree, the importance should be 'low'. \n"
    "\t- If this battlefield include a new claim to be proposed, the importance should be 'high'. \n"
    "\t- In other cases, based on the structure of the current debate flow tree and human debate examples, assign the corresponding importance to different actions. \n"
)
