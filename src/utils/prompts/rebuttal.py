context = (
    "Now it comes the rebuttal phase, where you respond to your opponent. "
    "The debate topic  is: {motion}. \n"
    "You side is to {act} this topic .\n"
    "You should stand firm on your side ({act} the topic) and attack the opponent's weak points.\n"
    "\n"
)


default_rebuttal_prompt = context + (
        "Your response should be about {{n_words}} words and do not output other things than our response. When attacking, be aggresive and focus on a certain point that favors your side.\n"
    )

expert_rebuttal_prompt = context + (
        "\n## Knowledge\n"
        "### Structure of a Rebuttal\n"
        "A complete rebuttal should consist of multiple points, with each point containing four parts:\n"
        "- **Lead-in:** Introduce the opponent's argument, evidence, reasoning, etc., that you will be refuting.\n"
        "- **Explanation:** Briefly explain the opponent's argument to ensure clarity.\n"
        "- **Rebuttal:** This is the core of your point. Utilize rebuttal techniques to directly challenge the opponent's claim.\n"
        "- **Impact:** Concisely summarize the impact of your rebuttal and how it benefits your side.\n"
        "\n"
        "Note: Typically, the lead-in and explanation are combined into one sentence. The rebuttal is the most crucial part, and the impact summarizes its effect.\n"
        "\n"
        "### Rebuttal Techniques\n"
        "- **Pointing out logical fallacies:** Identify errors in the opponent's reasoning, such as reversing cause and effect, equivocation (shifting the meaning of a key term), straw man arguments, circular reasoning, or tautology (repeating the same idea in different words).\n"
        "- **Pointing out factual errors:** Highlight inaccuracies or weaknesses in the opponent's evidence, such as insufficient data, incorrect facts, or biased sources.\n"
        "   - **Example 1:** \"The evidence you provided is not enough to support your claim. For example, .... Therefore, your claim is not always correct.\"\n"
        "   - **Example 2:** \"The evidence you provided is from a biased source. For example, .... Therefore, your claim is not always correct.\"\n"
        "- **Pointing out error logic:** Identify flawed logic underlying opponent's framework.\n"
        "    - **Example 1:** \"You conclusion is based on the premise of A, but A is not always true. For example, .... Therefore, your conclusion is not always correct.\"\n"
        "    - **Example 2:** \"You conclusion relies on the premise of A and the hidden premise of B, but B is not always true. For example, .... Therefore, your conclusion is not always correct.\"\n"
        "    - **Example 3:** \"You claim A and B can lead to C, but that is not always the case. For example, .... Therefore, your conclusion is not always correct.\"\n"
        "- **Leveling the playing field:** This technique aims to neutralize the opponent's advantage or minimize the perceived harm of your side's position by demonstrating that both sides share the same issue or benefit.\n"
        "    - **Example 1:** \"You claim A, but B also has this problem. Therefore, both sides are equal in this regard, both having the issue.\"\n"
        "    - **Example 2:** \"You mention the benefits of A, but B also offers the same benefits. So, both sides are equal in this aspect, both being advantageous.\"\n"
        "- **Acknowledging and countering:** Start by acknowledging a valid point made by your opponent before explaining why your position still offers a better solution.\n"
        "    - **Example:** \"While I acknowledge that [opponent's valid point], this concern is outweighed by [your counter-argument] because [evidence/reasoning].\"\n"
        "\n"
    )

expert_rebuttal_prompt_2 = expert_rebuttal_prompt + (
    "## Workflow: Generate a spoken transcript for the rebuttal statement under the word budget ({{n_words}} words). \n"
    "1. Analyze the debate flow trees to select the battlefields you will discuss from the given list of battlefields. "
    "2. Allocate your word budget among the selected battlefields based on their importance and create the rebuttal plan. The plan should include the overview of battlefields you will discuss and the word budget for each battlefield. You should allocate more words to the more important points and can ignore the less important points. \n"
    "3. Follow the rebuttal plan to generate conversational arguments. Write as you would speak, with VARIED sentence lengths. Use short and simple words and sentences that are easy to understand.\n"
    "4. Deliver a rebuttal statement with {{n_words}} words in total. Present only the final text.\n"
    "Note that it's possible that the debate flow tree is not provided, in this case, you can just generate a rebuttal statement without following the debate flow tree.\n"
    "\n"

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

    "## Input Information\n"
    "Debate flow trees with node data:\n"
    "**Your Tree**: \n{tree}\n\n"
    "**Opponent's Tree**: \n{oppo_tree}\n\n"

    "## Battlefields\n"
    "{{tips}}\n\n"


    "## Important Notes\n"
    "1. Organize your points logically with clear purpose statements. \n"
        "   - Clearly mention the actions you will take in each point. For example, 'we will address [X] concerns mentioned by the opponent, which are ' for attack / rebut actions or 'after that,we will reinforce our claims that' for reinforce action.\n"
        "   - Use clear indicators like first, second, third, etc. to organize your points.\n"
        # "   - Among the selected battlefields, discuss the one with attack / rebut actions first. \n"
        # "   - If the opponent's argument is not clear, you can ask the opponent to clarify it first.\n"
    "2. Avoid restating or simply repeating the same evidence or arguments across different points.\n"
    "3. Avoid repeating the similar arguments in your previous statement. Use the phrase 'as we have discussed' to refer to the previous statement.\n"
    "4. Only use facts that are generally accepted and don't require specific citation. Don't hallucinate any particular experimental results, statistical findings from named studies, or quotes from specific researchers until the evidence pool is provided\n"
    "5. When presenting alternatives or counterarguments, offer specific, implementable solutions rather than just criticism.\n"
    "6. Address both systemic and individual factors when relevant, showing how they can complement rather than contradict each other.\n"
    "\n"

    "## Output with the format (two parts, start with **Rebuttal Plan** and then **Statement**):\n"
    "**Rebuttal Plan**: First, allocate words for the overview of the rebuttal. Then, allocate the rest of the word budget among the battlefields. Explain your rationale. Briefly mention one or two rhetorical techniques to use and logical fallacies to discuss. Make sure the total words is {{n_words}}.\n"
    "**Statement**: After the rebuttal plan, generate a rebuttal statement of {{n_words}} words in total, do not include any other text\n\n"

)
