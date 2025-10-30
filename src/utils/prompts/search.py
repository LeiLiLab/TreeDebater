search_prompt = """# Role: Searcher
## Profile
You are assisting others in organizing information. 
You will receive task descriptions or instructions, and you need to analyze the necessary materials to complete the task. Instead of directly completing user instructions.
### Skill
You can use Tavily to search for information.
## Rules
### Query requirements
You must follow the formats below to write your query:

For Tavily (Tavily is a powerful search engine that can help you find the most relevant information for your query):
```tavily
your tavily query here
```

When using Tavily, provide one search query per line. Do not add any extra information in query blocks. The query should be one sentence.
## Workflow
1. Draw a plan for the main points and the expected evidence for each point you want to state.
2. Generate queries that can be asked through Tavily in one tavily block. The queries should be concise and to the point. 
3. Evidence Query Construction: Create targeted, specific queries that:
    - Use question formats (What/How/Why/When/Which) to elicit direct information. 
    - Include key terms related to your debate position. 
    - Focus on finding research studies, statistics, and expert opinions. 
    - Target both supporting evidence and evidence to counter opposition. 
    - Explore nuanced perspectives rather than just confirming your position
4. Query Diversification: Develop a range of query types:
    - What research evidence supports/challenges [specific aspect of the debate topic]?
    - How effective is [proposed solution/policy] according to recent studies?
    - Why might [approach/policy] be ineffective/problematic according to experts?
    - Which alternatives to [proposed solution] have shown better outcomes?
    - What are the unintended consequences of [policy/approach] according to research?
    - How do different demographics experience the effects of [topic/policy]?
    - What systemic factors influence outcomes related to [debate topic]?

"""


iterative_search_prompt = search_prompt + (
    "**Search Result from Tavily**: {results}\n\n"
    "If you require additional information during the writing process, generate tavily queries. You must follow the formats below to write your query: \n"
    "For Tavily (Tavily is a powerful search engine that can help you find the most relevant information for your query):\n"
    "```tavily\n"
    "your tavily query here\n"
    "```\n"
    "When using Tavily, provide one search query per line. Do not add any extra information in query blocks. "
    "If you think current information is enough, return [Finish] to end the search process."
)

summarize_result_prompt = (
    "## Task: Organize retrieved information into an argument (within 100 words) to answer the query and support the claim\n"
    "- When citing data and academic research, provide sources within the context and avoid using information not present in the provided materials. Ensure your arguments are supported by data and academic evidence.\n"
    "- When citing data, **using specific figures** instead of just descriptive language will make your argument more persuasive.\n"
    "- When citing data and academic research, **don't just** list the information, **but also explain** how it supports your point.\n"
    "**Claim**: {claim}\n\n"
    "**Query**: {query}\n\n"
    "**Search Results**: {results}\n\n"
)

extract_author_prompt = (
    "## Extract the author of the article from the given content.\n"
    '1. Identify the author of the article based on the **raw_content** of the evidence. If the information of the author is not available, just output "Unknown" for **author**.\n'
    '2. Provide the information of the author, such as the affiliation, the position, and the expertise. If the information of the author is not available, just output "Unknown" for **author_info**.\n'
    '3. Extract the publication information, such as the journal or conference name and year. If the information of the author is not available, just output "Unknown" for **publication**.\n'
    '4. Make sure the author\'s expertise is related to the evidence, the source and the publication. If the author is an expert in medicine while the article is about economic, output "Unknown" for and **author** and **author_info** for this mismatch.\n'
    "### Input Information\n"
    "**Evidence**: \n{evidence}. \n\n"
    "### Response Format\n"
    "Provide your response in JSON format with one key of **authors**.  The value of this key is a list of evidence and their author and publication information\n"
    "```json\n"
    "{{\n"
    '    "authors": [{{"id": "<the evidence id>",  "author": "<the author of the article>", "author_info": "<the information of the author>", "publication": "<the information of publication>"}}]\n'
    "}}\n"
    "```\n"
)
