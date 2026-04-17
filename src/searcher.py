import argparse
import json
import os
import re
from functools import partial

from tavily import TavilyClient

from utils.db import get_cached_answer, save_query
from utils.model import HelperClient
from utils.prompts import iterative_search_prompt, search_prompt, summarize_result_prompt
from utils.timing_log import log_llm_io
from utils.tool import logger

MAX_QUERY = 10


def find_tavily(text):
    tavily_blocks = []
    start_idx = 0

    while True:
        lidx = text.find("```tavily", start_idx)
        if lidx == -1:
            break
        ridx = text.find("```", lidx + 9)
        if ridx == -1:
            break
        tavily_block = text[lidx + 9 : ridx].strip().split("\n")
        tavily_blocks.extend(tavily_block)
        start_idx = ridx + 3

    return tavily_blocks


def clean_raw_content(raw_content):
    if raw_content is None:
        return ""
    pattern = re.compile(r"  +")
    raw_content = raw_content.replace("\n", " ")
    raw_content = raw_content.replace("\\n", " ")
    raw_content = raw_content.replace("\t", " ")
    raw_content = pattern.sub(" ", raw_content)
    return raw_content


def get_search_result(tavily_client, query_list):
    response_list = []
    for query in query_list:
        logger.debug(f"[Search-Helper] Searching for {query}")
        cached_answer = get_cached_answer(query)
        if cached_answer is not None:
            logger.debug(f"[Search-Helper] Searching Hit in Cache")
            response_list.extend(json.loads(cached_answer[0]))
            continue

        max_retry = 3
        while max_retry > 0:
            try:
                response = tavily_client.search(
                    query=query,
                    search_depth="advanced",
                    max_results=5,
                    include_raw_content=True,
                    exclude_domains=["arxiv.org"],
                )
                break
            except Exception as e:
                max_retry -= 1
                logger.error(f"[Search-Helper] Error: {e}, Remaining Retries: {max_retry}")
        if max_retry == 0:
            logger.error(f"[Search-Helper] Failed to get search result for {query}")
            continue
        results = []

        for r in response["results"]:
            if len(r["content"].split()) < 10:
                continue
            ins = {
                "query": query,
                "title": r["title"],
                "url": r["url"],
                "content": r["content"],
                "score": r["score"],
                "raw_content": clean_raw_content(r["raw_content"]),
            }

            title = ""
            if "raw_content" in ins and ins["raw_content"] is not None:
                if "Skip to main content" in ins["raw_content"]:
                    title = ins["raw_content"].split("Skip to main content")[0]
            if len(ins["title"].split(" - ")) > 1:
                ins["source"] = ins["title"].split(" - ")[-1]
            else:
                ins["source"] = title.split(" - ")[-1]

            results.append(ins)
        save_query(query, json.dumps(results))
        response_list.extend(results)
    return response_list


def get_search_query(llm_client, motion, stance, claim=None, extra_prompt=None):
    prompt = search_prompt
    prompt += f"**Topic**: {motion}\n\n" f"**Stance**: {stance}\n\n"
    if claim is not None:
        prompt += "\n\n**Claim**: {claim}\n\n".format(claim=claim)
    if extra_prompt is not None:
        prompt += "\n\n" + extra_prompt
    log_llm_io(logger, phase="searcher", title="Search-Helper-Prompt", body=prompt.strip())
    response = llm_client(prompt=prompt)[0]
    log_llm_io(logger, phase="searcher", title="Search-Helper-Response", body=response.strip())
    queries = find_tavily(response)
    queries = [q.replace('"', "") for q in queries]

    log_llm_io(logger, phase="searcher", title="Search-Helper-Queries", body=" ||| ".join(queries))
    return queries


def update_search_query(llm_client, motion, stance, claim, results):
    simple_results = [
        {"query": r["query"], "title": r["title"], "url": r["url"], "content": r["content"]} for r in results
    ]
    prompt = iterative_search_prompt.format(
        motion=motion, stance=stance, claim=claim, results=json.dumps(simple_results, indent=2)
    )
    log_llm_io(logger, phase="searcher", title="Search-Helper-Update-Prompt", body=prompt.strip())
    response = llm_client(prompt=prompt)[0]
    log_llm_io(logger, phase="searcher", title="Search-Helper-Update-Response", body=response.strip())
    queries = find_tavily(response)
    log_llm_io(logger, phase="searcher", title="Search-Helper-Queries", body=" ||| ".join(queries))
    return queries


def summarize_search_result(llm_client, claim, search_results):
    for r in search_results:
        query = r["query"]
        content = {"title": r["title"], "url": r["url"], "content": r["content"]}
        prompt = summarize_result_prompt.format(claim=claim, query=query, results=json.dumps(content, indent=2))
        log_llm_io(logger, phase="searcher", title="Search-Summarize-Prompt", body=prompt.strip())
        response = llm_client(prompt=prompt)[0]
        log_llm_io(logger, phase="searcher", title="Search-Summarize-Response", body=response.strip())
        r["argument"] = response
    return search_results


def get_source_info(llm, evidence):
    for e in evidence:
        e["raw_content"] = " ".join(e["raw_content"].split()[:256])

    for e in evidence:
        reliability = 0
        if e["raw_content"] != "":
            reliability += 1
        if "source" in e and e["source"] != "":
            reliability += 1
        if "authors" in e and e["authors"] != "":
            reliability += 1
        if "author_info" in e and e["author_info"] != "":
            reliability += 1
        if "publication" in e and e["publication"] != "":
            reliability += 1
        if (
            "arxiv" in e.get("url", "").lower()
            or "arxiv" in e.get("source", "").lower()
            or "arxiv" in e.get("publication", "").lower()
        ):
            reliability = 0
        e["reliability"] = reliability

    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gemini-1.5-pro")
    parser.add_argument(
        "--claim",
        type=str,
        default="Fat taxes disproportionately burden low-income households, exacerbating existing inequalities.",
    )
    parser.add_argument("--motion", type=str, default="we should use fat tax")
    parser.add_argument("--stance", type=str, default="support")
    args = parser.parse_args()

    # Step 1. Instantiating your TavilyClient
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    llm_client = partial(HelperClient, model=args.model, temperature=1, max_tokens=2048)

    # Step 2. Get search queries
    search_queries = get_search_query(llm_client, motion=args.motion, stance=args.stance, claim=args.claim)
    if len(search_queries) > MAX_QUERY:
        search_queries = search_queries[:MAX_QUERY]
    search_results = get_search_result(tavily_client, search_queries)

    # Step 3. Update search queries
    new_search_queries = update_search_query(llm_client, args.claim, search_results)
    if len(new_search_queries) + len(search_queries) > MAX_QUERY:
        new_search_queries = new_search_queries[: MAX_QUERY - len(search_queries)]
    new_search_results = get_search_result(tavily_client, new_search_queries)

    # Step 4. Save the results
    all_results = search_results + new_search_results
    all_results = summarize_search_result(llm_client, args.claim, all_results)
    json.dump(all_results, open(f"{args.claim}_search_results.json", "w"), indent=2)
    print(f"Saved to {args.claim}_search_results.json")
