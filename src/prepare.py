import argparse
import json
import os
import re
import time
from functools import partial

import google.generativeai as genai
import numpy as np
from openai import OpenAI
from tavily import TavilyClient

from debate_tree import PrepareTree
from searcher import MAX_QUERY, get_search_query, get_search_result, get_source_info, update_search_query
from utils.constants import EMBEDDING_MODEL, google_api_key
from utils.claim_clustering import cluster_claim_embeddings, embedding_text
from utils.semantic_clustering import semantic_cluster_claims, materialize_groups
from utils.llm_schemas import ResultsResponse
from utils.model import HelperClient, reward_model
from utils.prompts import claim_propose_prompt, propose_definition_prompt
from utils.timing_log import log_llm_io, log_timing
from utils.tool import get_response_with_retry, logger

genai.configure(api_key=google_api_key)


class ClaimPool:
    def __init__(
        self,
        motion,
        side,
        model="gpt-4o",
        pool_size=50,
        max_search_depth=3,
        max_search_branch=3,
        use_rm_model=True,
        max_claim_groups=10,
        cluster_similarity_threshold=0.8,
        cluster_audit_file=None,
        clustering_method="semantic",
        **kwargs,
    ) -> None:
        self.motion = motion
        self.side = side
        self.pool_size = pool_size
        if isinstance(max_claim_groups, bool) or not isinstance(max_claim_groups, int) or max_claim_groups < 1:
            raise ValueError("max_claim_groups must be a positive integer")
        if not -1 <= cluster_similarity_threshold <= 1:
            raise ValueError("cluster_similarity_threshold must be between -1 and 1")
        self.max_claim_groups = max_claim_groups
        self.cluster_similarity_threshold = cluster_similarity_threshold
        self.cluster_audit_file = cluster_audit_file
        if clustering_method not in {"semantic", "agglomerative"}:
            raise ValueError("Unknown clustering_method")
        self.clustering_method = clustering_method
        self.act = "support" if side == "for" else "oppose"
        self.problem = f"The debate motion is: {self.motion}. You side is to {self.act} this motion."
        self.max_search_depth = max_search_depth
        self.max_search_branch = max_search_branch

        temperature = kwargs.get("temperature", 1)

        self.pool = []

        # if "moonshot" in model.lower() or "kimi" in model.lower():
        #     max_tokens_limit = 8192  # Conservative limit for moonshot models
        # else:
        #     max_tokens_limit = 8192
        # self.client = partial(HelperClient, model=model, temperature=temperature, max_tokens=max_tokens_limit, n=1)
        if "moonshot" in model.lower() or "kimi" in model.lower():
            max_tokens_limit = 4096  # Conservative limit for moonshot models (8192 total - 4096 response = 4096 input)
        else:
            max_tokens_limit = 8192
        self.client = partial(HelperClient, model=model, temperature=temperature, max_tokens=max_tokens_limit, n=1)
        if use_rm_model:
            self.reward_model = partial(reward_model, soft=True)
        else:
            self.reward_model = self.client
        self.tavily_client = TavilyClient(api_key=os.environ["TVLY_API_KEY"])

    def create_claim(self, need_score=True, need_evidence=True, max_search_depth=2, max_search_branch=3):
        prompt = propose_definition_prompt.format(motion=self.motion, act=self.act)
        log_llm_io(logger, phase="prepare", title="Definition-Helper-Prompt", body=prompt.strip(), side=self.side)
        response = self.client(prompt=prompt)[0]
        log_llm_io(logger, phase="prepare", title="Definition-Helper-Response", body=response.strip(), side=self.side)
        if "None" in response:
            self.definition = ""
        else:
            self.definition = response.replace("**Definition**: ", "").strip()

        prompt = claim_propose_prompt.format(motion=self.motion, act=self.act, size=self.pool_size)

        log_llm_io(logger, phase="prepare", title="Claim-Propose-Prompt", body=prompt, side=self.side)
        results, response = get_response_with_retry(
            self.client,
            prompt,
            "results",
            response_model=ResultsResponse,
            temperature=1.0,
        )
        log_llm_io(logger, phase="prepare", title="Claim-Propose-Response", body=json.dumps(response, indent=2), side=self.side)

        for item in results:
            strength = item.get("strength", 5)
            if strength < 6:
                continue
            new_claim = item["claim"]
            perspective = item["perspective"]
            explanation = item["explanation"]
            self.pool.append(
                {
                    "definition": self.definition,
                    "claim": new_claim,
                    "perspective": perspective,
                    "explanation": explanation,
                    "strength": strength,
                }
            )

        if self.clustering_method == "semantic":
            def save_audit(audit):
                self.clustering_audit = audit
                if self.cluster_audit_file:
                    with open(self.cluster_audit_file, "w") as output:
                        json.dump(audit, output, indent=2)
            groups, audit = semantic_cluster_claims(
                self.pool, self.client, self.motion, self.side,
                max_groups=self.max_claim_groups, audit_sink=save_audit,
            )
            self.grouped_pool = materialize_groups(self.pool, groups)
        else:
            clusters = self.cluster_claims(self.pool)
            self.grouped_pool = [[self.pool[i] for i in cluster] for cluster in clusters]
        self.grouped_pool = sorted(self.grouped_pool, key=lambda x: x[0]["strength"], reverse=True)

        if need_score:
            for group in self.grouped_pool:
                # TODO: only consider the first claim in each group
                for claim in group[:1]:
                    tree_structure, score = self.minimax_search(
                        claim["claim"],
                        motion=self.motion,
                        side=self.side,
                        root_argument=claim["explanation"],
                        max_depth=max_search_depth,
                        max_branch=max_search_branch,
                    )
                    claim["minimax_search_score"] = score
                    claim["tree_structure"] = tree_structure
                    logger.debug(f'[Minimax Score] {claim}: {claim["minimax_search_score"]}')

        if need_evidence:
            self.grouped_pool = self.get_evidence_pool()

        return self.grouped_pool

    def cluster_claims(self, pool):
        if not pool:
            return []
        texts = [embedding_text(claim) for claim in pool]
        resp = OpenAI().embeddings.create(model=EMBEDDING_MODEL, input=texts)
        items = sorted(resp.data, key=lambda item: item.index)
        if [item.index for item in items] != list(range(len(pool))):
            raise ValueError("Embedding response indices do not match claims")
        embeddings = [item.embedding for item in items]
        groups, audit = cluster_claim_embeddings(
            pool, embeddings, self.max_claim_groups, self.cluster_similarity_threshold
        )
        self.clustering_audit = dict(audit, claims=pool, embedding_inputs=texts,
                                    embeddings=embeddings, embedding_model=EMBEDDING_MODEL)
        if self.cluster_audit_file:
            with open(self.cluster_audit_file, "w") as output:
                json.dump(self.clustering_audit, output, indent=2)
        logger.info("Clustered %d claims into %d groups (hard cap=%d, forced merges=%d)",
                    len(pool), len(groups), self.max_claim_groups,
                    sum(m["forced_by_cap"] for m in audit["merges"]))
        return groups

    def minimax_search(self, root_claim, motion, side, root_argument=None, max_depth=2, max_branch=3):
        tree = PrepareTree(root_claim, motion, side, self.client, self.reward_model, root_argument)
        tree.expand_tree(tree.root, max_level=max_depth, max_branch=max_branch)
        tree.backward(level_decoy=0.8, support_weight=0.5)
        logger.debug(tree.print_tree(prefix="\n", include_status=True))
        tree_structure = tree.get_tree_info()

        best_path_idx, best_path, best_score = tree.root.get_minimax_score(
            max_depth=max_depth - 1, level_decoy=0.8, support_weight=0.5
        )
        best_path_str = "\n=> ".join([node.data for node in best_path])
        logger.debug(f"Path IDX: {best_path_idx}")
        logger.debug(f"Path: {best_path_str}")
        logger.debug(f"Score: {best_score}")
        return tree_structure, best_score

    def get_evidence_pool(self):
        for k, g in enumerate(self.grouped_pool):
            claim = g[0]["claim"]

            tree = PrepareTree.from_json(g[0]["tree_structure"])
            tree_info = tree.print_tree(include_status=False)
            tree_info = tree_info.replace("Level-0 Data", "Level-0 (Root Claim) Data")
            tree_info = tree_info.replace("Level-1 Data", "Level-1 (Opponent's Rebuttal) Data")
            tree_info = tree_info.replace("Level-2 Data", "Level-2 (Your Defense) Data")

            minimax_simulated_tree_feedback = (
                "### Simulated Back-and-Forth between You and Your Opponent\n"
                "Simulated back-and-forth between you and your opponent for your main claims. You can use this simulated debate flow tree structure to help evidence retrieval.\n"
                "Simulated Debate Flow Tree Structure for your main claim:\n"
                "* Level-0: The main claim \n"
                "* Level-1: Your opponent's rebuttal to the main claim\n"
                "* Level-2: Your defense against the opponent's rebuttal\n\n"
                "Prepare your search queries based on the simulated debate flow tree structure. "
                "The search queries should support the arguments in your root claim (Level-0) and defense (Level-2), while rebut the opponent's rebuttal (Level-1).\n"
                "**Simulated Debate Flow Tree Structure for your main claim**\n"
                f"{tree_info}\n\n"
            )

            search_queries = get_search_query(
                self.client, self.motion, self.act, claim, extra_prompt=minimax_simulated_tree_feedback
            )
            if len(search_queries) > MAX_QUERY:
                search_queries = search_queries[:MAX_QUERY]

            search_results = get_search_result(self.tavily_client, search_queries)

            # Step 3. Update search queries
            new_search_queries = update_search_query(self.client, self.motion, self.act, claim, search_results)
            if len(new_search_queries) + len(search_queries) > MAX_QUERY:
                new_search_queries = new_search_queries[: MAX_QUERY - len(search_queries)]
            new_search_results = get_search_result(self.tavily_client, new_search_queries)

            # Step 4. Save the results
            all_results = search_results + new_search_results
            all_results = [e for e in all_results if "raw_content" in e and e["raw_content"] != ""]
            # all_results = summarize_search_result(self.client, claim, all_results)

            for i, res in enumerate(all_results):
                res["id"] = f"{k}_{i}"

            all_results_with_source = get_source_info(self.client, all_results)
            if len(all_results_with_source) > 0:
                logger.debug(
                    f"[Evidence-Pool-Helper] Retrieved {len(all_results_with_source)} evidences for {claim} . Example e: {all_results_with_source[0]}"
                )
            else:
                logger.debug(f"[Evidence-Pool-Helper] No evidences retrieved for {claim}")

            # g[0]["retrieved_evidence"] = all_results_with_source
            g[0]["retrieved_evidence"] = all_results
        return self.grouped_pool


# python3 prepare.py
# python3 prepare.py --motion_file ../data/motion_list.txt --model gpt-4o
# python3 prepare.py --motion_file ../data/motion_list.txt --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
# python create_config.py --motion_file ../../dataset/motion_list.txt  --model deepseek-chat --save_dir test --pool_version 0515 --template base.yml
# python3 prepare.py --motion "It is time to welcome an A.I. Tutor in the classroom" --model moonshot-v1-128k --save_dir ../results1022 --max_search_depth 4 --ban_rm_model
# python3 prepare.py --motion "AI will lead to the decline of human creative arts" --model moonshot-v1-128k --save_dir ../results1022 --max_search_depth 4 --ban_rm_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", type=str, default=None)  # ../data/motion_list.txt
    parser.add_argument("--motion", type=str, default="Fast fashion (cheap, trend-driven clothing) should be banned")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    parser.add_argument("--no_evidence", action="store_true", default=False)
    parser.add_argument("--no_score", action="store_true", default=False)
    parser.add_argument("--pool_size", type=int, default=30)
    parser.add_argument("--save_dir", type=str, default="../results0315")
    parser.add_argument("--clustering_method", choices=["semantic", "agglomerative"], default="semantic")
    parser.add_argument("--max_claim_groups", type=int, default=10)
    parser.add_argument("--cluster_similarity_threshold", type=float, default=0.8)
    parser.add_argument("--max_n", type=int, default=-1)
    parser.add_argument("--max_search_depth", type=int, default=2)
    parser.add_argument("--max_search_branch", type=int, default=3)
    parser.add_argument("--ban_rm_model", action="store_true", default=False)
    args = parser.parse_args()
    if args.max_claim_groups < 1 or not -1 <= args.cluster_similarity_threshold <= 1:
        parser.error("max_claim_groups must be positive; cluster_similarity_threshold must be in [-1, 1]")

    if args.motion_file:
        with open(args.motion_file, "r") as f:
            motions = [x.strip() for x in f.readlines()]
            if args.max_n > 0:
                motions = motions[: args.max_n]
    else:
        motions = [args.motion.strip()]

    for motion in motions:
        model = args.model
        model_name = args.model.split("/")[-1]
        motion_name = motion.replace(" ", "_").lower()
        pool_size = args.pool_size

        save_dir = f"{args.save_dir}/{model_name}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for side in ["for", "against"]:
            save_file_name = f"{save_dir}/{motion_name}_pool_{side}.json"
            if os.path.exists(save_file_name):
                logger.info(f"Skip motion: {motion}")
            else:
                logger.info(f"Create motion for {save_file_name}...")
                t_prep = time.perf_counter()
                claim_workspace = ClaimPool(
                    motion=motion, side=side, model=model, pool_size=pool_size, use_rm_model=not args.ban_rm_model,
                    max_claim_groups=args.max_claim_groups,
                    clustering_method=args.clustering_method,
                    cluster_similarity_threshold=args.cluster_similarity_threshold,
                    cluster_audit_file=save_file_name.replace(".json", "_clustering.json"),
                )
                claim_pool = claim_workspace.create_claim(
                    need_score=not args.no_score,
                    need_evidence=not args.no_evidence,
                    max_search_depth=args.max_search_depth,
                    max_search_branch=args.max_search_branch,
                )
                log_timing(
                    logger,
                    "prepare_claim_pool_wall",
                    time.perf_counter() - t_prep,
                    motion=motion_name,
                    side=side,
                    n_claims=len(claim_pool),
                )
                logger.info(f"Claim Pool Size: {len(claim_pool)}")

                if len(claim_pool) > 0:
                    with open(save_file_name, "w") as file:
                        json.dump(claim_pool, file, indent=2)
                        logger.info(f"Saved to {save_file_name}")
