import json
import pandas as pd
from .tool import logger, get_response_with_retry
from .time_estimator import LengthEstimator
from .prompts import *
from .tool import identify_number_in_text, sort_by_action
from debate_tree import PrepareTree

##################### Evidence #####################


def select_query(llm, motion, stance, claim, action, candidate_queries):
    """
    Select the retrieval query for the claim
    return: query, a list of queries, ["query1", "query2", ...]
    """
    prompt = select_query_prompt.format(claim=claim, motion=motion, stance=stance, action=action, candidate_queries=candidate_queries)
    logger.debug("[Query-Helper-Prompt] " + prompt.strip().replace('\n',' ||| '))
    query, response = get_response_with_retry(llm, prompt, "query")
    logger.debug("[Query-Helper-Response] " + response.strip().replace('\n',' ||| '))
    return query

def rank_evidence(candidate_evidence, selected_queries=None):
    if selected_queries is not None:
        selected_evidence = [x for x in candidate_evidence if x["query"] in selected_queries]
    else:
        selected_evidence = candidate_evidence  

    titles, uniq_evidence = [], []
    for e in selected_evidence:
        # DO NOT use author name if author_info is empty
        if "author" in e and "author_info" in e:
            if e["author_info"] == "":
                e["author"] = ""
        e["numbers"] = identify_number_in_text(e.get("raw_content", e["content"]))
        e["n_numbers"] = len(e["numbers"])
        if e["title"] not in titles:
            uniq_evidence.append(e)
            titles.append(e["title"])

    #NOTE: the database can have some evidences
    for e in uniq_evidence:
        if "arxiv" in e.get("source","").lower() or "arxiv" in e.get("publication","").lower():
            e["reliability"] = -1
    
    sorted_evidence = [x for _, _, x in sorted(zip([x.get("n_numbers", 0) for x in uniq_evidence], [x.get("reliability", 0) for x in uniq_evidence], uniq_evidence), key=lambda pair: (pair[0], pair[1]), reverse=True)]

    return sorted_evidence


##################### Opening #####################

def build_cot_claims(llm, motion, side, claim_pool):
    # if "perspective" in claim_pool[0][0]:
    #     claims = [{"claim": x[0]["claim"], "perspective": x[0]["perspective"], "concept": x[0]["concept"], "explanation": x[0]["explanation"]} for x in claim_pool]
    # else:
    claims = [x[0]["claim"] for x in claim_pool]

    prompt = (
        "You are a debate assistant. You are given a motion and a side. You need to rank the following claims based on their relevance to the motion and side.\n\n"
        f"**Motion:** {motion}\n"
        f"**Side:** {side}\n"
        f"**Claims:** \n{claims}\n\n"
        "Please the best 3 claims that can be used in this debate from the list of claims. The selected claims should be exactly the same as the given list.\n"
        "These claims should be relevant to the motion and side, and should be from different perspectives. "
        "Use Json format with one key of **selection**. The value is a list of selected claims (string) that can be used in this debate.\n"
    )

    logger.debug("[CoT-Claims-Prompt] " + prompt.strip().replace('\n',' ||| '))
    selected_claims, response = get_response_with_retry(llm, prompt, "selection")
    logger.debug("[CoT-Claims-Response] " + response.strip().replace('\n',' ||| '))

    # selected_claims = [x if x.endswith(".") else x + '.' for x in selected_claims]
    claim_content = [x[0]["claim"] for x in claim_pool]
    print(claim_content)
    selected_idx = [claim_content.index(x) for x in selected_claims]

    thoughts = {
        "stage": "preparation",
        "side": side,
        "mode": "choose_main_claims",
        "all_claims": claims,
        "selected_claims": selected_claims,
        "selected_idx": selected_idx
    }

    return selected_claims, selected_idx, thoughts


def build_logic_claims(llm, motion, side, claim_pool, context="", definition="", use_rehearsal_tree=True, top_k=None):
    """
    Choose the main claims from the sorted claims based on the logic chain
    # """

    # Step 1. Sort claim groups by their highest minimax_search_score
    sorted_idx = sorted(range(len(claim_pool)), key=lambda i: claim_pool[i][0]["minimax_search_score"], reverse=True)    # sort the groups by the highest value
    
    # Step 2. Keep only top-k if specified
    if top_k is not None:
        sorted_idx = sorted_idx[:top_k]
    
    ranked_claims = [claim_pool[i][0]["claim"] for i in sorted_idx]

    # Step 3. Gather original claims & tree info
    tree_info = ""
    ori_claims = []
    if use_rehearsal_tree:
        for i in sorted_idx:
            data = claim_pool[i][0]["tree_structure"]
            tree = PrepareTree.from_json(data)
            tree_info += tree.print_tree(include_status=True) + "\n\n\n"
            ori_claims.append(claim_pool[i][0]["claim"])
    else:
        for i in sorted_idx:
            ori_claims.append(claim_pool[i][0]["claim"])
        tree_info = ""

    # Step 4. Construct prompt
    prompt = main_claim_selection.format(motion=motion, side=side, tree=tree_info, claims="\n".join(ori_claims), context=context, definition=definition)
    logger.debug("[Logic-Claims-Prompt] " + prompt.strip().replace('\n',' ||| '))
    content, response = get_response_with_retry(llm, prompt, "selection")
    logger.debug("[Logic-Claims-Response] " + response.strip().replace('\n',' ||| '))

    # Step 5. Parse model outputs
    selected_claims = content["claims"]
    framework = content["framework"]
    explanation = content["explanation"]

    selected_claims = [x if x.endswith(".") else x + '.' for x in selected_claims]
    selected_idx = [ori_claims.index(x) for x in selected_claims]
    
    # Step 6. Record reasoning info
    thoughts = {
        "stage": "preparation",
        "mode": "choose_main_claims",
        "ranked_claims": ranked_claims,
        "framework": framework,
        "explanation": explanation,
        "selected_claims": selected_claims,
        "selected_idx": selected_idx,
        "top_k": top_k,
    }

    return selected_claims, selected_idx, thoughts

##################### Debate Flow Tree #####################


def get_actions_from_tree(claims, tree, oppo_tree):
    actions = []

    if tree.max_level == 0:
        for claim in claims:
            actions.append({
                "idx": len(actions),
                "action": "propose",
                "target_claim": claim,
                "target_argument": "",
                "importance": "high",
                "targeted_debate_tree": "you"
            })
    else:
        for level in range(tree.max_level):
            nodes = tree.get_nodes_by_level(level + 1)
            if len(nodes) == 0:
                break
            else:
                action = "reinforce" if (level + 1) % 2 == 1 else "rebut"
                for node in nodes:
                    actions.append({
                        "idx": len(actions),
                        "action": action,
                        "target_claim": node.claim, 
                        "target_argument": "".join(node.argument),
                        "targeted_debate_tree": "you",
                    })
    
    if oppo_tree.max_level != 0:
        for level in range(oppo_tree.max_level):
            nodes = oppo_tree.get_nodes_by_level(level + 1)
            if len(nodes) == 0:
                break
            else:
                action = "attack" if (level + 1) % 2 == 1 else "reinforce"
                for node in nodes:
                    actions.append({
                        "idx": len(actions),
                        "action": action,
                        "target_claim": node.claim, 
                        "target_argument": "".join(node.argument),
                        "targeted_debate_tree": "opponent",
                    })
    
    logger.debug(f"[Debate-Flow-Tree-Action] {actions}")

    df = pd.DataFrame(actions)
    df = df.drop_duplicates(subset=["target_claim"])
    actions = df.to_dict(orient="records")

    return actions


def get_battlefields_from_actions(llm, motion, side, claims, actions, tree, oppo_tree):
    prompt = debate_flow_tree_action_eval_prompt.format(motion=motion, side=side, claims=claims, actions=json.dumps(actions, indent=2), tree=tree.print_tree(include_status=True), oppo_tree=oppo_tree.print_tree(include_status=True))
    logger.debug("[Debate-Flow-Tree-Action-Eval-Prompt] " + prompt.strip().replace('\n',' ||| '))
    eval_results, response = get_response_with_retry(llm, prompt, "response")
    logger.debug("[Debate-Flow-Tree-Action-Eval-Response] " + response.strip().replace('\n',' ||| '))

    battlefields = []
    for eval_result in eval_results:
        actions_in_battlefield = []
        for action in actions:
            if action["idx"] in eval_result["idx_list"]:
                actions_in_battlefield.append(action)
        actions_in_battlefield = sorted(actions_in_battlefield, key=lambda x: sort_by_action(x["action"]), reverse=True)
        battlefield = {
            "battlefield": eval_result["battlefield"],
            "battlefield_importance": eval_result["importance"],
            "battlefield_argument": eval_result["unified_argument"],
            "actions": actions_in_battlefield,
        }
        battlefields.append(battlefield)


    return battlefields


def get_retrieval_from_rehearsal_tree(action_type, target_claim, side, oppo_side, prepared_tree_list, prepared_oppo_tree_list, look_ahead_num, query_embedding):
    additional_info = []
    retrieval_nodes = []

    if prepared_tree_list is None:
        return additional_info, retrieval_nodes

    
    for tree in prepared_tree_list:
        if action_type == "propose" or action_type == "reinforce":
            if action_type == "propose":
                match_node = tree.get_node_by_claim(target_claim, side=side)
                similarity = 1.0 if match_node is not None else 0.0
            else:
                match_node, similarity = tree.get_most_similar_node(target_claim, query_embedding=query_embedding, side=side, top_k=1, threshold=0.8)
            
            if match_node is not None:
                logger.debug(f"[Prepared-Tree-Retrieval] {action_type} Hit: [{target_claim}] with [{match_node.claim}], Similarity: {similarity:0.2f}")
                score = match_node.get_strength(max_depth=look_ahead_num)
                match_node.argument = [match_node.argument] if isinstance(match_node.argument, str) else match_node.argument
                if len(match_node.argument) > 0:
                    node_info = " ".join(match_node.argument) + f"(Strength: {score:.1f})\n\t"
                else:
                    node_info = f"(Strength: {score:.1f})\n\t"
                additional_info.append(node_info)
                retrieval_nodes.append(["Prepared-Tree-Retrieval", action_type, target_claim, match_node.claim, similarity, node_info])
                if action_type == "propose":
                    break
        elif action_type == "attack" or action_type == "rebut":
            match_node, similarity = tree.get_most_similar_node(target_claim, query_embedding=query_embedding, side=oppo_side, top_k=1, threshold=0.8)
            if match_node is not None:
                logger.debug(f"[Prepared-Tree-Retrieval] {action_type} Hit: [{target_claim}] with [{match_node.claim}], Similarity: {similarity:0.2f}")
                node_info = ""
                for c in match_node.children:
                    score = c.get_strength(max_depth=look_ahead_num)
                    node_info += f"{c.claim} (Strength: {score:.1f})\n\t"
                additional_info.append(node_info)
                retrieval_nodes.append(["Prepared-Tree-Retrieval", action_type, target_claim, match_node.claim, similarity, node_info])
        else:
            raise ValueError(f"Invalid action: {action_type}")
    

    if additional_info == [] and match_node is None:
        logger.debug(f"[Prepared-Tree-Retrieval-Summary] {action_type} Miss. No additional info found for [{target_claim}]")
    else:
        logger.debug(f"[Prepared-Tree-Retrieval-Summary] {action_type} Hit. Additional info: {additional_info}")

    additional_info_from_oppo_tree = []
    if prepared_oppo_tree_list is not None:
        for tree in prepared_oppo_tree_list:
            if action_type == "attack" or action_type == "rebut":
                match_node, similarity = tree.get_most_similar_node(target_claim, query_embedding=query_embedding, side=oppo_side, top_k=1, threshold=0.8)
                if match_node is not None:
                    logger.debug(f"[Prepared-Opponent-Tree-Retrieval] {action_type} Hit: [{target_claim}] with [{match_node.claim}], Similarity: {similarity:0.2f}")
                    node_info = ""
                    for c in match_node.children:
                        score = c.get_strength(max_depth=look_ahead_num)
                        node_info += f"{c.claim} (Strength: {score:.1f})\n\t"
                    additional_info_from_oppo_tree.append(node_info)
                    retrieval_nodes.append(["Prepared-Opponent-Tree-Retrieval", action_type, target_claim, match_node.claim, similarity, node_info])
            elif action_type == "propose" or action_type == "reinforce":
                match_node, similarity = tree.get_most_similar_node(target_claim, query_embedding=query_embedding, side=side, top_k=1, threshold=0.8)
                if match_node is not None:
                    logger.debug(f"[Prepared-Opponent-Tree-Retrieval] {action_type} Hit: [{target_claim}] with [{match_node.claim}], Similarity: {similarity:0.2f}")
                    score = match_node.get_strength(max_depth=look_ahead_num)
                    match_node.argument = [match_node.argument] if isinstance(match_node.argument, str) else match_node.argument
                    if len(match_node.argument) > 0:
                        node_info = " ".join(match_node.argument) + f"(Strength: {score:.1f})\n\t"
                    else:
                        node_info = f"(Strength: {score:.1f})\n\t"
                    additional_info_from_oppo_tree.append(node_info)
                    retrieval_nodes.append(["Prepared-Opponent-Tree-Retrieval", action_type, target_claim, match_node.claim, similarity, node_info])
            else:
                raise ValueError(f"Invalid action: {action_type}")
            
    if additional_info_from_oppo_tree == [] and match_node is None:
        logger.debug(f"[Prepared-Opponent-Tree-Retrieval-Summary] {action_type} Miss. No additional info found for [{target_claim}]")
    else:
        logger.debug(f"[Prepared-Opponent-Tree-Retrieval-Summary] {action_type} Hit. Additional info: {additional_info_from_oppo_tree}")

    additional_info = additional_info + additional_info_from_oppo_tree
    return additional_info, retrieval_nodes


##################### Time-Adjuster #####################

class TimeAdjuster:
    def __init__(self):
        self.L = None
        self.R = None

    def revise_helper(self, statement, n_words, budget, threshold=5, ratio=0.46, estimator=None):
        current_cost = estimator.query_time(statement)
        words_count = LengthEstimator(mode="words").query_time(statement)
        logger.debug("[Efficient-Fit-Length] " + f"use {n_words} words in the prompt, real words: {words_count}, real cost: {current_cost:0.2f}, target interval: [{budget-threshold}, {budget}]")

        if budget <= 0 or current_cost >= budget-threshold and current_cost <= budget + 1:
            return current_cost, n_words, True

        # step1. determine the first endpoint
        if self.L is None and self.R is None:
            if current_cost < budget-threshold:
                self.L = n_words
                return current_cost, n_words + int((budget - current_cost) / ratio), False
            else:
                self.R = n_words
                return current_cost, n_words - int((current_cost - (budget-threshold)) / ratio), False

        # step2. determine the second endpoint
        if self.L is None:
            if current_cost < budget-threshold:
                self.L = n_words
                return current_cost, (self.L + self.R) // 2, False
            else:
                return current_cost, n_words - int((current_cost - (budget-threshold)) / ratio), False
        if self.R is None:
            if current_cost > budget:
                self.R = n_words
                return current_cost, (self.L + self.R) // 2, False
            else:
                return current_cost, n_words + int((budget - current_cost) / ratio), False

        # step3. binary search in [L, R], always terminate and w.h.p. can terminate when R-L > 1
        if current_cost < budget-threshold:
            self.L = n_words
        else:
            self.R = n_words
        return current_cost, (self.L + self.R) // 2, False
    

##################### Anaylsis #####################

def extract_statement(llm, motion, statement, claims=None, tree=None, side=None, stage=None):
    if claims is not None:
        prompt = extract_statment_by_claim_prompt.format(motion=motion, statement=statement, claim=json.dumps(claims))
    elif tree is not None:
        prompt = extract_statment_with_tree_prompt.format(motion=motion, statement=statement, claim=json.dumps(claims), tree=tree[0], oppo_tree=tree[1], side=side, stage=stage)
    else:
        prompt = extract_statment_prompt.format(motion=motion, statement=statement)
        
        
    logger.debug("[Analyze-Helper-Prompt] " + prompt.strip().replace('\n',' ||| '))
    claims, response = get_response_with_retry(llm, prompt, "statements")
    logger.debug("[Analyze-Helper-Response] " + response.strip().replace('\n',' ||| '))
    return claims

