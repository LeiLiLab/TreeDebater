import argparse
import json
import os
import sys
import time
from functools import partial
from typing import List

import google.generativeai as genai
import nltk
import numpy as np
import pandas as pd
import torch
from sentence_transformers.util import dot_score, semantic_search

from evaluator import evaluate_defense_strength, evaluate_support_strength
from utils.constants import get_embeddings
from utils.model import HelperClient, reward_model
from utils.timing_log import log_llm_io
from utils.tool import get_response_with_retry, logger


def propose_new_claims(proposer, motion, side, history, n):
    # generate the next layer of rebuttal claims
    history_str = ""
    for i, x in enumerate(history):
        if (i % 2) != (len(history) % 2):
            history_str += "**Opponent**: " + x + "\n"
        else:
            history_str += "**You**: " + x + "\n"

    act = "support" if side == "for" else "oppose"
    num = n

    prompt = (
        "## Task: Generate Strategic Counter-Arguments\n"
        f"You are participating in a formal debate on the motion: {motion}\n"
        f"Your position: {act} the motion\n\n"
        "## Your Objective\n"
        f"Generate {num} persuasive counter-arguments that:\n"
        "1. Focus on countering the opponent's last argument\n"
        "2. Directly challenge and weaken your opponent's position\n"
        "3. Strengthen your stance on the motion\n"
        "4. Cover distinct aspects without overlap\n"
        "5. Maintain consistency with your previous statements\n\n"
        "### Counter-Argument Techniques\n"
        "- **Logical fallacies:** Identify errors in reasoning (cause-effect reversal, equivocation, straw man arguments, circular reasoning, tautology)\n"
        '    - **Example:** "You\'ve committed a false dilemma fallacy by suggesting only two possible outcomes when multiple alternatives exist."\n'
        "- **Factual errors:** Highlight inaccuracies in the opponent's factual statements\n"
        '    - **Example:** "Your argument relies on statistics from 2010, but more recent data from 2023 shows the opposite trend."\n'
        "- **Logic errors:** Identify flawed underlying logic\n"
        '    - **Example 1:** "Your conclusion is based on premise A, but A is not always true. For example, ... Therefore, your conclusion is flawed."\n'
        '    - **Example 2:** "Your conclusion relies on premises A and B, but B is not always true. For example, ... Therefore, your conclusion is not always correct."\n'
        '    - **Example 3:** "You claim A and B lead to C, but that is not always the case. For example, ... Therefore, your conclusion is questionable."\n'
        "- **Leveling the playing field:** Neutralize the opponent's advantage by showing both sides share the same issue or benefit\n"
        '    - **Example 1:** "You claim A, but B also has this problem. Therefore, both sides are equal in this regard."\n'
        '    - **Example 2:** "You mention the benefits of A, but B offers the same benefits. So, both sides are equally advantageous."\n'
        "## Context\n"
        "Previous debate exchanges:\n"
        f"{history_str}\n\n"
        "## Response Format\n"
        "Provide your response in JSON format:\n"
        "{{\n"
        '    "statements": [\n'
        "        {{\n"
        '            "claim": "Your counter-claim statement",\n'
        '            "argument": "A 100-word explanation of how this claim: 1. Undermines your opponent\'s last argument, 2. Reinforces your previous argument or stance"\n'
        "        }},\n"
        "        ...\n"
        "    ]\n"
        "}}\n"
    )
    log_llm_io(logger, phase="debate_tree", title="Proposer-Tree-Helper-Prompt", body=prompt.strip(), side=side)
    content, response = get_response_with_retry(proposer, prompt, "statements", temperature=1)
    log_llm_io(logger, phase="debate_tree", title="Proposer-Tree-Helper-Response", body=response.strip(), side=side)
    return content


def update_eval_score(node, scorer):
    if node.children:
        for child in node.children:
            update_eval_score(child, scorer)
    node.eval_score(scorer)


class Node:
    # claim and argument
    def __init__(self, motion: str, side: str, parent=None):

        self.motion = motion
        self.side = side
        assert self.side == "for" or self.side == "against"
        self.claim = ""
        self.argument = []
        self.evidence = []
        self.parent = parent
        self.children = []
        self.scores = None  # {"defense": x, "support": y}
        if parent is not None:
            self.level = parent.level + 1
        else:
            self.level = 0

        self.status = "prepared"
        self.visit_count = 0

    def eval_score(self, scorer):
        history = []
        cur_node = self
        while cur_node:
            history.append(cur_node.data)
            cur_node = cur_node.parent
        history = history[::-1]  # NOTE: history is from root to this node

        defense, support = 0, 0
        if self.parent:
            defense = evaluate_defense_strength(scorer, self.motion, self.data, self.parent.data, history=history)
            if self.parent.parent:
                support = evaluate_support_strength(
                    scorer, self.motion, self.data, self.parent.parent.data, history=history
                )
        else:
            if self.side == "for":
                support = evaluate_support_strength(
                    scorer, self.motion, self.data, self.motion, history=[self.motion, "", self.data]
                )
            else:
                support = evaluate_defense_strength(
                    scorer, self.motion, self.data, self.motion, history=[self.motion, self.data]
                )
        self.scores = {"defense": defense, "support": support}

    def get_minimax_score(self, max_depth=2, level_decoy=0.8, support_weight=0.5, root_type="support"):
        """
        Get the worst case for player 1 in a two-player game
        return the worst path idx, the worst path node, the score
        the utility is the score of player 1
        """
        root_score = self.scores[root_type]
        # player 1, utility = 0
        if max_depth == 0:
            return 0, self, root_score
        # player 1 -> player 2
        elif max_depth == 1:
            # player 2 use the best defense which minimize player 1's utility (-defense)
            child_scores = [-child.scores["defense"] for child in self.children]
            idx = np.argmin(child_scores)
            return [idx.item()], [self.children[idx]], root_score + level_decoy * child_scores[idx]
        # player 1 -> player 2 -> player 1
        elif max_depth == 2:
            child_utility = []
            chosen_child = []
            chosen_child_idx = []
            # self is player 1, child is player 2, child.children is player 1
            for child in self.children:
                # player 1 choose the best rebuttal to player 2, maximize player 1's utility (defense + support)
                child_scores = [
                    (1 - support_weight) * c.scores["defense"] + support_weight * c.scores["support"]
                    for c in child.children
                ]
                best_idx = np.argmax(child_scores).item()
                best_child = child.children[best_idx]

                # player 2's utility (towards player 1)
                utility = -child.scores["defense"] + level_decoy * child_scores[best_idx]
                chosen_child_idx.append(best_idx)
                chosen_child.append(best_child)
                child_utility.append(utility)
            # player 2 minimize player 1's utility
            idx = np.argmin(child_utility).item()
            path_idx = [idx, chosen_child_idx[idx]]
            path = [self.children[idx], chosen_child[idx]]
            return path_idx, path, root_score + level_decoy * child_utility[idx]
        # player 1 -> player 2 -> player 1 -> player 2
        elif max_depth == 3:
            child_utility = []
            chosen_child = []
            chosen_child_idx = []
            # self is player 1, child is player 2
            for child in self.children:
                # child_score is the score towards player 2
                child_score_idx, child_path, child_score = child.get_minimax_score(
                    max_depth=max_depth - 1, level_decoy=level_decoy, support_weight=support_weight
                )
                child_utility.append(-child_score)
                chosen_child_idx.append(child_score_idx)
                chosen_child.append(child_path)
            # player 2 minimize player 1's utility
            idx = np.argmin(child_utility).item()
            path_idx = [idx] + chosen_child_idx[idx]
            path = [self.children[idx]] + chosen_child[idx]
            return path_idx, path, root_score + level_decoy * child_utility[idx]
        else:
            raise ValueError("Currently only support depth 3")

    def get_strength(self, max_depth=1, level_decoy=0.8, support_weight=0.5):
        strength = 0
        if self.scores["defense"] != 0 and self.scores["support"] != 0:
            strength = support_weight * self.scores["defense"] + (1 - support_weight) * self.scores["support"]
        elif self.scores["defense"] != 0:
            strength = self.scores["defense"]
        elif self.scores["support"] != 0:
            strength = self.scores["support"]

        if max_depth == 0 or self.is_terminal():
            return strength
        else:
            children_strength = [
                c.get_strength(max_depth=max_depth - 1, level_decoy=level_decoy, support_weight=support_weight)
                for c in self.children
            ]
            return strength - level_decoy * max(children_strength)

    def add_node(self, data=None, new_claim=None, new_argument=None, side=None):
        if side is None:
            new_side = "against" if self.side == "for" else "for"
        else:
            new_side = side
        new_node = Node(self.motion, new_side, parent=self)

        if (new_claim is None) or (new_argument is None):
            if isinstance(data, dict):
                new_claim, new_argument = data.get("claim", ""), data.get("argument", [])
            else:
                if len(data.split("\t")) > 1:
                    new_claim, *new_argument = data.split("\t")
                elif len(data.split("[SEP]")) > 1:
                    new_claim, *new_argument = data.split("[SEP]")

        new_node.claim = new_claim
        new_node.argument = new_argument
        if isinstance(new_node.argument, str):
            new_node.argument = [new_node.argument]
        self.children.append(new_node)
        return new_node

    def expand(self, proposer, scorer, branch=3):
        history = []
        cur_node = self
        while cur_node:
            history.append(cur_node.data)
            cur_node = cur_node.parent
        history = history[::-1]  # NOTE: history is from root to this node

        new_data = propose_new_claims(
            proposer, self.motion, "against" if self.side == "for" else "for", history, n=branch
        )
        for data in new_data:
            child_node = self.add_node(data)
            child_node.eval_score(scorer)

    def get_node_info(self):
        info = {
            "side": self.side,
            "level": self.level,
            "claim": self.claim,
            "argument": self.argument,
            "evidence": self.evidence,
            "status": self.status,
            "visit_count": self.visit_count,
            "scores": self.scores,
            "children": [],
        }
        for child in self.children:
            child_info = child.get_node_info()
            info["children"].append(child_info)
        return info

    def update_status(self, status, keep_visit=False):
        self.status = status
        if status != "waiting" or not keep_visit:
            self.visit_count += 1

    def is_terminal(self):
        return len(self.children) == 0

    def update_evidence(self, new_evidence):
        if isinstance(new_evidence, list):
            self.evidence.extend(new_evidence)
        else:
            self.evidence.append(new_evidence)

    @property
    def data(self):
        info = {
            "claim": self.claim,
            "argument": self.argument,
        }
        return json.dumps(info)

    @property
    def statement(self):
        info = {
            "claim": self.claim,
            "argument": self.argument,
        }
        return json.dumps(info)

    @staticmethod
    def from_json(motion, side, parent, json_info):
        node = Node(motion, side, parent)
        node.claim = json_info["claim"]
        node.argument = json_info["argument"]
        node.evidence = json_info["evidence"]
        node.status = json_info["status"]
        node.visit_count = json_info["visit_count"]
        node.scores = json_info["scores"]
        for child_info in json_info["children"]:
            child_node = Node.from_json(motion, "against" if side == "for" else "for", node, child_info)
            node.children.append(child_node)
        return node


class Tree:
    def __init__(self, motion, side):
        self.motion = motion
        self.side = side
        self.level = 0
        self.embedding_cache = {}

    def get_all_nodes(self):
        return self.get_all_nodes_recursive(self.root)

    def get_all_nodes_recursive(self, node):
        all_nodes = []
        if node is None:
            return []
        all_nodes.append(node)
        for child in node.children:
            all_nodes.extend(self.get_all_nodes_recursive(child))
        return all_nodes

    def get_nodes_by_level(self, level):
        nodes = []
        for node in self.get_all_nodes():
            if node.level == level:
                nodes.append(node)
        return nodes

    def get_node_by_side(self, side):
        if side is None:
            side = self.side
        return self.get_node_by_side_recursive(self.root, side)

    def get_node_by_side_recursive(self, node, side):
        side_nodes = []
        if node is None:
            return []
        if node.side == side and node.status != "root":
            side_nodes.append(node)
        for child in node.children:
            side_nodes.extend(self.get_node_by_side_recursive(child, side))
        return side_nodes

    def get_node_by_claim(self, claim, side=None):
        if claim is None:
            return None
        return self.get_node_by_claim_recursive(self.root, claim, side=side)

    def get_node_by_claim_recursive(self, node, claim, side=None):
        if node.claim == claim and (side is None or node.side == side):
            return node
        for child in node.children:
            match = self.get_node_by_claim_recursive(child, claim, side=side)
            if match:
                return match
        return None

    def get_node_by_status(self, status, side=None):
        if status is None:
            return None
        if isinstance(status, list):
            status_nodes = []
            for s in status:
                status_nodes.extend(self.get_node_by_status_recursive(self.root, s, side=side))
            return status_nodes
        return self.get_node_by_status_recursive(self.root, status, side=side)

    def get_node_by_status_recursive(self, node, status, side=None):
        status_nodes = []
        if node is None:
            return []
        if node.status == status and (side is None or node.side == side):
            status_nodes.append(node)
        for child in node.children:
            status_nodes.extend(self.search_status(child, status, side=side))
        return status_nodes

    def print_tree_recursive(self, node, level=0, lines=None, include_status=False, max_print_level=None):
        if lines is None:
            lines = []
        if node is not None:
            if max_print_level is not None and level > max_print_level:
                return lines
            score_parts = []
            if node.scores is not None:
                for k, v in node.scores.items():
                    if k == "defense" and v != 0:
                        score_parts.append(f"Attack Score: {v:.1f}")
                    elif k == "support" and v != 0:
                        score_parts.append(f"Support Score: {v:.1f}")
            score_str = ", ".join(score_parts)
            lines.append(
                ' ' * level * 4
                + f"Level-{level} Data (Visit: {node.visit_count}, Status: {node.status}): {node.data}, Scores: {score_str}\n"
            )
            for child in node.children:
                self.print_tree_recursive(child, level + 1, lines=lines, include_status=False, max_print_level=max_print_level)
        return lines

    def print_tree(self, prefix="", include_status=False, max_print_level=None):
        lines = [prefix] if prefix else []
        self.print_tree_recursive(self.root, level=0, lines=lines, include_status=include_status, max_print_level=max_print_level)
        return "".join(lines)


    def get_tree_info(self):
        info = {
            "motion": self.motion,
            "root": self.root.data,
            "side": self.side,
            "structure": self.root.get_node_info(),
        }
        return info

    @staticmethod
    def from_json(json_info):
        motion = json_info["motion"]
        side = json_info["side"]
        root_info = json_info["structure"]
        tree = Tree(motion, side)
        tree.root = Node.from_json(motion, side, None, root_info)
        return tree

    @property
    def max_level(self):
        max_level = 0
        for node in self.get_all_nodes():
            if node.level > max_level:
                max_level = node.level
        return max_level

    def get_embedding_from_cache(self, contents: List[str]):
        if isinstance(contents, str):
            contents = [contents]

        embeddings = [None] * len(contents)
        new_content_idx = []
        for idx, content in enumerate(contents):
            if content in self.embedding_cache:
                embeddings[idx] = self.embedding_cache[content]
            else:
                new_content_idx.append(idx)

        if len(new_content_idx) == 0:
            return embeddings

        new_contents = [contents[i] for i in new_content_idx]
        max_retry = 3
        retry = 0
        while retry < max_retry:
            try:
                new_embeddings = get_embeddings(new_contents)
                break
            except Exception as e:
                logger.error(f"[Get-Embedding-From-Cache] Error: {e}. Sleep 30 seconds and retry.")
                time.sleep(30)
                retry += 1

        for idx, embedding in zip(new_content_idx, new_embeddings):
            embeddings[idx] = embedding
            self.embedding_cache[contents[idx]] = embedding

        return embeddings

    def get_most_similar_node(self, query, query_embedding=None, side=None, level=None, top_k=1, threshold=0.5):
        """Returns the most similar node and its similarity score given a query.

        Args:
            query (str): The query text to compare against
            side (str, optional): Filter nodes by side ("for" or "against"). If None, search all nodes.

        Returns:
            tuple: (most_similar_node, similarity_score)
        """

        nodes = self.get_node_by_side(side) if side else self.get_all_nodes()
        if level is not None:
            nodes = [node for node in nodes if node.level == level]
        nodes = [node for node in nodes if node.claim != ""]
        node_embedding = self.get_embedding_from_cache([node.claim for node in nodes])

        if not nodes:
            return None, 0.0

        query_embedding = self.get_embedding_from_cache(query) if query_embedding is None else query_embedding
        hits = semantic_search(
            torch.tensor(query_embedding), torch.tensor(node_embedding), score_function=dot_score, top_k=top_k
        )[0]
        retrieval_idx = [x["corpus_id"] for x in hits if x["score"] >= threshold]
        retrieval_node = [nodes[idx] for idx in retrieval_idx]
        retrieval_similarity = [x["score"] for x in hits]

        if len(retrieval_idx) == 0:
            highest_score = hits[0]["score"]
            highest_score_idx = hits[0]["corpus_id"]
            highest_score_claim = nodes[highest_score_idx].claim
            logger.warning(
                f"No retrieval node for query: {query} (threshold: {threshold}). The highest score is {highest_score} for [{highest_score_claim}]."
            )
            return None, 0.0
        elif len(retrieval_idx) == 1:
            return retrieval_node[0], retrieval_similarity[0]
        else:
            return retrieval_node, retrieval_similarity


class PrepareTree(Tree):
    def __init__(self, root_data, motion, side, proposer, scorer, root_argument=None):
        super().__init__(motion, side)
        self.root = Node(motion, side)
        self.root.claim = root_data
        self.proposer = proposer
        self.scorer = scorer
        if root_argument is not None:
            self.root.argument = root_argument

    def expand_tree(self, node, max_level=3, max_branch=3):
        node_list = [node]
        cur_level = 0
        node.eval_score(self.scorer)
        while cur_level < max_level - 1:
            num_node = len(node_list)
            for _ in range(num_node):
                node = node_list.pop(0)
                node.expand(self.proposer, self.scorer, branch=max_branch)
                node_list.extend(node.children)
            cur_level += 1

    def print_tree_recursive(self, node, level=0, lines=None, include_status=False, max_print_level=None):
        if lines is None:
            lines = []
        if level == 0:
            position = "Root Claim"
        elif level % 2 == 1:
            position = "Opponent's Attack"
        else:
            position = "Your Rebuttal"

        if node is not None:
            if max_print_level is not None and level > max_print_level:
                return lines
            
            score_parts = []
            if node.scores is not None:
                for k, v in node.scores.items():
                    if k == "defense" and v != 0:
                        score_parts.append(f"Attack Score: {v:0.1f}")
                    elif k == "support" and v != 0:
                        score_parts.append(f"Support Score: {v:0.1f}")
                    elif k == "minimax_strength" and v != 0:
                        score_parts.append(f"Strength: {v:0.1f}")
                score_str = ", ".join(score_parts)
                    
            if include_status:
                lines.append(' ' * level * 4 + f"Level-{level} {position}: {node.data}, Scores: {score_str}\n")
            else:
                lines.append(' ' * level * 4 + f"Level-{level} {position}: {node.data}\n")
            for child in node.children:
                self.print_tree_recursive(child, level + 1, lines=lines, include_status=include_status, max_print_level=max_print_level)
        return lines

    def backward(self, level_decoy=0.8, support_weight=0.5):
        self.backward_recursive(self.root, level_decoy, support_weight)

    def backward_recursive(self, node, level_decoy=0.8, support_weight=0.5):
        if node.scores["support"] != 0 and node.scores["defense"] != 0:
            strength = support_weight * node.scores["support"] + (1 - support_weight) * node.scores["defense"]
        elif node.scores["support"] != 0:
            strength = node.scores["support"]
        elif node.scores["defense"] != 0:
            strength = node.scores["defense"]
        else:
            strength = 0

        if node.is_terminal():
            node.scores["minimax_strength"] = strength
        else:
            for child in node.children:
                self.backward_recursive(child, level_decoy, support_weight)
            child_minimax_strength = [child.scores["minimax_strength"] for child in node.children]
            node.scores["minimax_strength"] = strength - level_decoy * max(child_minimax_strength)
            logger.debug(
                f"Child Minimax Strength: {child_minimax_strength}, Max: {max(child_minimax_strength)}, Strength: {strength}, Level Decoy: {level_decoy}, Minimax Strength: {node.scores['minimax_strength']}"
            )

    @staticmethod
    def from_json(json_info):
        motion = json_info["motion"]
        side = json_info["side"]
        root_json_info = json_info["structure"]
        root_data = ""
        tree = PrepareTree(root_data, motion, side, proposer=None, scorer=None)
        tree.root = Node.from_json(motion, side, None, root_json_info)
        return tree

    @property
    def max_level(self):
        max_level = 0
        for node in self.get_all_nodes():
            if node.level > max_level:
                max_level = node.level
        return max_level


class DebateTree(Tree):
    def __init__(self, motion, side):
        super().__init__(motion, side)
        self.root = Node(motion, side)
        self.root.status = "root"
        self.meta_attack_list = []
        self.meta_rebuttal_list = []

    def print_tree_recursive(self, node, level=0, lines=None, include_status=False, max_print_level=None, reverse=False):
        if lines is None:
            lines = []
        if level == 0:
            position = "Motion"
        elif level == 1:
            position = "Your Main Claim" if not reverse else "Opponent's Main Claim"
        elif level % 2 == 0:
            position = "Opponent's Attack" if not reverse else "Your Attack"
        else:
            position = "Your Rebuttal" if not reverse else "Opponent's Rebuttal"

        if node is not None:
            if max_print_level is not None and level > max_print_level:
                return lines
            if level == 0:
                lines.append(' ' * level * 4 + f"Level-{level} Motion: {self.motion}, Side: {self.side}\n")
            else:
                if include_status:
                    lines.append(' ' * level * 4 + f"Level-{level} {position} (Visit: {node.visit_count}, Status: {node.status}): {node.data}\n")
                else:
                    lines.append(' ' * level * 4 + f"Level-{level} {position}: {node.data}\n")
            for child in node.children:
                self.print_tree_recursive(child, level + 1, lines=lines, include_status=include_status, max_print_level=max_print_level, reverse=reverse)
        return lines

    def print_tree(self, prefix="", include_status=False, max_print_level=None, meta_info=True, reverse=False):
        lines = [prefix] if prefix else []
        self.print_tree_recursive(
            self.root,
            level=0,
            lines=lines,
            include_status=include_status,
            max_print_level=max_print_level,
            reverse=reverse,
        )
        if meta_info:
            if len(self.meta_attack_list) > 0:
                lines.append(f"Meta Attack to this debate tree: {self.meta_attack_list}\n")
            if len(self.meta_rebuttal_list) > 0:
                lines.append(f"Meta Rebuttal to the attacks on this debate tree: {self.meta_rebuttal_list}")
        return "".join(lines)

    def update_node(self, action, new_claim=None, new_argument=None, target=None):
        if len(new_claim) == 0:
            logger.warning(f"Empty claim: [{new_claim}]")
            return
        if target is None:
            logger.warning(f"Target is None for the action: {action} and claim: {new_claim}, skip")
            return
        if target == "N/A":
            if action == "attack":
                self.meta_attack_list.append(new_claim)
            elif action == "rebut":
                self.meta_rebuttal_list.append(new_claim)

            return

        if action == "propose":
            new_node = self.root.add_node(new_claim=new_claim, new_argument=new_argument, side=self.root.side)
            new_node.update_status("proposed")
            return

        if action == "rebut":
            target_node_side = "against" if self.root.side == "for" else "for"
        else:  # propose or reinforce or attack, the target is the same side
            target_node_side = self.root.side

        match_node = self.get_node_by_claim(target, side=target_node_side)
        if match_node is None:
            match_node, similarity = self.get_most_similar_node(target, side=target_node_side, top_k=1, threshold=0.8)

        # try to find the rebut node in the same side
        if match_node is None and action == "rebut":
            logger.info(
                f"Cannot find the matched node for: action: {action}, target: {target}, try to find the reinforce node in the same side"
            )
            action = "reinforce"
            match_node = self.get_node_by_claim(target, side=self.root.side)
            if match_node is None:
                match_node, similarity = self.get_most_similar_node(target, side=self.root.side, top_k=1, threshold=0.8)

        if match_node is None:
            logger.warning(f"Cannot find the matched node for action: {action}, target: {target}")
            return

        if action == "reinforce":
            match_node.argument.extend(new_argument)
            match_node.argument = list(set(match_node.argument))
            match_node.update_status(match_node.status)
        elif action == "rebut" or action == "attack":
            new_node = match_node.add_node(new_claim=new_claim, new_argument=new_argument)
            match_node.update_status("attacked")
            new_node.update_status("proposed")
        else:
            raise ValueError(f"Unknown action: {action}")
        return

    def get_tree_info(self):
        info = {
            "motion": self.motion,
            "root": self.root.data,
            "side": self.side,
            "structure": self.root.get_node_info(),
            "meta_attack_list": self.meta_attack_list,
            "meta_rebuttal_list": self.meta_rebuttal_list,
        }
        return info

    @staticmethod
    def from_json(json_info):
        motion = json_info["motion"]
        side = json_info["side"]
        root_info = json_info["structure"]
        meta_attack_list = json_info["meta_attack_list"] if "meta_attack_list" in json_info else []
        meta_rebuttal_list = json_info["meta_rebuttal_list"] if "meta_rebuttal_list" in json_info else []
        tree = DebateTree(motion, side)
        tree.root = Node.from_json(motion, side, None, root_info)
        tree.meta_attack_list = meta_attack_list
        tree.meta_rebuttal_list = meta_rebuttal_list
        return tree


# python debate_tree.py --mode update --save_suffix RMH --load_from ../results1217_3/gemini-1.5-pro/if_health_care_is_a_scarce_resource,_government_should_step_in_to_ration_care,_deciding_whose_life_is_worth_saving_pool_against.json --use_reward_model --soft_logits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="prepare", choices=["prepare", "update", "backward"])
    parser.add_argument(
        "--motion", type=str, default="Artists should be free to borrow from cultures other than their own"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="Restricting cultural borrowing would undermine the universality of art and limit its ability to transcend cultural differences.",
    )
    parser.add_argument("--side", type=str, default="for")
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--branch", type=int, default=2)
    parser.add_argument("--use_reward_model", action="store_true", default=False)
    parser.add_argument("--soft_logits", action="store_true", default=False)
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--save_suffix", type=str, default="gemini")
    args = parser.parse_args()

    if args.mode == "prepare":
        logger.info("Prepare Debate Tree for\nMotion: {}\nRoot: {}\nSide: {}".format(args.motion, args.root, args.side))
        motion = args.motion
        root = args.root
        side = args.side
        use_reward_model = args.use_reward_model

        proposer = partial(HelperClient, model="gemini-2.0-flash", max_tokens=2048, temperature=1)
        """use (scorer.func.__name__ == 'reward_model') to differentiate"""
        if use_reward_model:
            scorer = partial(reward_model, soft=args.soft_logits)
        else:
            scorer = partial(HelperClient, model="gemini-2.0-flash", max_tokens=2048, temperature=0)

        # proposer = partial(gpt, model="gpt-3.5-turbo", max_tokens=2048)
        # scorer = partial(gpt, model="gpt-3.5-turbo", max_tokens=2048)
        tree = PrepareTree(root, motion, side, proposer, scorer)
        tree.expand_tree(tree.root, max_level=args.level, max_branch=args.branch)
        logger.info(tree.print_tree(prefix="\n"))
        logger.info(tree.get_tree_info())

        best_path_idx, best_path, best_score = tree.root.get_minimax_score(
            max_depth=args.level - 1, support_weight=0.5, level_decoy=0.8
        )
        best_path_str = "\n=> ".join([node.data for node in best_path])
        logger.info(f"Path IDX: {best_path_idx}")
        logger.info(f"Path: {best_path_str}")
        logger.info(f"Score: {best_score}")

    elif args.mode == "update":
        logger.info(
            "Update Debate Tree for\nLoad From: {}\nNew Scorer: {}\nSoft Mode: {}".format(
                args.load_from, "gemini-2.0-flash" if not args.use_reward_model else "reward_model", args.soft_logits
            )
        )
        data = json.load(open(args.load_from))
        if args.use_reward_model:
            scorer = partial(reward_model, soft=args.soft_logits)
        else:
            scorer = partial(gemini, model="gemini-2.0-flash", max_tokens=2048, temperature=0)

        for group in data:
            for item in group[:1]:
                tree = PrepareTree.from_json(item["tree_structure"])
                update_eval_score(tree.root, scorer)
                best_path_idx, best_path, best_score = tree.root.get_minimax_score(
                    max_depth=2, support_weight=0.5, level_decoy=0.8
                )
                best_path_str = "\n=> ".join([node.data for node in best_path])
                logger.debug(f"Path IDX: {best_path_idx}")
                logger.debug(f"Path: {best_path_str}")
                logger.debug(f"Minimax Score: {best_score}")
                item["minimax_search_score"] = best_score
                item["tree_structure"] = tree.get_tree_info()

        save_file = args.load_from.replace(".json", f"_{args.save_suffix}.json")
        logger.info(f"Save to: {save_file}")
        with open(save_file, "w") as f:
            json.dump(data, f, indent=2)

    elif args.mode == "backward":
        logger.info("backward Debate Tree for Minimax Strength")
        data = json.load(open(args.load_from))

        for group in data:
            for item in group[:1]:
                tree = PrepareTree.from_json(item["tree_structure"])
                tree.backward(level_decoy=0.8, support_weight=0.5)
                item["tree_structure"] = tree.get_tree_info()
                logger.debug(tree.print_tree(include_status=True))

        save_file = args.load_from.replace(".json", f"_backward.json")
        logger.info(f"Save to: {save_file}")
        with open(save_file, "w") as f:
            json.dump(data, f, indent=2)

    else:
        raise NotImplementedError(f"Unknown mode: {args.mode}")
