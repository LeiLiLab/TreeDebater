import argparse
import json
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.timing_log import log_llm_io
from src.utils.tool import extract_numbers, find_json, logger


def extract_claims(llm, title, side, content, verbose=False):
    system_prompt = """You are an advanced analytical tool designed to dissect and summarize arguments. Your task is to extract the main claims and their supporting statements from given texts. Follow these guidelines:

    1. Identify main claims:
    - Recognize the central arguments or assertions made in the text
    - Distinguish between primary claims and secondary or supporting points

    2. Extract supporting statements:
    - For each main claim, identify the statements, evidence, or reasoning used to support it
    - Include relevant examples, statistics, or expert opinions cited

    3. Organize information:
    - Present each main claim followed by its supporting statements
    - Use clear, concise language to summarize claims and support.

    4. Maintain neutrality:
    - Extract claims and statements without evaluating their validity or strength
    - Preserve the original intent and context of the arguments

    When presented with a text, extract and list the main claims and their supporting statements according to these guidelines. 
    """

    TEMPLATE = """The topic of the debate is "{title}".

    You should analyze the {myside} side and here is the transcript of the {myside} side:


    {content}


    Return the main claims and supporting statements in a Json format. The key is the main claim and the value is the support materials.
    """

    prompt = TEMPLATE.format(title=title, myside=side, content=content)
    if verbose:
        print(prompt)

    response = llm(prompt=prompt, temperature=0, max_tokens=1500, n=1, sys=system_prompt)
    claims = find_json(response[0])

    try:
        claims = eval(claims)
    except:
        claims = {}
        print("Error in extracting claims")

    return claims


def extract_obj_aspect(llm, title, side, content, claim_against=None):
    system_prompt = """You are an expert debate judge tasked with evaluating debaters' performances based on specific criteria. Your assessment should be objective and focus on the following key areas:

    1. Count the number of flaws or logical inconsistencies:
    - Potential logical fallacies of statements
    - Unsupported assertions or assumptions

    2. Count the number of each type of evidence used:
    - Inferences drawn from available information
    - Statistics presented to support arguments
    - Case studies or examples used to illustrate points

    3. Count the number of any significant arguments left unanswered during the rebuttal

    4. Overall Convincingness of claims (1-5):
    - Flaws or logical inconsistencies will decrease the score
    - Significant arguments left unanswered will decrease the score
    - The variety and quality of evidence used will increase the score


    Your evaluation should:
    - Remain objective, focusing on the quality of argumentation rather than personal opinions on the topic

    When presented with a debate transcript or summary, analyze it according to these guidelines. Be prepared to explain your evaluation and provide detailed feedback on specific aspects of the debate performance."""

    TEMPLATE = """The topic of the debate is "{title}".

    You should analyze the {myside} side and here is the transcript of the {myside} side:


    {content}

    
    The following are the other side's claims that are helpful to determine 'Unanswered Arguments', how many opponent's attacks are not addressed.

    {claim_against}


    Return the aspects and their scores in a Json format. The key is the aspect and the value is the score. The keys are :
    * "Logical Inconsistencies"
    * "Unsupported Assertions"
    * "Inferences"
    * "Statistics"
    * "Case Studies"
    * "Unanswered Arguments"
    * "Overall Convincingness"
    """

    prompt = TEMPLATE.format(title=title, myside=side, content=content, claim_against=claim_against)
    # print(prompt)

    response = llm(prompt=prompt, temperature=0, max_tokens=800, n=1, sys=system_prompt)

    # print("extract_obj_aspect:", response[0])
    scores = find_json(response[0])

    try:
        scores = eval(scores)
    except:
        scores = {}
        print("Error in extracting scores")

    return scores, response[0]


def eval_surprise(llm, title, side, claims, n=3, reduction=False, verbose=False):
    system_prompt = """You are an advanced debate analysis tool designed to assess the surprise factor of claims and arguments presented in debates. Your task is to evaluate how unexpected, novel, or counterintuitive the claims are within the context of the debate topic. Follow these guidelines in your assessment:
    1. Novelty of ideas:
    - Identify claims that present new perspectives or unconventional approaches to the topic
    - Assess how different these ideas are from commonly held views or standard arguments

    2. Counterintuitive reasoning:
    - Recognize arguments that challenge conventional wisdom or initial assumptions
    - Evaluate the effectiveness of paradoxical or seemingly contradictory claims

    3. Unexpected evidence:
    - Identify the use of surprising statistics, case studies, or examples
    - Assess how effectively unexpected evidence is used to support claims

    4. Innovative connections:
    - Recognize when debaters make unique links between ideas or concepts
    - Evaluate the relevance and effectiveness of these connections

    5. Context consideration:
    - Consider the level of surprise within the specific debate context and topic
    - Assess how the audience's likely expectations are challenged

    6. Impact of surprise:
    - Evaluate how the surprise factor contributes to the overall persuasiveness of the argument
    - Consider whether the surprise enhances or potentially detracts from the argument's credibility

    7. Originality vs. effectiveness balance:
    - Assess whether surprising elements are merely novel or also logically sound and relevant

    8. Quantification:
    - Rate the level of surprise for each main claim on a scale (e.g., 1-5, where 1 is entirely expected and 5 is highly surprising)

    When analyzing a debate or argument, identify the main claims and evaluate their surprise factor according to these criteria. Provide a brief explanation for each rating, highlighting what makes the claim surprising or unexpected. Be prepared to elaborate on your assessment and discuss how the surprise factor influences the overall effectiveness of the argument."""

    TEMPLATE = """The topic of the debate is "{title}".

    You should analyze the {myside} side and here is the JSON format of the main claims of the {myside} side. The key is the main claim and the value is the support materials.


    {claims}


    Return the main claims and their surprise scores in a Json format. The responses should be in Json format with one key of **result**. 
    The value of this key is a list of claims and their surprise scores.
    The key of each element in the list should be **claim**, **surprise_score**, and **explanation**. 
    The main claims should be exactly the above ones. 
    """

    prompt = TEMPLATE.format(title=title, myside=side, claims=claims)
    if verbose:
        print(prompt)

    response = llm(prompt=prompt, temperature=1, max_tokens=2048, n=n, sys=system_prompt)

    surprises = []
    score_list = []
    for i in range(n):
        surprise = find_json(response[i])
        try:
            surprise = eval(surprise)
            surprise = surprise["result"]
        except:
            print(surprise)
            print("Error in extracting surprise scores")
            surprise = {}
            continue

        non_number_score = [
            v for v in surprise if ("surprise_score" not in v) or not isinstance(v["surprise_score"], (int, float))
        ]
        if len(non_number_score) > 0:
            print("There is non-number values in the surprise scores")
            # print(non_number_score)

        scores = {}
        for v in surprise:
            scores[v["claim"]] = v["surprise_score"]
        surprises.append(surprise)
        score_list.append(scores)

    if reduction:
        result = {}
        for key in score_list[0].keys():
            values = [d[key] for d in surprises if key in d]
            # values = [v for v in values if isinstance(v, (int, float))]
            result[key] = sum(values) / len(values) if values else 0
        return result, surprises

    return score_list, surprises


def evaluate_support_strength(llm, motion, argument1, argument2, history=None):
    if llm.func.__name__ == "reward_model":  # trained reward model, should be partial() type
        relation_ship = "supporting"
        assert history[-1] == argument1 and history[-3] == argument2
        prompt = f"You are given a chain of arguments, each one supporting or attacking the previous one. The first argument is: {history[0]} The second last one is: {history[-3]} The last one is: {history[-1]} Now you need to determine the impact of the last one to the second last one, given their relationship {relation_ship}. Output only a number among 0, 1, or 2 in your response. 0 means not impactful; 1 means medium impactful; 2 means impactful."
        return llm(prompt)

    prompt = (
        f"""There is a debate with a title \"{motion}\" Please evaluate the support strength of the first argument to the second argument.\n"""
        f"Argument 1: {argument1}\n"
        f"Argument 2: {argument2}\n"
        f"""The two arguments are from the same side in a debate, and the support strength refers to how well the first argument adds to the second argument. Each score ranges from 1 to 3, with 1 being the lowest and 3 being the highest. Provide your evaluation as a single number in the format "Score: [score]". You can additionally provide a brief explanation of your evaluation."""
    )
    log_llm_io(logger, phase="evaluator", title="Support-Strength-Prompt", body=prompt.strip())
    response = llm(prompt=prompt, temperature=0)[0]
    log_llm_io(logger, phase="evaluator", title="Support-Strength-Response", body=response.strip())
    response = response.replace("*", "")
    pos = response.find("Score: ")
    numbers = extract_numbers(response[pos : pos + 15])
    return numbers[0]


def evaluate_defense_strength(llm, motion, argument1, argument2, history=None):
    if llm.func.__name__ == "reward_model":  # trained reward model, should be partial() type
        relation_ship = "attacking"
        # print('[DEBUG]', argument1, argument2, history)
        assert history[-1] == argument1 and history[-2] == argument2
        prompt = f"You are given a chain of arguments, each one supporting or attacking the previous one. The first argument is: {history[0]} The second last one is: {history[-2]} The last one is: {history[-1]} Now you need to determine the impact of the last one to the second last one, given their relationship {relation_ship}. Output only a number among 0, 1, or 2 in your response. 0 means not impactful; 1 means medium impactful; 2 means impactful."
        return llm(prompt)

    prompt = (
        f"""There is a debate with a title \"{motion}\" Please evaluate the rebuttal strength of the first argument to the second argument.\n"""
        f"Argument 1: {argument1}\n"
        f"Argument 2: {argument2}\n"
        """The two arguments are from the different sides in a debate, and the rebuttal strength refers to how well the first argument undermines the second argument. Each score ranges from 1 to 3, with 1 being the lowest and 3 being the highest. Provide your evaluation as a single number in the format "Score: [score]". You can additionally provide a brief explanation of your evaluation."""
    )
    log_llm_io(logger, phase="evaluator", title="Support-Defense-Prompt", body=prompt.strip())
    response = llm(prompt=prompt, temperature=0)[0]
    log_llm_io(logger, phase="evaluator", title="Support-Defense-Response", body=response.strip())
    pos = response.find("Score: ")
    numbers = extract_numbers(response[pos : pos + 15])
    return numbers[0]
