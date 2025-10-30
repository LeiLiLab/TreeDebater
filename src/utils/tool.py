import re
import os
import logging
from typing import List
from pulp import LpMaximize, LpProblem, LpVariable
import time

from .constants import MAX_TRY_NUM
from .prompts import debater_system_prompt

log_file_path = ""

def get_output_path(base_dir="../log_files/", suffix="log"):
    global log_file_path
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    log_files = [f for f in os.listdir(base_dir) if f.endswith('.log')]
    if log_files:
        max_num = max(int(f.split('.')[0]) for f in log_files)
        new_log_file = f"{max_num + 1}.{suffix}"
    else:
        new_log_file = f"1.{suffix}"
    log_file_path = os.path.join(base_dir, new_log_file)
    return log_file_path

def create_log(log_file=None):
    log = logging.getLogger('debate_logger')
    if not log.hasHandlers():  # Check if the logger already has handlers
        log.setLevel(logging.DEBUG)  # Set the logger level to DEBUG

        if log_file is None:
            log_file = get_output_path()
            print(f"Log file: {log_file}")
        
        # File handler for logging to a file with DEBUG level
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)

        # Stream handler for logging to console with INFO level
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        stream_handler.setFormatter(stream_formatter)

        # Add handlers to the logger
        log.addHandler(file_handler)
        log.addHandler(stream_handler)
    return log

logger = create_log() 

def find_json(x):
    idx = x.find('{')
    ridx = x.rfind('}')
    if idx == -1 or ridx == -1:
        return ''
    return x[idx:ridx+1]

def extract_numbers(s):
    numbers = [float(n) for n in re.findall(r'[-+]?[0-9]*\.?[0-9]+', s)]
    return numbers

def remove_citation(text: str, keep_main=False) -> str:
    """
    Remove citations in formats like (1), (1,2), [1], [1,2,3] from text.
    
    Args:
        text (str): Input text containing citations
        
    Returns:
        str: Text with citations removed
    """
    # Remove citations with parentheses ()
    statement = re.sub(r"[\*]*Reference[\S]*[\*]*", "Reference", text)
    statement = statement.replace("## Reference", "Reference")
    statement = statement.replace("Reference", "**Reference**")


    if "**Reference**" in statement:
        content, reference = statement.split("**Reference**")[:2]
    else:
        content, reference = statement, ''

    if not keep_main:
        pattern = r'\([^0-9_\s\W]*\)'
        content = re.sub(pattern, '', content)
        
        # Remove citations with square brackets []
        pattern = r'\[[^0-9_\s\W]*\]'
        content = re.sub(pattern, '', content)
        
        # Remove extra whitespace that might be left
        # content = re.sub(r'\s+', ' ', content)
    
    return content.strip(), reference.strip()

def remove_subtitles(content):
    pattern = r'^\*\*.*\*\*$'
    # Remove lines matching the pattern
    cleaned_text = re.sub(pattern, '', content, flags=re.MULTILINE)

    # Remove any leading/trailing whitespace or empty lines
    cleaned_text = '\n'.join([line for line in cleaned_text.split('\n')])

    return cleaned_text

def find_max_three_indices(list_of_lists):
    max_values = []
    for sublist in list_of_lists:
        max_value = max(sublist)
        index = sublist.index(max_value)
        max_values.append((max_value, index))

    sorted_indices = sorted(range(len(max_values)), key=lambda x: max_values[x][0], reverse=True)[:3]

    result_indices = []
    for global_index in sorted_indices:
        result_indices.append((global_index, max_values[global_index][1], max_values[global_index][0]))

    return result_indices

def extract_list_from_response(response):
    '''
    response format:
    1. [...]
    2. [...]
    '''
    # Split the response into lines
    lines = response.strip().split('\n')
    
    # Initialize an empty list to store the extracted items
    extracted_list = []
    
    # Iterate through each line
    for line in lines:
        # Remove leading/trailing whitespace
        line = line.strip()
        
        # Check if the line starts with a number followed by a period
        if line and line[0].isdigit() and '. ' in line:
            # Split the line at the first occurrence of '. '
            _, item = line.split('. ', 1)
            
            # Add the item to the extracted list
            extracted_list.append(item)
    
    return extracted_list



def lp_optimize(actions: List[str], rewards: List[float], costs: List[float], budget: float):
    assert len(actions) == len(rewards) == len(costs)
    assert len(actions) % 3 == 0

    # Define the problem
    problem = LpProblem("Maximize_Rewards_With_Constraints", LpMaximize)
    # Define binary decision variables
    x = {a: LpVariable(a, cat='Binary') for a in actions}
    # Objective function
    problem += sum(rewards[i] * x[actions[i]] for i in range(len(actions))), "Total_Reward"
    # Budget constraint
    problem += sum(costs[i] * x[actions[i]] for i in range(len(actions))) <= budget, "Budget"

    # Add "no three actions" constraints (e.g., a1, a2, a3 cannot all be selected)
    for i in range(0, len(actions), 3):
        problem += x[actions[i]] + x[actions[i+1]] + x[actions[i+2]] <= 1, f"No_Three_Actions_{i}_{i+1}_{i+2}"

    # Solve the problem
    problem.solve()

    # Display results
    selected_actions = [a for i, a in enumerate(actions) if x[a].value() == 1]
    total_reward = sum(rewards[i] for i, a in enumerate(actions) if x[a].value() == 1)
    total_cost = sum(costs[i] for i, a in enumerate(actions) if x[a].value() == 1)

    return selected_actions, total_reward, total_cost


def get_response_with_retry(llm, prompt, required_key, **kwargs):
    retry = 0
    response = ""
    content = {}
    while len(content) == 0 and retry < MAX_TRY_NUM:
        try:
            response = llm(prompt=prompt, sys=debater_system_prompt, **kwargs)[0]
            content = find_json(response)
            response = response.replace("null", "")
            content = eval(content)
            content = content[required_key]
        except Exception as e:
            logger.warning(f"Error {e} in extracting {required_key} from: {response}")
            content = {}
            retry += 1
            logger.debug(f"Retry {retry} times.")
            time.sleep(30)
    return content, response

def convert_messages_to_prompt(messages):
    prompt = []
    for message in messages:
        role = message.get('role', '')
        content = message.get('content', '')
        if role == 'system':
            prompt.append(f"System: {content}")
        elif role == 'user':
            prompt.append(f"Human: {content}")
        elif role == 'assistant':
            prompt.append(f"Assistant: {content}")
    return "\n\n".join(prompt)


def identify_number_in_text(text):
    pattern = r' [-+]?[0-9]*\.?[0-9]+(?:%|\b) '
    numbers = re.findall(pattern, text)
    numbers = [x.strip() for x in numbers]
    return numbers



def sort_by_importance(importance):
    importance_order = {"high": 3, "medium": 2, "low": 1}
    return importance_order.get(importance, 0)

def sort_by_action(action):
    action_order = {"attack": 3, "reinforce": 2, "propose": 1}
    return action_order.get(action, 0)