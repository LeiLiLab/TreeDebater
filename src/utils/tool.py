import logging
import os
import re
import time
import json
from typing import Any, List, TypeVar

from pydantic import BaseModel, ValidationError
from pulp import LpMaximize, LpProblem, LpVariable

from .constants import MAX_TRY_NUM
from .prompts import debater_system_prompt

log_file_path = ""
io_log_file_path = ""

debate_io_logger = logging.getLogger("debate_io_logger")
debate_io_logger.setLevel(logging.DEBUG)
debate_io_logger.propagate = False


def io_logging_enabled() -> bool:
    """True when prompt/response blocks go to the I/O log file (default on)."""
    if os.environ.get("DEBATE_LOG_PROMPTS", "1").lower() in ("0", "false", "no", "off"):
        return False
    return bool(debate_io_logger.handlers)


def _setup_debate_io_logger(main_log_file: str) -> None:
    """Sibling file ``N_io.log`` next to ``N.log`` for large prompt/response bodies."""
    global io_log_file_path
    if not main_log_file or not main_log_file.endswith(".log"):
        return
    if os.environ.get("DEBATE_LOG_PROMPTS", "1").lower() in ("0", "false", "no", "off"):
        return
    if debate_io_logger.handlers:
        return
    io_path = main_log_file.replace(".log", "_io.log")
    io_log_file_path = io_path
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    h = LazyFileHandler(io_path, mode="a", encoding="utf-8")
    h.setLevel(logging.DEBUG)
    h.setFormatter(fmt)
    debate_io_logger.addHandler(h)


class LazyFileHandler(logging.FileHandler):
    """
    FileHandler that only creates the log file when the first log record is emitted.
    This prevents creation of empty log files when programs exit early or crash during init.
    """

    def __init__(self, filename, mode='a', encoding=None, delay=True):
        """
        Initialize with delay=True to defer file creation.
        File will be created on first emit() call.
        """
        # Store filename for later use
        self._lazy_filename = filename
        self._lazy_mode = mode
        self._lazy_encoding = encoding

        # Don't call parent __init__ yet - we'll do it lazily
        logging.Handler.__init__(self)

        self.baseFilename = os.path.abspath(filename)
        self.mode = mode
        self.encoding = encoding
        self.stream = None
        self._file_created = False

    def _open(self):
        """Open the log file (called on first emit)."""
        # Ensure directory exists
        log_dir = os.path.dirname(self.baseFilename)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        return open(self.baseFilename, self.mode, encoding=self.encoding)

    def emit(self, record):
        """
        Emit a record. Create file on first call.
        """
        if not self._file_created:
            # Create file now that we have something to log
            self.stream = self._open()
            self._file_created = True

        # Now emit normally
        if self.stream:
            try:
                msg = self.format(record)
                stream = self.stream
                stream.write(msg + self.terminator)
                self.flush()
            except Exception:
                self.handleError(record)

    def close(self):
        """Close file handler."""
        self.acquire()
        try:
            if self.stream and self._file_created:
                try:
                    self.flush()
                    if hasattr(self.stream, "close"):
                        self.stream.close()
                finally:
                    self.stream = None
        finally:
            self.release()
        logging.Handler.close(self)


def get_output_path(base_dir="../log_files/", suffix="log"):
    global log_file_path
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    log_files = [f for f in os.listdir(base_dir) if f.endswith(".log")]
    # Only "N.log" (integer N), not e.g. "19_io.log" or "debug.log"
    numbered_logs = [f for f in log_files if len(f) > 4 and f[:-4].isdigit()]
    if numbered_logs:
        max_num = max(int(f[:-4]) for f in numbered_logs)
        new_log_file = f"{max_num + 1}.{suffix}"
    else:
        new_log_file = f"1.{suffix}"
    log_file_path = os.path.join(base_dir, new_log_file)
    return log_file_path


def create_log(log_file=None):
    log = logging.getLogger("debate_logger")
    if not log.hasHandlers():  # Check if the logger already has handlers
        log.setLevel(logging.DEBUG)  # Set the logger level to DEBUG

        if log_file is None:
            log_file = get_output_path()
            print(f"Log file: {log_file}")

        # Lazy file handler for logging to a file with DEBUG level
        # File will only be created when first log record is written
        file_handler = LazyFileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)

        # Stream handler for logging to console with INFO level
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        stream_handler.setFormatter(stream_formatter)

        # Add handlers to the logger
        log.addHandler(file_handler)
        log.addHandler(stream_handler)
        _setup_debate_io_logger(log_file)
        if io_log_file_path:
            log.debug(f"[timing] phase=io_log_ready io_log={io_log_file_path}")
    return log


logger = create_log()

# Alias for imports: ``from utils.tool import io_logger``
io_logger = debate_io_logger

T = TypeVar("T", bound=BaseModel)


def _strip_markdown_json_fence(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text


def find_json(x):
    return extract_json_object(x)


def extract_json_object(text: str) -> str:
    if text is None:
        return ""
    if isinstance(text, (dict, list)):
        return json.dumps(text)
    if not isinstance(text, str):
        text = str(text)

    text = _strip_markdown_json_fence(text).strip()
    idx = text.find("{")
    ridx = text.rfind("}")
    if idx != -1 and ridx != -1 and idx <= ridx:
        return text[idx : ridx + 1]
    lidx = text.find("[")
    rridx = text.rfind("]")
    if lidx != -1 and rridx != -1 and lidx <= rridx:
        return text[lidx : rridx + 1]
    return text


def parse_llm_json(text: Any, *, response_model: type[T] | None = None, required_key: str | None = None) -> T | Any:
    if isinstance(text, BaseModel):
        parsed = text
    elif isinstance(text, (dict, list)):
        parsed = text
    else:
        payload = extract_json_object(text)
        parsed = json.loads(payload)

    if response_model is not None:
        if isinstance(parsed, response_model):
            validated = parsed
        else:
            validated = response_model.model_validate(parsed)
        if required_key is None:
            return validated
        dumped = validated.model_dump()
        if required_key not in dumped:
            raise KeyError(f"Missing required key '{required_key}' in validated response.")
        return dumped[required_key]

    if required_key is None:
        return parsed
    if not isinstance(parsed, dict):
        raise TypeError(f"Expected dict for required_key='{required_key}', got {type(parsed).__name__}")
    if required_key not in parsed:
        raise KeyError(f"Missing required key '{required_key}' in parsed response.")
    return parsed[required_key]


def extract_numbers(s):
    numbers = [float(n) for n in re.findall(r"[-+]?[0-9]*\.?[0-9]+", s)]
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
        content, reference = statement, ""

    if not keep_main:
        pattern = r"\([^0-9_\s\W]*\)"
        content = re.sub(pattern, "", content)

        # Remove citations with square brackets []
        pattern = r"\[[^0-9_\s\W]*\]"
        content = re.sub(pattern, "", content)

        # Remove extra whitespace that might be left
        # content = re.sub(r'\s+', ' ', content)

    return content.strip(), reference.strip()


def remove_subtitles(content):
    pattern = r"^\*\*.*\*\*$"
    # Remove lines matching the pattern
    cleaned_text = re.sub(pattern, "", content, flags=re.MULTILINE)

    # Remove any leading/trailing whitespace or empty lines
    cleaned_text = "\n".join([line for line in cleaned_text.split("\n")])

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
    """
    response format:
    1. [...]
    2. [...]
    """
    # Split the response into lines
    lines = response.strip().split("\n")

    # Initialize an empty list to store the extracted items
    extracted_list = []

    # Iterate through each line
    for line in lines:
        # Remove leading/trailing whitespace
        line = line.strip()

        # Check if the line starts with a number followed by a period
        if line and line[0].isdigit() and ". " in line:
            # Split the line at the first occurrence of '. '
            _, item = line.split(". ", 1)

            # Add the item to the extracted list
            extracted_list.append(item)

    return extracted_list


def lp_optimize(actions: List[str], rewards: List[float], costs: List[float], budget: float):
    assert len(actions) == len(rewards) == len(costs)
    assert len(actions) % 3 == 0

    # Define the problem
    problem = LpProblem("Maximize_Rewards_With_Constraints", LpMaximize)
    # Define binary decision variables
    x = {a: LpVariable(a, cat="Binary") for a in actions}
    # Objective function
    problem += sum(rewards[i] * x[actions[i]] for i in range(len(actions))), "Total_Reward"
    # Budget constraint
    problem += sum(costs[i] * x[actions[i]] for i in range(len(actions))) <= budget, "Budget"

    # Add "no three actions" constraints (e.g., a1, a2, a3 cannot all be selected)
    for i in range(0, len(actions), 3):
        problem += x[actions[i]] + x[actions[i + 1]] + x[actions[i + 2]] <= 1, f"No_Three_Actions_{i}_{i+1}_{i+2}"

    # Solve the problem
    problem.solve()

    # Display results
    selected_actions = [a for i, a in enumerate(actions) if x[a].value() == 1]
    total_reward = sum(rewards[i] for i, a in enumerate(actions) if x[a].value() == 1)
    total_cost = sum(costs[i] for i, a in enumerate(actions) if x[a].value() == 1)

    return selected_actions, total_reward, total_cost


def get_response_with_retry(llm, prompt, required_key, *, response_model: type[T] | None = None, **kwargs):
    from utils.timing_log import log_timing

    retry = 0
    response = ""
    content = {}
    while retry < MAX_TRY_NUM:
        try:
            t0 = time.perf_counter()
            response_obj = llm(prompt=prompt, sys=debater_system_prompt, response_model=response_model, **kwargs)[0]
            llm_s = time.perf_counter() - t0
            log_timing(
                logger,
                "get_response_with_retry_llm",
                llm_s,
                required_key=required_key,
                attempt=retry + 1,
                response_model=response_model.__name__ if response_model is not None else None,
            )
            if isinstance(response_obj, BaseModel):
                response = json.dumps(response_obj.model_dump(), ensure_ascii=False)
                content = parse_llm_json(
                    response_obj,
                    response_model=response_model or type(response_obj),
                    required_key=required_key,
                )
            else:
                response = response_obj if isinstance(response_obj, str) else json.dumps(response_obj, ensure_ascii=False)
                content = parse_llm_json(response_obj, response_model=response_model, required_key=required_key)
            if content is not None and content != {}:
                return content, response
        except (json.JSONDecodeError, ValidationError, KeyError, TypeError, ValueError) as e:
            logger.warning(f"Error {e} in extracting {required_key} from: {response}")
            content = {}
            retry += 1
            logger.debug(f"Retry {retry} times.")
            time.sleep(30)
        except Exception as e:
            logger.warning(f"Unexpected error {e} in extracting {required_key} from: {response}")
            content = {}
            retry += 1
            logger.debug(f"Retry {retry} times.")
            time.sleep(30)
    return content, response


def convert_messages_to_prompt(messages):
    prompt = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        if role == "system":
            prompt.append(f"System: {content}")
        elif role == "user":
            prompt.append(f"Human: {content}")
        elif role == "assistant":
            prompt.append(f"Assistant: {content}")
    return "\n\n".join(prompt)


def identify_number_in_text(text):
    pattern = r" [-+]?[0-9]*\.?[0-9]+(?:%|\b) "
    numbers = re.findall(pattern, text)
    numbers = [x.strip() for x in numbers]
    return numbers


def sort_by_importance(importance):
    importance_order = {"high": 3, "medium": 2, "low": 1}
    return importance_order.get(importance, 0)


def sort_by_action(action):
    action_order = {"attack": 3, "reinforce": 2, "propose": 1}
    return action_order.get(action, 0)
