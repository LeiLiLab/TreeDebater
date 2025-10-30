import copy
import json
import math
import os
import random
import re
import time
import traceback
from dataclasses import dataclass
from functools import partial

import litellm
import requests
from openai import OpenAI

from evaluator import eval_surprise, extract_claims, extract_obj_aspect
from tts import convert_text_to_speech, trim_audio_by_sentences
from utils.constants import CLOSING_TIME, OPENING_TIME, REBUTTAL_TIME, WORDRATIO, deepseek_api_key
from utils.model import HelperClient, safety_setting
from utils.prompts import *
from utils.tool import log_file_path, logger


@dataclass
class AgentConfig:
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    system_prompt: str = ""


@dataclass
class DebaterConfig(AgentConfig):
    side: str = "for"
    type: str = "default"
    use_retrieval: bool = False
    pool_file: str = None
    system_prompt: str = expert_debater_system_prompt + rhetorical_techniques_prompt
    add_retrieval_feedback: bool = True
    use_rehearsal_tree: bool = True
    use_debate_flow_tree: bool = True
    url: str = "http://127.0.0.1:8081/"


@dataclass
class JudgeConfig(AgentConfig):
    system_prompt: str = judge_system_prompt
    temperature: float = 0.0


@dataclass
class AudienceConfig(AgentConfig):
    pre_prompt: str = audience_system_prompt_pre
    post_prompt: str = audience_system_prompt_post
    temperature: float = 1.0
    n: int = 1


class Agent:
    def __init__(self, config) -> None:
        self.config = config
        self.system_prompt = config.system_prompt
        print(f"[Agent Init] Model: {self.config.model}")
        if self.config.model.startswith("gpt") or self.config.model.startswith("o1"):
            self.client = partial(
                litellm.completion,
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        elif self.config.model.startswith("gemini"):
            self.client = partial(
                litellm.completion,
                model="gemini/" + self.config.model,
                api_key=os.environ["GOOGLE_API_KEY"],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                safety_settings=safety_setting,
            )
        elif "llama" in self.config.model.lower():
            self.client = partial(
                litellm.completion,
                model="together_ai/" + self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        elif "deepseek" in self.config.model.lower():
            self.client = partial(
                litellm.completion,
                model="deepseek/" + self.config.model,
                api_key=deepseek_api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        elif "moonshot" in self.config.model.lower():
            self.client = partial(
                litellm.completion,
                model="moonshot/" + self.config.model,
                api_base="https://api.moonshot.cn/v1",
                api_key=os.environ["MOONSHOT_API_KEY"],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
        else:
            raise ValueError(f"Invalid model: {self.config.model}")

        self.conversation = []
        if self.config.system_prompt != "":
            self._add_message("system", self.config.system_prompt)

        self.client_cost = 0

    def speak(self, prompt, **kwargs):
        self._add_message("user", prompt)
        logger.debug(f"[Conversation-History] {json.dumps(self.conversation)}")
        logger.debug("[Prompt] " + prompt.strip().replace("\n", " ||| "))
        response = self._get_response(self.conversation, **kwargs)
        if kwargs.get("n", 1) > 1:
            logger.info(f"[Response] {response}")
            return response
        logger.debug("[Response-Before-Post-Process] " + response.strip().replace("\n", " ||| "))
        response = self.post_process(response, **kwargs)
        logger.debug("[Response-After-Post-Process] " + response.strip().replace("\n", " ||| "))
        return response

    def post_process(self, statement, **kwargs):
        self._add_message("assistant", f"{statement}")
        logger.info("[Response] " + statement.strip().replace("\n", " ||| "))
        return statement

    def _get_response(self, messages, **kwargs):
        kwargs.pop("max_time", None)
        kwargs.pop("history", None)
        kwargs.pop("max_words", None)
        kwargs.pop("time_control", None)
        retry = 0
        while retry < 3:
            try:
                response = self.client(messages=messages, **kwargs)
                self.client_cost += response._hidden_params["response_cost"]
                response = [choice.message.content for choice in response.choices]
                if len(response) == 1:
                    response = response[0]
            except Exception as e:
                logger.error(f"Error in getting response: {e}")
                traceback.print_exc()
                response = None
            if response is None:
                retry += 1
                logger.warning(f"Waiting for 90 seconds before retrying...")
                time.sleep(90)
                logger.warning(f"Retry {retry} times")
            else:
                return response
        return ""

    def _add_message(self, role, content):
        if isinstance(content, list):
            content = "\n".join(content)
        self.conversation.append({"role": role, "content": content})

    def reset(self):
        self.conversation = []
        if self.config.system_prompt != "":
            self._add_message("system", self.config.system_prompt)


##################### Debater #####################


class Debater(Agent):
    def __init__(self, config, motion) -> None:
        super().__init__(config)

        self.motion = motion
        self.side = config.side
        self.oppo_side = "against" if self.side == "for" else "for"
        self.type = config.type
        self.act = "SUPPORT" if self.side == "for" else "OPPOSE"
        self.counter_act = "OPPOSE" if self.side == "for" else "SUPPORT"
        self.use_retrieval = config.use_retrieval
        self.debate_thoughts = []
        self.status = "prepare"

    def opening_generation(self, history, **kwargs):
        self.status = "opening"
        self.listen(history)
        prompt = default_opening_prompt.format(motion=self.motion, act=self.act)
        prompt = prompt.replace("{n_words}", str(math.ceil(kwargs.get("max_time", OPENING_TIME) / WORDRATIO["time"])))
        response = self.speak(prompt, **kwargs)
        return response

    def rebuttal_generation(self, history, **kwargs):
        self.status = "rebuttal"
        self.listen(history)
        opponent = history[-1]["content"]
        prompt = default_rebuttal_prompt.format(counter_act=self.counter_act, opponent=opponent, act=self.act)
        prompt = prompt.replace("{n_words}", str(math.ceil(kwargs.get("max_time", REBUTTAL_TIME) / WORDRATIO["time"])))
        response = self.speak(prompt, **kwargs)
        return response

    def closing_generation(self, history, **kwargs):
        self.status = "closing"
        self.listen(history)
        opponent = history[-1]["content"]
        prompt = default_closing_prompt.format(counter_act=self.counter_act, opponent=opponent, act=self.act)
        prompt = prompt.replace("{n_words}", str(math.ceil(kwargs.get("max_time", CLOSING_TIME) / WORDRATIO["time"])))
        response = self.speak(prompt, **kwargs)
        return response

    def post_process(self, statement, max_time=-1, time_control=False, **kwargs):
        """
        statement: AI生成的原始辩论陈述文本
        max_time: 最大允许的发言时间（秒），-1表示无限制
        time_control: 是否启用时间控制功能
        """
        statement = statement.strip()
        if statement is None:
            self._add_message("assistant", f"")
            logger.warning("[Response] Statement is None. Return empty string.")
            return ""

        # TODO: 这里没有设置固定去提取LLM回答的函数嘛？
        start_idx = statement.find("```")
        end_idx = statement.find("```", start_idx + 1)
        if start_idx != -1 and end_idx != -1:
            format = re.search(r"```(.*)\n", statement)
            if format is not None:
                format = format.group(0)
                length = len(format)
            else:
                length = 3
            statement = statement[start_idx + length : end_idx].strip()
        else:
            format = re.search(r"```(.*)\n", statement)
            if format is not None:
                end_idx = format.span()[1]
                trunc_statement = statement[end_idx:].strip()
                if len(trunc_statement) > 0:
                    statement = trunc_statement

        if max_time <= 0 or not time_control:
            self._add_message("assistant", f"{statement}")
            logger.info("[Response] " + statement.strip().replace("\n", " ||| "))
            return statement

        # NOTE the below part is time-consuming, can comment them and add "new_statement = statement" when developing
        prefix = log_file_path.replace(".log", "")
        audio_file = f"{prefix}_{self.config.type}_{self.status}_{self.side}.mp3"
        logger.debug("[Time-Control] Statement: " + statement.replace("\n", " ||| "))
        content, reference, duration = convert_text_to_speech(statement, audio_file)
        logger.debug(f"[Time-Control] Save Audio: {audio_file}")
        logger.debug(f"[Time-Control] Original Time: {duration:0.2f}")

        if duration <= max_time:
            logger.debug(f"[Time-Control] Final Time: {duration:0.2f}")
            new_content = content
        else:
            save_file = audio_file.replace(".mp3", "_trimmed.mp3")
            logger.debug(f"[Time-Control] Save Trimmed Audio: {save_file}")
            # duration, new_content = trim_audio(audio_file, save_file, max_minute=max_time/60)
            # TODO: 这里可以优化？
            duration, new_sentences = trim_audio_by_sentences(audio_file, save_file, max_duration=max_time * 1000)
            last_sentence = new_sentences[-1]
            idx = content.lower().find(last_sentence[:-1].lower())  # remove the punctuation
            if idx == -1:
                print(f"Last sentence not found in content")
                new_content = " ".join(new_sentences)
            else:
                new_content = content[: idx + len(last_sentence)]

            logger.debug(f"[Time-Control] Final Time: {duration:0.2f}")

        if new_content is None or len(new_content) == 0:
            logger.warning(f"[Time-Control] Trimmed Content is None. Use the original content as the transcript.")
            new_content = content

        new_statement = new_content + "\n\n**Reference**\n" + reference
        # new_statement = statement
        self._add_message("assistant", f"{new_statement}")
        logger.info("[Response] " + new_statement.strip().replace("\n", " ||| "))

        return new_statement

    def listen(self, history):
        if len(history) == 0:
            return
        assert history[-1]["side"] == self.oppo_side, "The opponent should be the last speaker"
        content = f"**Opponent's {history[-1]['stage']} Statement**\n" + history[-1]["content"]
        self._add_message("user", content)

    @property
    def next_oppo_status(self):
        # for-opening -> against-opening -> for-rebuttal -> against-rebuttal ->for-closing -> against-closing -> finished
        if self.side == "for":
            return self.status
        else:
            if self.status == "opening":
                return "rebuttal"
            elif self.status == "rebuttal":
                return "closing"
            else:
                return "finished"


class HumanDebater(Debater):
    def __init__(self, config, motion) -> None:
        super().__init__(config, motion)

    def get_multiline_input(self, instruction):
        print("\n[User Input]" + instruction)
        print("Please input your response in the command. End with 'END'.")
        lines = []
        while True:
            line = input()
            if line == "END":
                break
            lines.append(line)
        return "\n".join(lines)

    def opening_generation(self, **kwargs):
        self.status = "opening"
        max_time = kwargs.get("max_time", OPENING_TIME)
        max_words = math.ceil(max_time / WORDRATIO["time"])
        response = self.get_multiline_input(
            f"Please give an opening statement using three claims with {max_words} words, do not output other things. Please input the response in the command."
        )
        response = self.post_process(response, **kwargs)
        return response

    def rebuttal_generation(self, history, **kwargs):
        self.status = "rebuttal"
        max_time = kwargs.get("max_time", REBUTTAL_TIME)
        max_words = math.ceil(max_time / WORDRATIO["time"])
        response = self.get_multiline_input(
            f"Now it comes the rebuttal phase, where you respond to your opponent. You should stand firm on your position and attack the opponent's weak points. Give your response within {max_words} words and do not output other things than our response. Please input the response in the command."
        )
        response = self.post_process(response, **kwargs)
        return response

    def closing_generation(self, history, **kwargs):
        self.status = "closing"
        max_time = kwargs.get("max_time", CLOSING_TIME)
        max_words = math.ceil(max_time / WORDRATIO["time"])
        response = self.get_multiline_input(
            f"Now it comes the closing statement, where you summarize your key points and reaffirm your position. Give your response within {max_words} words and do not output other things than our response. Please input the response in the command."
        )
        response = self.post_process(response, **kwargs)
        return response


class BaselineDebater(Debater):
    def __init__(self, config, motion, port=8081) -> None:
        super().__init__(config, motion)
        language = "en"
        topic = motion
        model = config.model

        self.input = {
            "Language": language,
            "Topic": topic,
            "Position": "positive" if self.side == "for" else "negative",
            "Model": model,
        }
        self.BASE_URL = f"http://127.0.0.1:{port}/"
        logger.info(f"[BaselineDebater URL] {self.BASE_URL}")
        logger.debug("[BaselineDebater init] " + str(self.input))

    def _make_request(self, url, data):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=data).json()
                return response
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                else:
                    logger.error(f"Attempt {attempt + 1} failed: {e}. Maximum retries reached.")
                    raise
        return ""

    def opening_generation(self, history, **kwargs):
        self.status = "opening"
        if len(history) > 0:
            self.input["PositiveArgument"] = history[0]["content"]
            assert len(history) == 1 and self.oppo_side == "for"

        opening_response = self._make_request(self.BASE_URL + "v1/argument", self.input)
        opening = opening_response["Result"]
        self.input["Reference"] = opening_response["Reference"]
        logger.debug("[Baseline-opening-input] " + str(self.input).replace("\n", " ||| "))
        logger.debug("[Baseline-opening-before] " + opening.strip().replace("\n", " ||| "))
        opening = self.post_process(opening, **kwargs)
        logger.debug("[Baseline-opening-after] " + opening.strip().replace("\n", " ||| "))
        return opening

    def rebuttal_generation(self, history, **kwargs):
        self.status = "rebuttal"
        self.input.update(
            {
                "PositiveArgument": history[0]["content"],
                "NegativeArgument": history[1]["content"],
            }
        )
        if len(history) > 2:
            self.input["PositiveRebuttal"] = history[2]["content"]
            assert len(history) == 3 and self.oppo_side == "for"
        rebuttal_response = self._make_request(self.BASE_URL + "v1/rebuttal", self.input)
        rebuttal = rebuttal_response["Result"]
        self.input["Reference"] = rebuttal_response["Reference"]
        logger.debug("[Baseline-rebuttal-input] " + str(self.input).replace("\n", " ||| "))
        logger.debug("[Baseline-rebuttal-before] " + rebuttal.strip().replace("\n", " ||| "))
        rebuttal = self.post_process(rebuttal, **kwargs)
        logger.debug("[Baseline-rebuttal-after] " + rebuttal.strip().replace("\n", " ||| "))
        return rebuttal

    def closing_generation(self, history, **kwargs):
        self.status = "closing"
        self.input.update(
            {
                "PositiveRebuttal": history[2]["content"],
                "NegativeRebuttal": history[3]["content"],
            }
        )
        if len(history) > 4:
            self.input["PositiveSummary"] = history[4]["content"]
            assert len(history) == 5 and self.oppo_side == "for"
        # only consider the first 2 stages!
        summary_response = self._make_request(self.BASE_URL + "v1/summary", self.input)
        summary = summary_response["Result"]
        self.input["Reference"] = summary_response["Reference"]
        logger.debug("[Baseline-summary-input] " + str(self.input).replace("\n", " ||| "))
        logger.debug("[Baseline-summary-before] " + summary.strip().replace("\n", " ||| "))
        summary = self.post_process(summary, **kwargs)
        logger.debug("[Baseline-summary-after] " + summary.strip().replace("\n", " ||| "))
        return summary

    def reset_stage(self, stage, side, new_content):
        if stage == "opening":
            self.input["Topic"] = self.motion + "\nThe three main claims you should focus on are: " + new_content
            if side == "for":
                self.input["PositiveArgument"] = new_content
            else:
                self.input["NegativeArgument"] = new_content
        elif stage == "rebuttal":
            if side == "for":
                self.input["PositiveRebuttal"] = new_content
            else:
                self.input["NegativeRebuttal"] = new_content
        elif stage == "closing":
            return


##################### Judge #####################


class Judge(Agent):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.helper_client = partial(
            HelperClient, model=self.config.model, temperature=0, max_tokens=config.max_tokens, n=1
        )

    def eval(self, motion, debate_process, **kwargs):
        prompt = (
            f"The debate topic is {motion}. The for side is to support this motion while the against side is to oppose it. The debate process is as follows: \n"
            + json.dumps(debate_process, indent=2)
        )
        prompt += 'By adhering to these principles and criteria, you will provide an impartial and comprehensive evaluation of each side\'s performance, ensuring a fair and constructive outcome for the debate. Do determine the winner even if you find the two sides perform similarly. Please output your final judgment in the format: "The winning side is [For/Against] due to [reasons]."'
        response = self.speak(prompt, **kwargs)
        winner = self.extract_winner(response)
        return winner, response

    def comparison(self, motion, context, side, a, b, **kwargs):
        self.reset()
        prompt = (
            f"The debate topic is {motion}. The for side is to support this motion while the against side is to oppose it. "
            f"The debate process is as follows: \n{json.dumps(context, indent=2)}\n\n"
            f"Here are the two versions of the {side} side's response based on the debate process: \n\n"
            f"=========Version A Start======== \n{a}\n=========Version A End========\n\n"
            f"=========Version B Start======== \n{b}\n=========Version B End========\n\n"
            f"By adhering to these principles and criteria, you will provide an impartial and comprehensive evaluation of each version's performance, ensuring a fair and constructive outcome for the debate. Do determine the version even if you find the two sides perform similarly. "
            'Please output your final judgment in the format: "The better version is Version [A/B] due to [reasons]."'
        )
        response = self.speak(prompt, **kwargs)
        winner = self.extract_version(response)
        self.reset()
        return winner, response

    def extract_winner(self, comments):
        pos = comments.find("The winning side is")
        if "For" in comments[pos + 20 : pos + 33]:
            return "For wins"
        elif "Against" in comments[pos + 20 : pos + 33]:
            return "Against wins"
        else:
            return "[GGG judge not detected]"

    def extract_version(self, comments):
        pos = comments.find("better version is")
        if "A" in comments[pos + 18 : pos + 36]:
            return "A"
        elif "B" in comments[pos + 18 : pos + 36]:
            return "B"
        else:
            return "[GGG judge not detected]"

    def finegrained_check(self, motion, side_info, side):
        oppo = "against" if side == "for" else "for"
        claims = extract_claims(self.helper_client, motion, side, side_info[side]["content"])
        side_info[side]["claims"] = claims

        try:
            obj_scores, obj_scores_explanation = extract_obj_aspect(
                self.helper_client, motion, side, side_info[side]["content"], side_info[oppo]["claims"]
            )
        except:
            traceback.print_exc()
            exit(0)
        return obj_scores, obj_scores_explanation


class Audience(Agent):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.n = config.n
        self.helper_client = partial(
            HelperClient, model=self.config.model, temperature=0, max_tokens=config.max_tokens, n=1
        )

    def vote(self, process, motion):
        prompt = f"{self.config.pre_prompt}\n\n The for side is to support the motion of {motion}. The against side is to oppose the motion. "
        pre_vote = self.speak(prompt)
        prompt = f"The debate process is as follows." + json.dumps(process, indent=2) + self.config.post_prompt
        post_vote = self.speak(prompt)

        return self.extract_winner(pre_vote) + " -> " + self.extract_winner(post_vote)

    def comparison(self, motion, context, side, a, b, **kwargs):
        self.reset()
        prompt = (
            f"The debate topic is {motion}. The for side is to support this motion while the against side is to oppose it. "
            f"The debate process is as follows: \n{json.dumps(context, indent=2)}\n\n"
            f"Here are the two versions of the {side} side's response based on the debate process: \n\n"
            f"=========Version A Start======== \n{a}\n=========Version A End========\n\n"
            f"=========Version B Start======== \n{b}\n=========Version B End========\n\n"
            "Now that you've heard arguments from both sides, it's time to cast your final vote. Consider the following factors: \n"
            "Strength and clarity of each team's arguments \n"
            "Evidence and reasoning used to support their claims \n"
            "Effectiveness in addressing and countering the opposing team's points \n"
            "Overall persuasiveness and impact of each team's case \n"
            'Please output your final judgment in the format: "The better version is Version [A/B] due to [reasons]."'
        )
        response = self.speak(prompt, n=self.n, **kwargs)
        if self.n > 1:
            winner = [self.extract_version(r) for r in response]
        else:
            winner = self.extract_version(response)
        self.reset()
        return winner, response

    def extract_winner(self, response):
        pos = response.find("vote is")
        if "For" in response[pos + 8 : pos + 21]:
            return "For"
        elif "Against" in response[pos + 8 : pos + 21]:
            return "Against"
        else:
            return "[GGG]"

    def extract_version(self, response):
        pos = response.find("better version is")
        if "A" in response[pos + 18 : pos + 36]:
            return "A"
        elif "B" in response[pos + 18 : pos + 36]:
            return "B"
        else:
            return "[GGG]"

    def surprise(self, motion, side, claims):
        scores, explanations = eval_surprise(self.helper_client, motion, side, claims, n=1)
        return scores, explanations

    def feedback(self, prompt, **kwargs):
        response = self.speak(prompt, **kwargs)
        return response
