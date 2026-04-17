import copy
import json
import math
import os
import random
import re
import threading
import time
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional

import google.generativeai as genai
import litellm
import requests
import torch
from openai import OpenAI
from sentence_transformers.util import dot_score, normalize_embeddings, semantic_search
from tavily import TavilyClient

from agents import Audience, AudienceConfig, Debater
from debate_tree import DebateTree, PrepareTree
from prepare import ClaimPool
from utils.constants import REMAINING_ROUND_NUM, TIME_MODE_FOR_STATEMENT, TIME_TOLERANCE, WORDRATIO, get_embeddings
from utils.helper import (
    TimeAdjuster,
    build_logic_claims,
    extract_statement,
    get_actions_from_tree,
    get_battlefields_from_actions,
    get_retrieval_from_rehearsal_tree,
    rank_evidence,
)
from utils.model import HelperClient
from utils.prompts import *
from utils.time_estimator import LengthEstimator
from utils.timing_log import (
    clear_speak_io_context,
    log_io_block,
    log_llm_io,
    log_timing,
    next_call_id,
    set_speak_io_context,
    timed_phase,
)
from utils.tool import get_response_with_retry, io_logger, io_logging_enabled, logger, sort_by_action, sort_by_importance


class TreeDebater(Debater):
    def __init__(self, config, motion):
        super().__init__(config, motion)
        self.definition = None
        self.evidence_pool = []
        self.high_quality_evidence_pool = []  # this instead of self.evidence_pool is later used
        self.pool_file = config.pool_file

        self.add_retrieval_feedback = config.add_retrieval_feedback
        # Add new flags for controlling the use of rehearsal tree and debate flow tree
        self.use_rehearsal_tree = config.use_rehearsal_tree
        self.use_debate_flow_tree = config.use_debate_flow_tree
        logger.debug(
            "[TreeDebater] "
            + f"use_rehearsal_tree: {self.use_rehearsal_tree}, use_debate_flow_tree: {self.use_debate_flow_tree}"
        )

        self.helper_client = partial(
            HelperClient, model=self.config.model, temperature=0, max_tokens=config.max_tokens, n=1
        )
        self.simulated_audience = [Audience(AudienceConfig(model=self.config.model, temperature=1)) for _ in range(1)]

        # Initialize debate trees only if they are enabled
        if self.use_debate_flow_tree:
            self.debate_tree = DebateTree(motion=motion, side=self.side)
            self.oppo_debate_tree = DebateTree(motion=motion, side=self.oppo_side)
        else:
            # also create a dummy debate tree, otherwise in `_get_retrieval_debate_tree` will have error
            self.debate_tree = DebateTree(motion=motion, side=self.side)
            self.oppo_debate_tree = DebateTree(motion=motion, side=self.oppo_side)

        self.claim_pool, self.oppo_claim_pool = [], []
        self.prepared_tree_list, self.prepared_oppo_tree_list = None, None

        self.embedding_cache = {}

        if self.add_retrieval_feedback:
            data_list = tree_data_list
            for data in data_list:
                data["pro_debate_tree_obj"] = DebateTree.from_json(data["pro_debate_tree"])
                data["con_debate_tree_obj"] = DebateTree.from_json(data["con_debate_tree"])
            self.data_list = data_list
            self.pro_embeddings = [
                torch.tensor([x["pro_embedding_level_1"] for x in data_list]),
                torch.tensor([x["pro_embedding_level_2"] for x in data_list]),
                torch.tensor([x["pro_embedding_level_3"] for x in data_list]),
            ]
            self.con_embeddings = [
                torch.tensor([x["con_embedding_level_1"] for x in data_list]),
                torch.tensor([x["con_embedding_level_2"] for x in data_list]),
                torch.tensor([x["con_embedding_level_3"] for x in data_list]),
            ]

        self.used_evidence = set()

        self._streaming_input_env = None
        self._streaming_listen_thread = None

    def start_streaming_listen(
        self,
        watch_dir: Path,
        stage: str,
        *,
        min_audio_seconds: float = 30.0,
        min_text_words: int = 50,
        poll_interval: float = 1.0,
        audio_format: str = "mp3",
        max_audio_wait_seconds: Optional[float] = None,
        max_text_wait_seconds: Optional[float] = None,
        max_total_audio_seconds: Optional[float] = None,
        playback_cursor: Optional[list] = None,
    ) -> None:
        """Run :class:`StreamingInputEnv` on a thread (chunk audio → ASR → ``_analyze_statement``)."""
        from streaming.env import StreamingInputConfig, StreamingInputEnv

        self.stop_streaming_listen(join_timeout=5.0)

        watch_dir = Path(watch_dir)
        cfg = StreamingInputConfig(
            watch_dir=watch_dir,
            motion=self.motion,
            stage=stage,
            statement_side=self.oppo_side,
            min_audio_seconds=min_audio_seconds,
            min_text_words=min_text_words,
            poll_interval=poll_interval,
            audio_file_glob=f"*.{audio_format}",
            max_audio_wait_seconds=max_audio_wait_seconds,
            max_text_wait_seconds=max_text_wait_seconds,
            max_total_audio_seconds=max_total_audio_seconds,
            audio_format=audio_format,
            playback_cursor=playback_cursor,
        )
        self._streaming_input_env = StreamingInputEnv(self, cfg)
        self._streaming_listen_thread = threading.Thread(
            target=self._streaming_input_env.run,
            name="StreamingInputListen",
            daemon=True,
        )
        self._streaming_listen_thread.start()

    def stop_streaming_listen(self, join_timeout: float = 300.0) -> None:
        t = self._streaming_listen_thread
        env = self._streaming_input_env
        self._streaming_listen_thread = None
        self._streaming_input_env = None
        if env is not None:
            env.stop()
        if t is not None:
            t.join(timeout=join_timeout)

    def _get_evidence(self, claim):
        if self.use_retrieval:
            evidence = [x for x in claim["retrieved_evidence"] if "PDF" not in x["title"]]
            for e in evidence:
                if "score" in e:
                    e.pop("score")
                e["content"] = e["content"].replace("\n", " ")
                if "raw_content" in e and e["raw_content"] is not None:
                    e["raw_content"] = e["raw_content"].replace("\n", " ")[:2048]
                    e["raw_content"] = re.sub(r"https?://\S+", "", e["raw_content"])
                if "url" in e:
                    e.pop("url")
        else:
            evidence = claim.get("arguments", [])
        return evidence

    def claim_generation(self, pool_size, definition=None, **kwargs):
        """
        Generate the claim pool for the debater
        """
        if self.pool_file is not None and os.path.exists(self.pool_file):
            with open(self.pool_file, "r") as file:
                self.claim_pool = json.load(file)[:8]
            oppo_pool_file = self.pool_file.replace(f"pool_{self.side}", f"pool_{self.oppo_side}")
            with open(oppo_pool_file, "r") as file:
                self.oppo_claim_pool = json.load(file)[:8]
            self.definition = self.claim_pool[0][0].get("definition", None)
        else:
            logger.info(f"Starting to create a pool of size {pool_size}")
            motion = self.motion
            if definition is not None:
                motion = motion + "\nYour definition is: " + definition

            for side in ["for", "against"]:
                claim_workspace = ClaimPool(
                    motion=motion, side=side, model=self.config.model, pool_size=pool_size, **kwargs
                )
                claim_pool = claim_workspace.create_claim(need_score=True, need_evidence=(side == self.side))
                # print(pool)
                logger.info(f"Claim Pool Size: {len(self.claim_pool)}")
                save_file_name = f"{self.motion}_pool_{side}.json".replace(" ", "_").lower()
                with open(save_file_name, "w") as file:
                    json.dump(self.claim_pool, file, indent=2)

                if side == self.side:
                    self.claim_pool = claim_pool
                else:
                    self.oppo_claim_pool = claim_pool

            prompt = propose_definition_prompt.format(motion=self.motion, act=self.act)
            log_llm_io(
                logger,
                phase="ouragents_prepare",
                title="Definition-Helper-Prompt",
                body=prompt.strip(),
                stage=getattr(self, "status", None),
                side=getattr(self, "side", None),
            )
            response = self.helper_client(prompt=prompt)[0]
            log_llm_io(
                logger,
                phase="ouragents_prepare",
                title="Definition-Helper-Response",
                body=response.strip(),
                stage=getattr(self, "status", None),
                side=getattr(self, "side", None),
            )
            if "None" in response:
                self.definition = None
            else:
                self.definition = response.strip()

    def claim_selection(self, history=None):
        # NOTE: claim selection by overall framework, not sure if it is good
        if history and len(history) > 0:
            context = history[-1]["content"]
        else:
            context = ""
        main_claims, group_idx, thoughts = build_logic_claims(
            self.helper_client,
            self.motion,
            self.side,
            self.claim_pool,
            context=context,
            definition=self.definition,
            use_rehearsal_tree=self.use_rehearsal_tree,
        )
        # main_claims, group_idx, thoughts = build_cot_claims(self.helper_client, self.motion, self.side, self.claim_pool)

        self.main_claims = [self.claim_pool[idx][0] for idx in group_idx]
        self.main_claims_content = [self.claim_pool[idx][0]["claim"] for idx in group_idx]
        logger.debug(f"[Claim-Selection] Selected Claims: {main_claims}")
        self.build_evidence_pool()

        self.debate_thoughts.append(thoughts)

        # Only build prepared tree list if rehearsal tree is enabled
        if self.use_rehearsal_tree:
            self.prepared_tree_list = self._get_prepared_tree(self.side)
        else:
            self.prepared_tree_list = None

        return self.claim_pool, self.main_claims

    def build_evidence_pool(self):
        self.evidence_pool = [self._get_evidence(x) for x in self.main_claims]
        self.evidence_pool = sum(self.evidence_pool, [])
        high_quality_evidence_pool = rank_evidence(self.evidence_pool)
        high_quality_evidence_pool = [x for x in high_quality_evidence_pool if x.get("reliability", 0) >= 1]
        logger.debug(f"High-Quality Evidence Pool with reliability >= 1 Size: {len(high_quality_evidence_pool)}")
        self.evidence_pool = high_quality_evidence_pool[:10]
        self.high_quality_evidence_pool = high_quality_evidence_pool

    def _add_additional_info(self, prompt, history, **kwargs):
        tips = ""

        # add debate flow tree related tips if debate flow tree is enabled, if no rehearsal tree, it will be empty
        if self.status != "closing" and self.use_debate_flow_tree:
            actions = get_actions_from_tree(self.main_claims_content, self.debate_tree, self.oppo_debate_tree)
            action_str = ""
            for action in actions:
                action["prepared_materials"] = self._retrieve_on_prepared_tree(action).strip()
            battlefields = get_battlefields_from_actions(
                self.helper_client,
                self.motion,
                self.side,
                self.main_claims_content,
                actions,
                self.debate_tree,
                self.oppo_debate_tree,
            )
            battlefields = sorted(
                battlefields,
                key=lambda x: (sort_by_importance(x["battlefield_importance"]), len(x["actions"])),
                reverse=True,
            )

            battlefield_str = "Allocate time to the most important battlefields first. Present each battlefield as a complete unit. \n\n"
            used_actions = set()
            for battlefield in battlefields:
                actions = []
                for action in battlefield["actions"]:
                    if action["idx"] in used_actions:
                        continue
                    used_actions.add(action["idx"])
                    actions.append(action)

                action_str = ""
                for action in actions:
                    action_type = action["action"]
                    target_claim = action["target_claim"]
                    target_argument = (
                        action["target_argument"] if action_type != "propose" else action["prepared_materials"]
                    )
                    action_str += (
                        "\n\t" + f'*{action_type}* the claim: "{target_claim}" and the argument: "{target_argument}"'
                    )
                battlefield_str += (
                    f"**Battlefield Importance**: {battlefield['battlefield_importance']}\n"
                    f"**Battlefield**: {battlefield['battlefield']}\n"
                    f"**Battlefield Rationale**: {battlefield['battlefield_argument']}\n"
                    f"**Actions**:{action_str}\n"
                )
                battlefield_str += "\n"
            tips += "\n\n" + battlefield_str

        prompt = prompt.replace("{tips}", tips + "\n\n")
        return prompt

    def opening_generation(self, history, max_time, time_control=False, **kwargs):
        self.status = "opening"
        self.listen(history)
        max_words = math.ceil(max_time / WORDRATIO["time"])

        self.claim_selection(history)

        opening_thoughts = [x for x in self.debate_thoughts if x["mode"] == "choose_main_claims"]
        framework, explanation = opening_thoughts[-1]["framework"], (
            opening_thoughts[-1]["explanation"] if opening_thoughts else ("", "")
        )

        # Only include tree information if debate flow tree is enabled
        if self.use_debate_flow_tree:
            tree = self.debate_tree.print_tree(include_status=True)
            oppo_tree = self.oppo_debate_tree.print_tree(include_status=True, reverse=True)
            prompt = expert_opening_prompt_2.format(
                motion=self.motion,
                act=self.act,
                claims="* " + "\n* ".join(self.main_claims_content),
                tree=tree,
                oppo_tree=oppo_tree,
                framework=framework,
                explanation=explanation,
            )
        else:
            # Use a simplified prompt without tree information
            prompt = expert_opening_prompt_2.format(
                motion=self.motion,
                act=self.act,
                claims="* " + "\n* ".join(self.main_claims_content),
                tree="",
                oppo_tree="",
                framework=framework,
                explanation=explanation,
            )

        prompt = prompt.replace("{n_words}", str(max_words))

        if self.side == "for":
            prompt = prompt.replace("{definition}", "**Your Definition of the Motion**: \n" + self.definition + "\n\n")
        else:
            prompt = prompt.replace("{definition}", "")

        prompt = self._add_additional_info(prompt, history, **kwargs)

        response = self.speak(prompt, max_time=max_time, time_control=time_control, history=history, **kwargs)
        if self.use_debate_flow_tree:
            self._analyze_statement(response, self.side)
        return response

    def rebuttal_generation(self, history, max_time, time_control=False, **kwargs):
        self.status = "rebuttal"
        self.listen(history)
        max_words = math.ceil(max_time / WORDRATIO["time"])

        # Only include tree information if debate flow tree is enabled
        if self.use_debate_flow_tree:
            your_tree = self.debate_tree.print_tree(include_status=True)
            oppo_tree = self.oppo_debate_tree.print_tree(include_status=True, reverse=True)
            prompt = expert_rebuttal_prompt_2.format(
                motion=self.motion, act=self.act, counter_act=self.counter_act, tree=your_tree, oppo_tree=oppo_tree
            )
        else:
            # Use a simplified prompt without tree information
            prompt = expert_rebuttal_prompt_2.format(
                motion=self.motion, act=self.act, counter_act=self.counter_act, tree="", oppo_tree=""
            )

        prompt = prompt.replace("{n_words}", str(max_words))

        prompt = self._add_additional_info(prompt, history, **kwargs)

        response = self.speak(prompt, max_time=max_time, time_control=time_control, history=history, **kwargs)
        if self.use_debate_flow_tree:
            self._analyze_statement(response, self.side)
        return response

    def closing_generation(self, history, max_time, time_control=False, **kwargs):
        self.status = "closing"
        self.listen(history)
        max_words = math.ceil(max_time / WORDRATIO["time"])

        # Only include tree information if debate flow tree is enabled
        if self.use_debate_flow_tree:
            your_tree = self.debate_tree.print_tree(include_status=True)
            oppo_tree = self.oppo_debate_tree.print_tree(include_status=True, reverse=True)
            prompt = expert_closing_prompt_2.format(
                act=self.act, counter_act=self.counter_act, tree=your_tree, oppo_tree=oppo_tree
            )
        else:
            # Use a simplified prompt without tree information
            prompt = expert_closing_prompt_2.format(act=self.act, counter_act=self.counter_act, tree="", oppo_tree="")

        prompt = prompt.replace("{n_words}", str(max_words))

        prompt = self._add_additional_info(prompt, history, **kwargs)

        response = self.speak(prompt, max_time=max_time, time_control=time_control, history=history, **kwargs)
        response = response.split("**Reference**")[0].strip()
        if self.use_debate_flow_tree:
            self._analyze_statement(response, self.side)
        return response

    def speak(self, prompt, max_time, time_control=False, history=None, **kwargs):
        call_id = next_call_id()
        ctx = dict(call_id=call_id, stage=self.status, side=self.side)
        set_speak_io_context(call_id, "tree_debater_speak")
        try:
            with timed_phase(logger, "tree_debater_speak", **ctx):
                self._add_message("user", prompt)
                if io_logging_enabled():
                    log_io_block(
                        io_logger,
                        call_id=call_id,
                        phase="tree_debater_speak",
                        title="Conversation-History",
                        body=json.dumps(self.conversation),
                        stage=self.status,
                        side=self.side,
                    )
                    log_io_block(
                        io_logger,
                        call_id=call_id,
                        phase="tree_debater_speak",
                        title="Prompt",
                        body=prompt,
                        stage=self.status,
                        side=self.side,
                    )
                else:
                    log_llm_io(
                        logger,
                        phase="tree_debater_speak",
                        title="Conversation-History",
                        body=json.dumps(self.conversation),
                        stage=self.status,
                        side=self.side,
                    )
                    log_llm_io(
                        logger,
                        phase="tree_debater_speak",
                        title="Prompt",
                        body=prompt.strip(),
                        stage=self.status,
                        side=self.side,
                    )
                logger.debug(
                    f"[timing-meta] call_id={call_id} speak_session=tree_debater_speak "
                    f"n_messages={len(self.conversation)}"
                )

                with timed_phase(logger, "main_get_response", **ctx):
                    response = self._get_response(self.conversation, **kwargs)
                if io_logging_enabled():
                    log_io_block(
                        io_logger,
                        call_id=call_id,
                        phase="tree_debater_speak",
                        title="Response-Before-Post-Process",
                        body=str(response).strip(),
                        stage=self.status,
                        side=self.side,
                    )
                else:
                    log_llm_io(
                        logger,
                        phase="tree_debater_speak",
                        title="Response-Before-Post-Process",
                        body=str(response).strip(),
                        stage=self.status,
                        side=self.side,
                    )

                with timed_phase(logger, "revision_suggestion", pass_index=1, add_evidence=True, **ctx):
                    feedback_for_revision, new_evidence, allocation_plan, ori_statement = self._get_revision_suggestion(
                        statement=response, history=history, add_evidence=True, call_id=call_id, **kwargs
                    )
                with timed_phase(logger, "length_adjust", block=1, max_retry=1, **ctx):
                    response = self._length_adjust(
                        ori_statement,
                        feedback_for_revision,
                        new_evidence,
                        allocation_plan,
                        max_time,
                        max_retry=1,
                        call_id=call_id,
                        **kwargs,
                    )

                with timed_phase(logger, "revision_suggestion", pass_index=2, add_evidence=False, **ctx):
                    feedback_for_revision, new_evidence, _, _ = self._get_revision_suggestion(
                        statement=response, history=history, add_evidence=False, call_id=call_id, **kwargs
                    )

                streaming_tts = kwargs.get("streaming_tts", getattr(self.config, "streaming_tts", False))
                if not time_control or streaming_tts:
                    with timed_phase(logger, "length_adjust", block=2, max_retry=1, **ctx):
                        response = self._length_adjust(
                            response,
                            feedback_for_revision,
                            new_evidence,
                            allocation_plan,
                            max_time,
                            max_retry=1,
                            call_id=call_id,
                            **kwargs,
                        )
                else:
                    with timed_phase(logger, "length_adjust", block=2, max_retry=10, **ctx):
                        response = self._length_adjust(
                            response,
                            feedback_for_revision,
                            new_evidence,
                            allocation_plan,
                            max_time,
                            max_retry=10,
                            call_id=call_id,
                            **kwargs,
                        )

                with timed_phase(logger, "post_process", **ctx):
                    out = super().post_process(response, max_time, time_control, **kwargs)
                return out
        finally:
            clear_speak_io_context()

    def listen(self, history):
        if len(history) == 0:
            return
        assert history[-1]["side"] == self.oppo_side, "The opponent should be the last speaker"

        content = f"**Opponent's {history[-1]['stage'].title()} Statement**\n" + history[-1]["content"]
        self._add_message("user", content)

        # Only analyze statement if debate flow tree is enabled
        if self.use_debate_flow_tree:
            skip_full = getattr(self.config, "streaming_listen", False) and history[-1].get(
                "tree_via_streaming"
            ) is True
            if not skip_full:
                st = history[-1]["stage"]
                with timed_phase(
                    logger,
                    "listen_analyze_statement",
                    stage=st,
                    side=self.side,
                    opponent_side=self.oppo_side,
                ):
                    logger.debug(
                        f"[BatchListener] analyze_start stage={st} side={self.side} "
                        f"opponent_side={self.oppo_side} t={time.time():.3f}"
                    )
                    self._analyze_statement(history[-1]["content"], self.oppo_side)
                    logger.debug(f"[BatchListener] analyze_end stage={st} side={self.side} t={time.time():.3f}")

        # Only prepare opponent tree list if both debate flow tree and rehearsal tree are enabled
        if self.use_rehearsal_tree and self.prepared_oppo_tree_list is None:
            self.prepared_oppo_tree_list = self._get_prepared_tree(self.oppo_side)
        else:
            self.prepared_oppo_tree_list = None

    def _get_feedback_from_audience(self, statement, history, **kwargs):
        extra_tree_info = ""
        if self.add_retrieval_feedback and self.use_debate_flow_tree:
            with timed_phase(
                logger,
                "audience_exemplar_retrieval",
                stage=self.status,
                side=self.side,
            ):
                retrieval, retrieval_feedback = self._get_retrieval_debate_tree(include_points=False)
            if retrieval is not None:
                extra_tree_info += "\n\n" + retrieval_feedback

        history_str = ""
        for h in history:
            side = f"Opponent ({self.oppo_side})" if h["side"] == self.oppo_side else f"You ({self.side})"
            history_str += f"*{side}'s {h['stage'].title()} Statement*\t" + h["content"].replace("\n", " ") + "\n\n"
        prompt = audience_feedback_prompt.format(
            motion=self.motion,
            side=self.side,
            stage=self.status.title(),
            statement=statement,
            retrieval=extra_tree_info,
            history=history_str,
        )
        call_id = kwargs.get("call_id")
        if io_logging_enabled() and call_id is not None:
            log_io_block(
                io_logger,
                call_id=call_id,
                phase="audience_feedback",
                title="Audience-Feedback-Prompt",
                body=prompt.strip(),
                stage=self.status,
                side=self.side,
            )
        else:
            log_llm_io(
                logger,
                phase="audience_feedback",
                title="Audience-Feedback-Prompt",
                body=prompt.strip(),
                stage=self.status,
                side=self.side,
            )
        audience_feedback = []
        flat_audience_feedback = ""
        with timed_phase(logger, "audience_simulated_feedback_llm", stage=self.status, side=self.side, n_audience=len(self.simulated_audience)):
            for i, au in enumerate(self.simulated_audience):
                feedback = au.feedback(prompt)
                audience_feedback.append(feedback)
                key_feedback = (
                    "Critical Issues and Minimal Revision Suggestions"
                    + feedback.split("Critical Issues and Minimal Revision Suggestions")[-1]
                )
                flat_audience_feedback += f"\n\n\nAudience {i+1} Feedback:\n" + key_feedback
        if io_logging_enabled() and call_id is not None:
            log_io_block(
                io_logger,
                call_id=call_id,
                phase="audience_feedback",
                title="Audience-Feedback-Response",
                body=flat_audience_feedback.strip(),
                stage=self.status,
                side=self.side,
            )
        else:
            log_llm_io(
                logger,
                phase="audience_feedback",
                title="Audience-Feedback-Response",
                body=flat_audience_feedback.strip(),
                stage=self.status,
                side=self.side,
            )
        return flat_audience_feedback, audience_feedback

    def _get_retrieval_debate_tree(self, **kwargs):
        if self.debate_tree.get_all_nodes() == []:
            current_tree_info = self.motion
        else:
            current_tree_info = self.debate_tree.print_tree(include_status=False, meta_info=False)
        logger.debug(
            f"[Retrieval-Debate-Tree] Search for {self.side} side: " + current_tree_info.strip().replace("\\n", " ||| ")
        )
        t_embed = time.perf_counter()
        current_tree_embedding = self._get_embedding_from_cache(current_tree_info)
        log_timing(
            logger,
            "exemplar_retrieval_query_embedding",
            time.perf_counter() - t_embed,
            stage=self.status,
            side=self.side,
        )
        memory_tree_embedding = self.pro_embeddings if self.side == "for" else self.con_embeddings
        if self.status == "opening":
            memory_tree_embedding = memory_tree_embedding[0]
        elif self.status == "rebuttal":
            memory_tree_embedding = memory_tree_embedding[1]
        elif self.status == "closing":
            memory_tree_embedding = memory_tree_embedding[2]

        t_search = time.perf_counter()
        hits = semantic_search(
            torch.tensor([current_tree_embedding]),
            torch.tensor(memory_tree_embedding),
            score_function=dot_score,
            top_k=1,
        )[0]
        log_timing(
            logger,
            "exemplar_retrieval_semantic_search",
            time.perf_counter() - t_search,
            stage=self.status,
            side=self.side,
            top_k=1,
        )
        retrieval_idx = [x["corpus_id"] for x in hits]
        retrieval_data = [self.data_list[idx] for idx in retrieval_idx]
        retrieval_motion = [data["motion"] for data in retrieval_data]
        retrieval_similarity = [x["score"] for x in hits]
        retrieval_tree = [
            data["pro_debate_tree_obj"] if self.side == "for" else data["con_debate_tree_obj"]
            for data in retrieval_data
        ]
        retrieval_tree_info = [tree.print_tree(include_status=False) for tree in retrieval_tree]
        retrieval_stage_statement = [
            x
            for data in retrieval_data
            for x in data["structured_arguments"]
            if x["stage"] == self.status and x["side"] == self.side
        ]
        logger.debug(
            f"[Retrieval-Debate-Tree] Retrieval Index: {retrieval_idx}, Retrieval Similarity: {retrieval_similarity}, Retrieval Motion: {retrieval_motion}"
        )
        logger.debug(f"[Retrieval-Debate-Tree] Retrieval Tree Info: {retrieval_tree_info}")

        retrieval = [
            {
                "idx": idx,
                "motion": motion,
                "side": self.side,
                "similarity": score,
                "tree_info": tree_info,
                "stage_statement": stage_statement,
            }
            for idx, score, motion, tree_info, stage_statement in zip(
                retrieval_idx, retrieval_similarity, retrieval_motion, retrieval_tree_info, retrieval_stage_statement
            )
        ]

        retrieval_feedback = ""
        for ex in retrieval:
            point_str = ""
            if kwargs.get("include_points", False):
                for x in ex["stage_statement"]["claims"]:
                    point_str += f"**Claim:** {x['claim']}\n"
                    point_str += f"**Purpose:** "
                    for y in x["purpose"]:
                        point_str += f"{y['action']} => {y['target']};"
                    point_str += "\n"
                    point_str += f"**Content:** {x['content']}\n"
                    point_str += f"**Argument:** {' '.join(x['arguments'])}\n\n"
            point_str = point_str.strip()
            retrieval_feedback += f"**Examplar Motion:** {ex['motion']}\n"
            retrieval_feedback += f"**Examplar Side:** {ex['side']}\n"
            retrieval_feedback += f"**Examplar Debate Flow Tree** {ex['tree_info']}\n"
            if point_str != "":
                retrieval_feedback += f"**Examplar Stage Statement** \n{point_str}\n"
            retrieval_feedback += "===================================\n"

        retrieval_feedback += "\n"

        thoughts = {
            "stage": self.status,
            "side": self.side,
            "mode": "retrieval",
            "retrieval": retrieval,
            "retrieval_feedback": retrieval_feedback,
        }
        self.debate_thoughts.append(thoughts)

        return retrieval, retrieval_feedback

    def _get_prepared_tree(self, side):
        prepared_tree = []
        if side == self.side:
            for x in self.main_claims:
                data = x["tree_structure"]
                tree = PrepareTree.from_json(data)
                prepared_tree.append(tree)
        else:
            # opponent's tree
            match_trees = []
            if self.oppo_debate_tree.max_level > 0:
                for x in self.oppo_claim_pool:
                    data = x[0]["tree_structure"]
                    tree = PrepareTree.from_json(data)
                    root_claim = tree.root.claim
                    match_node, similarity = self.oppo_debate_tree.get_most_similar_node(
                        root_claim, side=side, level=1, top_k=1, threshold=0.8
                    )
                    if match_node is not None:
                        match_trees.append((tree, similarity, match_node.claim))

            sorted_match_trees = sorted(match_trees, key=lambda x: x[1], reverse=True)
            for i in range(min(len(sorted_match_trees), 3)):
                prepared_tree.append(sorted_match_trees[i][0])
                similarity = sorted_match_trees[i][1]
                query_claim = sorted_match_trees[i][2]

                log_llm_io(
                    logger,
                    phase="get_prepared_tree",
                    title=f"Opponent's Tree (similarity: {similarity:0.2f}) for claim: {query_claim}",
                    body=tree.print_tree(include_status=True),
                )

        thoughts = {
            "stage": self.status,
            "side": self.side,
            "mode": "get_prepared_tree",
            "prepared_tree": [t.root.claim for t in prepared_tree],
        }
        self.debate_thoughts.append(thoughts)

        return prepared_tree

    def _retrieve_on_prepared_tree(self, action):
        # Skip retrieval if rehearsal tree is disabled
        if not self.use_rehearsal_tree:
            return ""

        # retrieve similar action from the prepared tree
        target_claim = action["target_claim"]
        action_type = action["action"]
        look_ahead_num = REMAINING_ROUND_NUM[f"{self.status}_{self.side}"]
        query_embedding = self._get_embedding_from_cache(target_claim)

        with timed_phase(
            logger,
            "rehearsal_retrieve_on_prepared_tree",
            stage=self.status,
            side=self.side,
            action_type=action_type,
        ):
            additional_info, retrieval_nodes = get_retrieval_from_rehearsal_tree(
                action_type,
                target_claim,
                self.side,
                self.oppo_side,
                self.prepared_tree_list,
                self.prepared_oppo_tree_list,
                look_ahead_num,
                query_embedding,
            )

        thoughts = {
            "stage": self.status,
            "side": self.side,
            "mode": "retrieve_on_prepared_tree",
            "action_type": action_type,
            "target_claim": target_claim,
            "retrieval_nodes": retrieval_nodes,
            "additional_info": additional_info,
        }
        self.debate_thoughts.append(thoughts)

        return "\n".join(additional_info)

    def _get_revision_suggestion(self, statement, history, add_evidence=True, call_id=None, **kwargs):
        statement = statement.replace("**Statement:**", "**Statement**").replace("**Statement**:", "**Statement**")
        parts = statement.split("**Statement**")
        if len(parts) > 1:
            allocation_plan = parts[0].strip()
            statement = parts[1].strip()
        else:
            allocation_plan = ""
            statement = statement.strip()

        if self.status == "closing":
            return "", "", allocation_plan, statement

        feedback_from_audience, audience_feedback = self._get_feedback_from_audience(
            statement, history, call_id=call_id, **kwargs
        )
        feedback_for_revision = f"Revision Guidance:\n{feedback_from_audience}"

        new_evidence = []
        selected_ids = []
        if add_evidence:
            selected_ids = []
            # new_evidence = self._retrieve_new_evidence(f"The current statement is: \n{statement}\n. We have the following feedback:\n{feedback_for_revision}\n. Your searched new evidence should be very suitable and helpful for the revision.")
            new_evidence = [x for x in self.high_quality_evidence_pool if x["id"] not in self.used_evidence]
            selected_ids = [x["id"] for x in new_evidence]
            if len(new_evidence) > 10:
                evidence_str = json.dumps([{k: v for k, v in x.items() if k != "raw_content"} for x in new_evidence])
                prompt = evidence_selection_prompt.format(
                    motion=self.motion,
                    side=self.side,
                    stage=self.status,
                    evidence=evidence_str,
                    statement=statement,
                    feedback=feedback_for_revision,
                )
                if io_logging_enabled() and call_id is not None:
                    log_io_block(
                        io_logger,
                        call_id=call_id,
                        phase="evidence_selection",
                        title="Evidence-Selection-Prompt",
                        body=prompt.strip(),
                        stage=self.status,
                        side=self.side,
                    )
                else:
                    log_llm_io(
                        logger,
                        phase="evidence_selection",
                        title="Evidence-Selection-Prompt",
                        body=prompt.strip(),
                        stage=self.status,
                        side=self.side,
                    )
                with timed_phase(logger, "evidence_selection_llm", stage=self.status, side=self.side):
                    selected_ids, response = get_response_with_retry(self.helper_client, prompt, "selected_ids")
                if io_logging_enabled() and call_id is not None:
                    log_io_block(
                        io_logger,
                        call_id=call_id,
                        phase="evidence_selection",
                        title="Evidence-Selection-Response",
                        body=response.strip(),
                        stage=self.status,
                        side=self.side,
                    )
                else:
                    log_llm_io(
                        logger,
                        phase="evidence_selection",
                        title="Evidence-Selection-Response",
                        body=response.strip(),
                        stage=self.status,
                        side=self.side,
                    )
                new_evidence = [
                    e for e in new_evidence if e["id"] in selected_ids and e["id"] not in self.used_evidence
                ]
                if len(new_evidence) != len(selected_ids):
                    logger.warning(
                        f"[Get-Expert-Audience-Revision-Evidence-Selection] Select {selected_ids}, finally {len(new_evidence)}"
                    )
                logger.debug(
                    f"[Get-Expert-Audience-Revision-Evidence-Selection] From {len(self.high_quality_evidence_pool)} evidence select {len(selected_ids)} evidence: {selected_ids}"
                )

            self.used_evidence.update(selected_ids)
            logger.debug(f"[Used-Evidence] {self.used_evidence}")

        self.debate_thoughts.append(
            {
                "stage": self.status,
                "side": self.side,
                "mode": "revision",
                "original_statement": statement,
                "allocation_plan": allocation_plan,
                "simulated_audience_feedback": audience_feedback,
                "feedback_for_revision": feedback_for_revision,
                "selected_evidence_id": selected_ids,
            }
        )

        return feedback_for_revision, new_evidence, allocation_plan, statement

    def _length_adjust(
        self, statement, feedback_for_revision, new_evidence, allocation_plan, max_time, max_retry=10, **kwargs
    ):
        call_id = kwargs.pop("call_id", None)
        budget, threshold = max_time, TIME_TOLERANCE
        time_adjuster = TimeAdjuster()
        estimator = LengthEstimator(mode=TIME_MODE_FOR_STATEMENT)
        ratio = WORDRATIO[TIME_MODE_FOR_STATEMENT]
        n_words = math.ceil(max_time / WORDRATIO["time"])

        flag = False
        retry = 0
        response_list = []
        while not flag and retry < max_retry:
            iter_t0 = time.perf_counter()
            evidence_str = json.dumps([{k: v for k, v in x.items() if k != "raw_content"} for x in new_evidence])
            prompt = post_process_prompt.format(
                motion=self.motion,
                side=self.side,
                stage=self.status,
                evidence=evidence_str,
                statement=statement,
                feedback=feedback_for_revision,
                max_words=n_words,
                allocation_plan=allocation_plan,
            )

            if io_logging_enabled() and call_id is not None:
                log_io_block(
                    io_logger,
                    call_id=call_id,
                    phase="length_adjust",
                    title=f"Get-Expert-Audience-Revision-Prompt_iter{retry + 1}",
                    body=prompt.strip(),
                    stage=self.status,
                    side=self.side,
                    iteration=retry + 1,
                )
            else:
                log_llm_io(
                    logger,
                    phase="length_adjust",
                    title=f"Get-Expert-Audience-Revision-Prompt_iter{retry + 1}",
                    body=prompt.strip(),
                    stage=self.status,
                    side=self.side,
                    iteration=retry + 1,
                )
            revision = self.helper_client(prompt=prompt)[0]
            if io_logging_enabled() and call_id is not None:
                log_io_block(
                    io_logger,
                    call_id=call_id,
                    phase="length_adjust",
                    title=f"Get-Expert-Audience-Revision-Response_iter{retry + 1}",
                    body=revision.strip(),
                    stage=self.status,
                    side=self.side,
                    iteration=retry + 1,
                )
            else:
                log_llm_io(
                    logger,
                    phase="length_adjust",
                    title=f"Get-Expert-Audience-Revision-Response_iter{retry + 1}",
                    body=revision.strip(),
                    stage=self.status,
                    side=self.side,
                    iteration=retry + 1,
                )
            new_statement = revision.replace("Revised Statement:\n", "")
            new_statement = new_statement.replace("et al.,", "")
            new_statement = new_statement.replace("[X]", "")
            response = re.sub(r" [X-Z][ \%]", "", new_statement)

            if io_logging_enabled() and call_id is not None:
                log_io_block(
                    io_logger,
                    call_id=call_id,
                    phase="length_adjust",
                    title=f"Response-After-Post-Process_iter{retry + 1}",
                    body=response.strip(),
                    stage=self.status,
                    side=self.side,
                    iteration=retry + 1,
                )
            else:
                log_llm_io(
                    logger,
                    phase="length_adjust",
                    title=f"Response-After-Post-Process_iter{retry + 1}",
                    body=response.strip(),
                    stage=self.status,
                    side=self.side,
                    iteration=retry + 1,
                )
            current_cost, n_words, flag = time_adjuster.revise_helper(
                response, n_words, budget, threshold=threshold, ratio=ratio, estimator=estimator
            )
            log_timing(
                logger,
                "length_adjust_iteration",
                time.perf_counter() - iter_t0,
                stage=self.status,
                side=self.side,
                iteration=retry + 1,
                max_retry=max_retry,
                call_id=call_id,
                fit_ok=flag,
                current_cost=current_cost,
            )
            response_list.append([response, current_cost])
            retry += 1
            if not flag and max_retry > 1:
                logger.info(f"[Efficient-Fit-Length] Retry {retry} times. Next words: {n_words}")
            else:
                if max_retry > 1:
                    logger.info(f"[Efficient-Fit-Length] Success in {retry} times.")
                else:
                    logger.info(f"[Efficient-Fit-Length] No retry. The cost is {current_cost}.")
                    flag = True
                break

        if retry >= max_retry and not flag:
            logger.warning(f"[Efficient-Fit-Length] Failed to fit the length in {max_retry} times.")
            longest_response_id = max(enumerate(response_list), key=lambda x: x[1][1] if x[1][1] <= budget else 0)[0]
            response = response_list[longest_response_id][0]
            current_cost = response_list[longest_response_id][1]
            logger.info(f"[Efficient-Fit-Length] Reach the maximum retry times {retry}. The cost is {current_cost}. ")
            # thought_idx = len(response_list) - longest_response_id
            # thoughts = self.debate_thoughts[-thought_idx]

        thoughts = {
            "stage": self.status,
            "side": self.side,
            "mode": "length_adjust",
            "response_list": response_list,
            "n_trials": len(response_list),
            "final_response": response,
            "final_cost": current_cost,
        }
        self.debate_thoughts.append(thoughts)

        return response

    def _get_embedding_from_cache(self, content: str):
        if content in self.embedding_cache:
            return self.embedding_cache[content]

        max_retry = 3
        retry = 0
        while retry < max_retry:
            try:
                t0 = time.perf_counter()
                embedding = get_embeddings([content])[0]
                log_timing(
                    logger,
                    "embedding_api_fetch",
                    time.perf_counter() - t0,
                    stage=self.status,
                    side=self.side,
                    cache_hit=False,
                    attempt=retry + 1,
                )
                break
            except Exception as e:
                logger.error(f"[Get-Embedding-From-Cache] Error: {e}. Sleep 30 seconds and retry.")
                time.sleep(30)
                retry += 1

        self.embedding_cache[content] = embedding
        return embedding

    def _analyze_statement(self, statements, statement_side):
        """
        Analyze the statements:
        1. Extract the claims from the statements
        2. Match the opponent's claims with the debater's claims
        3. Update the claim status
        """
        # Skip analysis if debate flow tree is disabled
        if not self.use_debate_flow_tree:
            return []

        if statement_side == self.side:
            tree, oppo_tree = self.debate_tree, self.oppo_debate_tree
        else:
            tree, oppo_tree = self.oppo_debate_tree, self.debate_tree
        with timed_phase(
            logger,
            "analyze_statement",
            stage=self.status,
            side=self.side,
            statement_side=statement_side,
        ):
            claims = extract_statement(
                self.helper_client,
                self.motion,
                statements,
                tree=[tree.print_tree(include_status=True), oppo_tree.print_tree(include_status=True, reverse=True)],
                side=statement_side,
                stage=self.status,
            )

            for x in claims:
                for p in x["purpose"]:
                    target_tree = tree if p["targeted_debate_tree"] == "you" else oppo_tree
                    if p["target"] == "N/A" and target_tree.max_level == 0:
                        if p["action"] == "propose" or p["action"] == "rebut" or p["action"] == "reinforce":
                            p["target"] = x["claim"]

            for x in claims:
                claim = x["claim"]
                arguments = x["arguments"]
                if isinstance(x["purpose"], dict):
                    purpose = [x["purpose"]]
                else:
                    purpose = x["purpose"]
                for p in purpose:
                    target_tree = tree if p["targeted_debate_tree"] == "you" else oppo_tree
                    action = p["action"]
                    target = p["target"]
                    target_tree.update_node(action, new_claim=claim, new_argument=arguments, target=target)

            thoughts = {
                "stage": self.status,
                "side": statement_side,
                "mode": "analyze_statement",
                "statement": statements,
                "claims": claims,
            }
            self.debate_thoughts.append(thoughts)

        return claims

    def reset_stage(self, stage, side, new_content, history):
        conversation = [x for x in self.conversation]
        self.conversation = []
        for x in conversation:
            if x["role"] == "system":
                self.conversation.append(x)
            elif x["role"] == "user":
                if x["content"].startswith("**Opponent's"):
                    self.conversation.append(x)
            elif x["role"] == "assistant":
                self.conversation.append(x)

        assert self.conversation[-1]["role"] == "assistant", "The last message should be an assistant message"
        self.conversation[-1]["content"] = new_content  # update the last assistant message

        # reset the debate flow tree
        self.debate_tree = DebateTree(motion=self.motion, side=self.side)
        self.oppo_debate_tree = DebateTree(motion=self.motion, side=self.oppo_side)
        if self.use_debate_flow_tree:
            for x in history:
                self._analyze_statement(x["content"], x["side"])
            self._analyze_statement(new_content, side)

        return
