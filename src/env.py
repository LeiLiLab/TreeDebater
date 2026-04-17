import argparse
import json
import time
from dataclasses import dataclass
from typing import List

import yaml

from agents import Audience, AudienceConfig, BaselineDebater, Debater, DebaterConfig, HumanDebater, Judge, JudgeConfig
from ouragents import TreeDebater
from utils.constants import CLOSING_TIME, OPENING_TIME, REBUTTAL_TIME
from utils.timing_log import log_timing
from utils.tool import logger


@dataclass
class EnvConfig:
    motion: str
    debater_config: List[DebaterConfig]
    judge_config: JudgeConfig
    audience_config: AudienceConfig
    judge_num: int
    audience_num: int
    claim_pool_size: int = 50
    reverse: bool = False
    time_control: bool = True


def extract_overall_score(obj_scores):  # larger is better
    return (
        -obj_scores["Logical Inconsistencies"]
        - obj_scores["Unsupported Assertions"]
        + obj_scores["Inferences"]
        + obj_scores["Statistics"]
        + obj_scores["Case Studies"]
        - obj_scores["Unanswered Arguments"]
    )


class Env:
    def __init__(self, config, debug) -> None:
        self.config = config
        self.motion = config.motion
        self.claim_pool_size = config.claim_pool_size
        self.reverse = config.reverse
        self.time_control = config.time_control
        self.debug = debug

        # init players
        self.debaters = {}
        for conf in config.debater_config:
            if conf.type == "default":
                self.debaters[conf.side] = Debater(conf, motion=self.motion)
            elif conf.type == "human":
                self.debaters[conf.side] = HumanDebater(conf, motion=self.motion)
            elif conf.type == "baseline":
                self.debaters[conf.side] = BaselineDebater(conf, motion=self.motion)
            elif conf.type == "treedebater":
                self.debaters[conf.side] = TreeDebater(conf, motion=self.motion)  # prompt-based with expert prompt
            else:
                raise ValueError(f"Type {conf.type} is not supported.")

        # init judge
        if config.judge_num > 1:
            print("Multiple judges are not supported yet.")
        self.judge = Judge(config.judge_config)

        # init audience
        self.audiences = [Audience(config.audience_config) for _ in range(config.audience_num)]

        self.debate_process = []
        self.debate_process.append(
            {
                "stage": "settings",
                "motion": self.motion,
                "debaters": {side: debater.config for side, debater in self.debaters.items()},
                "judges": self.judge.config,
                "audiences": [audience.config for audience in self.audiences],
            }
        )

    def play(self, pre_only=False):
        order = ["for", "against"] if not self.reverse else ["against", "for"]
        for stage in ["preparation", "opening", "rebuttal", "closing"]:
            logger.info(f"[{stage}] Start")
            t_stage = time.perf_counter()
            if stage == "preparation":
                for side in order:
                    if self.debaters[side].type in ["treedebater"]:
                        self.debaters[side].claim_generation(self.claim_pool_size, temperature=1)

            elif stage == "opening":
                for side in order:
                    player = self.debaters[side]
                    response = player.opening_generation(
                        history=self.debate_process[1:],
                        max_time=OPENING_TIME,
                        time_control=self.time_control,
                        streaming_tts=player.config.streaming_tts,
                    )
                    self.debate_process.append({"stage": stage, "side": side, "content": response})
            elif stage == "rebuttal":
                for side in order:
                    player = self.debaters[side]
                    response = player.rebuttal_generation(
                        history=self.debate_process[1:],
                        max_time=REBUTTAL_TIME,
                        time_control=self.time_control,
                        streaming_tts=player.config.streaming_tts,
                    )
                    self.debate_process.append({"stage": stage, "side": side, "content": response})
            elif stage == "closing":
                for side in order:
                    player = self.debaters[side]
                    response = player.closing_generation(
                        history=self.debate_process[1:],
                        max_time=CLOSING_TIME,
                        time_control=self.time_control,
                        streaming_tts=player.config.streaming_tts,
                    )
                    self.debate_process.append({"stage": stage, "side": side, "content": response})
            log_timing(logger, "env_stage_wall", time.perf_counter() - t_stage, stage=stage, motion=self.motion[:80])
            logger.info(f"[{stage}] Done")
            if self.debug:
                response = input("Press N to stop: ")
                if response.lower() == "n":
                    break

    def eval(self, process=None):
        logger.info("[Evaluation] Start")
        t0 = time.perf_counter()
        output = {}
        process = self.debate_process if process is None else process
        process = [x for x in process if x["stage"] != "settings"]

        # judge
        winner, comments = self.judge.eval(motion=self.motion, debate_process=process)
        output["judge_winner"] = winner
        output["judge_comment"] = comments

        side_info = {
            "for": {"content": [p["content"] for p in process if p["side"] == "for"], "claims": [], "surprises": []},
            "against": {
                "content": [p["content"] for p in process if p["side"] == "against"],
                "claims": [],
                "surprises": [],
            },
        }
        for side in ["for", "against"]:
            obj_scores, obj_scores_explanation = self.judge.finegrained_check(self.motion, side_info, side)
            output[f"{side}_objective_scores"] = extract_overall_score(obj_scores)
            output[f"{side}_objective_scores_explanation"] = obj_scores_explanation

        # audience
        output["audience_votes"] = []
        output["for_surprise"], output["against_surprise"] = [], []
        output["for_surprise_explanation"], output["against_surprise_explanation"] = [], []
        for audience in self.audiences:
            audience_votes = audience.vote(process, self.motion)
            output[f"audience_votes"].append(audience_votes)
            for side in ["for", "against"]:
                surprise_scores, surprise_explanation = audience.surprise(self.motion, side, side_info[side]["claims"])
                side_info[side]["surprises"] = surprise_scores[0]
                output[f"{side}_surprise"].append(
                    sum(side_info[side]["surprises"].values()) / len(side_info[side]["surprises"])
                )
                output[f"{side}_surprise_explanation"].append(surprise_explanation[0])

        log_timing(logger, "evaluation_wall", time.perf_counter() - t0, motion=self.motion[:80])
        logger.info("[Evaluation] Done")
        return output, side_info

    def compare_debate(self, comparison_process, order_reverse=False):
        logger.info("[Comparison Evaluation] Start")
        t_cmp = time.perf_counter()
        order = ["baseline_response", "test_response"] if not order_reverse else ["test_response", "baseline_response"]

        output = {}
        context = []
        for i, phase in enumerate(comparison_process.keys()):
            logger.info(f"[{phase}] Start")
            t_ph = time.perf_counter()
            output[phase] = {}
            stage, side = phase.split("_")

            versiona = comparison_process[phase][order[0]].split("**Reference**")[0]
            versionb = comparison_process[phase][order[1]].split("**Reference**")[0]

            # judge
            winner, comments = self.judge.comparison(
                motion=self.motion, context=context, side=side, a=versiona, b=versionb, max_tokens=200
            )
            output[phase]["judge_version"] = winner
            output[phase]["judge_version_comment"] = comments

            # audience
            output[phase]["audience_version"] = []
            output[phase]["audience_version_comment"] = []
            for audience in self.audiences:
                winner, comments = audience.comparison(
                    motion=self.motion, context=context, side=side, a=versiona, b=versionb, max_tokens=200
                )
                output[phase]["audience_version"].append(winner)
                output[phase]["audience_version_comment"].append(comments)

            # logger.info(f"[{phase}] Done. Sleep 5 seconds")
            # if phase != "closing_against":
            #     time.sleep(5)

            context.append(
                {
                    "stage": stage,
                    "side": side,
                    "content": comparison_process[phase]["keep_response"].split("**Reference**")[0],
                }
            )
            log_timing(logger, "comparison_phase_wall", time.perf_counter() - t_ph, phase=phase, motion=self.motion[:80])

        log_timing(logger, "comparison_evaluation_total_wall", time.perf_counter() - t_cmp, motion=self.motion[:80])
        logger.info("[Comparison Evaluation] Done")
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="eval.yml")
    parser.add_argument("--eval_file", type=str, default="")
    parser.add_argument("--pool_file", type=str, default="")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--pre_only", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--comparison", action="store_true", default=False)
    parser.add_argument("--reverse", action="store_true", default=False)
    args = parser.parse_args()

    log_file = logger.handlers[0].baseFilename
    save_file = log_file.replace(".log", ".json")
    logger.info(f"Saving to {save_file}")
    with open(f"configs/{args.config}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info(f"Config: {config}")

    if args.pool_file != "":
        for debater in config["debater"]:
            debater["pool_file"] = f"{args.pool_file}_pool_{debater['side']}.json"
            print(f"Load pool file for {debater['side']} from {debater['pool_file']}")

    # Set use_rehearsal_tree and use_debate_flow_tree based on command line arguments
    # use_rehearsal_tree = not args.no_rehearsal_tree
    # use_debate_flow_tree = not args.no_debate_flow_tree

    # if "env" not in config:
    #     config["env"] = {}
    # config["env"]["use_rehearsal_tree"] = use_rehearsal_tree
    # config["env"]["use_debate_flow_tree"] = use_debate_flow_tree

    # use_rehearsal_tree = config["env"]["use_rehearsal_tree"]
    # use_debate_flow_tree = config["env"]["use_debate_flow_tree"]
    # logger.info(f"Use rehearsal tree: {use_rehearsal_tree}")
    # logger.info(f"Use debate flow tree: {use_debate_flow_tree}")

    env_config = EnvConfig(
        debater_config=[DebaterConfig(**config) for config in config["debater"]],
        judge_config=JudgeConfig(**config["judge"]),
        audience_config=AudienceConfig(**config["audience"]),
        **config["env"],
    )
    env = Env(env_config, args.debug)

    if args.pre_only:
        sorted_claim = env.play(pre_only=True)
        save_file = save_file.replace(".json", "_pre.json")
        with open(save_file, "w") as fout:
            fout.write(json.dumps(sorted_claim) + "\n")
        logger.info(f"Save Claim Selection in Preparation in {save_file}")
    elif args.eval_only:
        # load external debate log file for evaluation
        save_file = args.eval_file
        data = json.load(open(save_file))
        if args.comparison:
            evaluation = env.compare_debate(data["comparison"], order_reverse=args.reverse)
            logger.info(f"Result: {evaluation}")
            data["comparison_evaluation"] = evaluation
        else:
            process = data["debate_process"][1:]
            evaluation, side_into = env.eval(process=process)
            logger.info(f"Result: {evaluation}")
            data["evaluation"] = evaluation
            data["eval_side_info"] = side_into
        json.dump(data, open(save_file, "w"), indent=2)
    else:
        env.play()
        record = {
            "motion": env.motion,
            "config": config,
            "debate_process": env.debate_process[1:],
            "debate_thoughts": {
                "for": env.debaters["for"].debate_thoughts,
                "against": env.debaters["against"].debate_thoughts,
            },
            "debate_tree": {
                "for": [
                    (
                        env.debaters["for"].debate_tree.get_tree_info()
                        if env.debaters["for"].type in ["treedebater"]
                        else {}
                    ),
                    (
                        env.debaters["for"].oppo_debate_tree.get_tree_info()
                        if env.debaters["for"].type in ["treedebater"]
                        else {}
                    ),
                ],
                "against": [
                    (
                        env.debaters["against"].debate_tree.get_tree_info()
                        if env.debaters["against"].type in ["treedebater"]
                        else {}
                    ),
                    (
                        env.debaters["against"].oppo_debate_tree.get_tree_info()
                        if env.debaters["against"].type in ["treedebater"]
                        else {}
                    ),
                ],
            },
            "conversation": {"for": env.debaters["for"].conversation, "against": env.debaters["against"].conversation},
        }
        json.dump(record, open(save_file, "w"), indent=2)

        if not args.debug:
            evaluation, side_into = env.eval()
            logger.info(f"Result: {evaluation}")
            record.update({"evaluation": evaluation, "eval_side_info": side_into})
            json.dump(record, open(save_file, "w"), indent=2)
