import argparse
import copy
import json
import os

import yaml

from agents import AudienceConfig, BaselineDebater, DebaterConfig, JudgeConfig
from env import Env, EnvConfig, extract_overall_score
from ouragents import TreeDebater
from utils.constants import CLOSING_TIME, OPENING_TIME, REBUTTAL_TIME
from utils.tool import logger


def get_debater_class(type):
    if type == "baseline":
        return BaselineDebater
    elif type == "treedebater":
        return TreeDebater
    else:
        raise ValueError(f"Type {type} is not supported.")


class CompareEnv:
    def __init__(self, config, debug, baseline_type="baseline", test_type="treedebater", port=8081) -> None:
        self.config = config
        self.motion = config.motion
        self.claim_pool_size = config.claim_pool_size
        self.reverse = config.reverse  # if to reverse the order of the debaters
        self.time_control = config.time_control
        self.debug = debug
        self.baseline_type = baseline_type
        self.test_type = test_type
        self.stance_dict = {"for": None, "against": None}

        assert self.baseline_type == "baseline", "Baseline type must be baseline"

        # init players
        self.test_debaters = {}
        self.baseline_debaters = {}

        config_for_type = {conf.type: conf for conf in config.debater_config}
        self.debater_type_for_side = {
            conf.side: "baseline" if conf.type == self.baseline_type else "test" for conf in config.debater_config
        }

        if self.baseline_type not in config_for_type:
            logger.warning(
                f"Baseline type {self.baseline_type} not found in config_for_type, using {self.test_type} config"
            )
            baseline_conf = copy.deepcopy(config_for_type[self.test_type])
            baseline_conf.type = self.baseline_type
            config_for_type[self.baseline_type] = baseline_conf

        logger.debug(f"Config for type: {config_for_type}")

        for side in ["for", "against"]:
            cur_config = config_for_type[self.test_type]
            cur_config.side = side
            cur_config.pool_file = cur_config.pool_file.split("_pool_")[0] + f"_pool_{side}.json"
            self.test_debaters[side] = get_debater_class(self.test_type)(cur_config, motion=self.motion)
            cur_config = config_for_type[self.baseline_type]
            cur_config.side = side
            self.baseline_debaters[side] = get_debater_class(self.baseline_type)(
                cur_config, motion=self.motion, port=port
            )

    def step_play(self, side, stage, history, max_time):

        baseline_call = {
            "opening": self.baseline_debaters[side].opening_generation,
            "rebuttal": self.baseline_debaters[side].rebuttal_generation,
            "closing": self.baseline_debaters[side].closing_generation,
        }

        test_call = {
            "opening": self.test_debaters[side].opening_generation,
            "rebuttal": self.test_debaters[side].rebuttal_generation,
            "closing": self.test_debaters[side].closing_generation,
        }

        base_response = baseline_call[stage](history=history, max_time=max_time, time_control=self.time_control)

        # Generate test response using reference history
        test_response = test_call[stage](history=history, max_time=max_time, time_control=self.time_control)
        return base_response, test_response

    def compare_play(self):
        order = ["for", "against"] if not self.reverse else ["against", "for"]
        debate_process = []
        debate_settings = {
            "stage": "settings",
            "motion": self.motion,
            "debaters": {x.side: x for x in self.config.debater_config},
        }
        debate_process.append(debate_settings)

        comparison_results = {}

        # Run through each stage
        for stage in ["preparation", "opening", "rebuttal", "closing"]:
            logger.info(f"[{stage}] Start Comparison")

            if stage == "preparation":
                # Generate claims for both reference and test debaters
                # let the TreeDebater-type debaters generate a set of candidate claims as the knowledge or material for the subsequent发言
                for side in order:
                    if self.test_debaters[side].type in ["treedebater"]:
                        self.test_debaters[side].claim_generation(self.claim_pool_size, temperature=1)
                        # self.test_debaters[side].claim_selection()
            else:
                for side in order:
                    keep_response_for_side = self.debater_type_for_side[side]

                    # Generate reference response
                    if stage == "opening":
                        base_response, test_response = self.step_play(
                            side, stage, history=debate_process[1:], max_time=OPENING_TIME
                        )
                    elif stage == "rebuttal":
                        base_response, test_response = self.step_play(
                            side, stage, history=debate_process[1:], max_time=REBUTTAL_TIME
                        )
                    elif stage == "closing":
                        base_response, test_response = self.step_play(
                            side, stage, history=debate_process[1:], max_time=CLOSING_TIME
                        )

                    # according to the type of the debater, decide which response to use as the material for the subsequent发言
                    # if the debater type is baseline, use base_response as the material for the subsequent发言
                    # if the debater type is test, use test_response as the material for the subsequent发言
                    if keep_response_for_side == "baseline":  # when test is TreeDebater
                        self.test_debaters[side].reset_stage(stage, side, base_response, history=debate_process[1:])
                        keep_response = base_response
                    else:  # keep_response_for_side == "test"
                        self.test_debaters[side].reset_stage(stage, side, test_response, history=debate_process[1:])
                        self.baseline_debaters[side].reset_stage(stage, side, test_response)
                        keep_response = test_response

                    debate_process.append({"stage": stage, "side": side, "content": keep_response})

                    comparison_results[f"{stage}_{side}"] = {
                        "baseline_response": base_response,
                        "test_response": test_response,
                        "keep_response": keep_response,
                    }

            logger.info(f"[{stage}] Comparison Complete")

            if self.debug:
                response = input("Press N to stop: ")
                if response.lower() == "n":
                    break

        return comparison_results, debate_process


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="compare.yml")
    parser.add_argument("--baseline_type", type=str, default="baseline")
    parser.add_argument("--test_type", type=str, default="treedebater")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()

    log_file = logger.handlers[0].baseFilename
    save_file = log_file.replace(".log", "_test_comparison.json")
    logger.info(f"Saving to {save_file}")

    with open(f"configs/{args.config}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        logger.info(f"Config: {config}")

    env_config = EnvConfig(
        debater_config=[DebaterConfig(**config) for config in config["debater"]],
        judge_config=JudgeConfig(**config["judge"]),
        audience_config=AudienceConfig(**config["audience"]),
        **config["env"],
    )

    compare_env = CompareEnv(env_config, args.debug, args.baseline_type, args.test_type, port=args.port)
    comparison_results, debate_process = compare_env.compare_play()

    test_record = {
        "motion": compare_env.motion,
        "config": config,
        "debate_thoughts": {
            "for": compare_env.test_debaters["for"].debate_thoughts,
            "against": compare_env.test_debaters["against"].debate_thoughts,
        },
        "debate_tree": {
            "for": [
                (
                    compare_env.test_debaters["for"].debate_tree.get_tree_info()
                    if compare_env.test_debaters["for"].type in ["treedebater"]
                    else {}
                ),
                (
                    compare_env.test_debaters["for"].oppo_debate_tree.get_tree_info()
                    if compare_env.test_debaters["for"].type in ["treedebater"]
                    else {}
                ),
            ],
            "against": [
                (
                    compare_env.test_debaters["against"].debate_tree.get_tree_info()
                    if compare_env.test_debaters["against"].type in ["treedebater"]
                    else {}
                ),
                (
                    compare_env.test_debaters["against"].oppo_debate_tree.get_tree_info()
                    if compare_env.test_debaters["against"].type in ["treedebater"]
                    else {}
                ),
            ],
        },
        "conversation": {
            "for": compare_env.test_debaters["for"].conversation,
            "against": compare_env.test_debaters["against"].conversation,
        },
    }

    # Save results
    with open(save_file, "w") as f:
        full_results = {
            "config": config,
            "baseline_type": args.baseline_type,
            "test_type": args.test_type,
            "comparison": comparison_results,
            "debate_process": debate_process[1:],
            "test_record": test_record,
        }
        json.dump(full_results, f, indent=2)

    logger.info(f"Comparison results saved to {save_file}")

    total_usage = 0
    for side in ["for", "against"]:
        logger.info(f"Usage for Agent {side} " + str(compare_env.test_debaters[side].client_cost))
        total_usage += compare_env.test_debaters[side].client_cost
