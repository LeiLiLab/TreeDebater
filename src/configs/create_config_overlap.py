import argparse
import copy
import os

import yaml

from create_config import dict_order_preserving_yaml_dump


def get_options():
    parser = argparse.ArgumentParser(
        description="Generate overlap configs from a template, updating motion, model, and pool_file."
    )
    parser.add_argument("--template", type=str, default="overlap_debate_2.yml")
    parser.add_argument(
        "--motion",
        type=str,
        nargs="+",
        default=["Learning to be a good writer still matters in the age of ai"],
    )
    parser.add_argument("--motion_file", type=str, default=None)
    parser.add_argument("--model", type=str, nargs="+", default=None)
    parser.add_argument("--pool_version", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="overlap")
    return parser.parse_args()


def motion_to_filename(motion: str) -> str:
    return motion.replace(" ", "_").lower()


def pool_file_path(pool_version: str, model_name: str, motion: str, side: str) -> str:
    motion_name = motion_to_filename(motion)
    suffix = "pool_for" if side == "for" else "pool_against"
    return f"../results{pool_version}/{model_name}/{motion_name}_{suffix}.json"


def pool_check_path(pool_version: str, model_name: str, motion: str, side: str) -> str:
    """Filesystem path relative to src/configs/ (same as create_config.py)."""
    motion_name = motion_to_filename(motion)
    suffix = "pool_for" if side == "for" else "pool_against"
    return f"../../results{pool_version}/{model_name}/{motion_name}_{suffix}.json"


def pool_files_exist(motion: str, pool_version: str, model_name: str) -> bool:
    pool_for = pool_check_path(pool_version, model_name, motion, "for")
    pool_against = pool_check_path(pool_version, model_name, motion, "against")
    if not os.path.exists(pool_for) or not os.path.exists(pool_against):
        print(f"Pool file {pool_for} or {pool_against} does not exist")
        return False
    return True


def model_from_template(configs: dict) -> str:
    for debater in configs.get("debater", []):
        if "model" in debater:
            return debater["model"]
    raise ValueError("No debater with model found in template")


def short_model_name(model: str) -> str:
    return model.split("/")[-1]


def apply_overrides(configs: dict, motion: str, model: str, pool_version: str) -> dict:
    configs = copy.deepcopy(configs)
    configs["env"]["motion"] = motion
    model_name = short_model_name(model)
    for debater in configs["debater"]:
        if "model" in debater:
            debater["model"] = model
        if "helper_model" in debater:
            debater["helper_model"] = model
        side = debater.get("side")
        if side in ("for", "against") and "pool_file" in debater:
            debater["pool_file"] = pool_file_path(pool_version, model_name, motion, side)
    return configs


# python create_config_overlap.py --model deepseek-chat --motion "Learning to be a good writer still matters in the age of ai"
# python create_config_overlap.py --model deepseek/deepseek-chat --motion_file ../../dataset/motion_list.txt --save_dir emnlp --pool_version 0808

if __name__ == "__main__":
    args = get_options()
    with open(args.template, "r", encoding="utf-8") as f:
        template = yaml.load(f, Loader=yaml.FullLoader)

    if args.motion_file:
        with open(args.motion_file, "r", encoding="utf-8") as f:
            motions = [line.strip() for line in f if line.strip()]
    else:
        motions = args.motion
    print(motions)

    template_name = args.template.rsplit(".", 1)[0]
    model = args.model[0] if args.model else model_from_template(template)
    model_name = short_model_name(model)

    for i, motion in enumerate(motions):
        save_path = f"{args.save_dir}/case{i + 1}"
        os.makedirs(save_path, exist_ok=True)

        if not pool_files_exist(motion, args.pool_version, model_name):
            continue

        configs = apply_overrides(template, motion, model, args.pool_version)

        save_file = f"{save_path}/{template_name}_{model_name}.yml"
        dict_order_preserving_yaml_dump(configs, save_file)
