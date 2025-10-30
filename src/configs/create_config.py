import argparse
import os
from collections import OrderedDict

import yaml


def dict_order_preserving_yaml_dump(data, file_path=None, **kwargs):
    class OrderPreservingDumper(yaml.SafeDumper):
        def represent_mapping(self, tag, mapping, flow_style=None):
            return super().represent_mapping(tag, mapping, flow_style)

    def dict_representer(dumper, data):
        return dumper.represent_mapping(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items())

    OrderPreservingDumper.add_representer(dict, dict_representer)

    if file_path:
        with open(file_path, "w", encoding="utf-8") as file:
            yaml.dump(data, file, Dumper=OrderPreservingDumper, width=float("inf"), **kwargs)
        print(f"YAML content has been written to {file_path}")
    else:
        return yaml.dump(data, Dumper=OrderPreservingDumper, width=float("inf"), **kwargs)


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=str, default="compare.yml")
    parser.add_argument("--motion", type=str, nargs="+", default=["AI will lead to the decline of human creative arts"])
    parser.add_argument("--motion_file", type=str, default=None)
    parser.add_argument("--model", type=str, nargs="+", default=["gemini-2.0-flash"])
    parser.add_argument("--baseline", type=str, default="baseline", choices=["treedebater", "baseline"])
    parser.add_argument("--pool_version", type=str, default="1228")
    parser.add_argument("--pool_size", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="1228")
    args = parser.parse_args()
    return args


# python create_config.py --model deepseek/deepseek-chat
# python create_config.py --model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo gemini-1.5-pro
# python create_config.py --model gemini-2.0-flash --motion "We should abolish the debt ceiling" --save_dir 0410 --pool_version 0410
# python create_config.py --model deepseek-chat --motion "AI will lead to the decline of human creative arts"  --save_dir testw --pool_version 10 --template compare.yml
# python create_config.py --model deepseek-chat --motion "AI will lead to the decline of human creative arts"  --save_dir test3 --pool_version 10 --template compare.yml
# python create_config.py --model moonshot-v1-32k --motion "AI will lead to the decline of human creative arts" --save_dir test4 --pool_version 11 --template compare.yml
# python create_config.py --model moonshot-v1-32k --motion "It is time to welcome an A.I. Tutor in the classroom" --save_dir test5 --pool_version 11 --template compare.yml
# create config for head2head debate
# python create_config.py --motion_file ../../dataset/motion_list.txt  --save_dir test --pool_version 0515 --template compare.yml

if __name__ == "__main__":
    args = get_options()
    with open(args.template, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    if args.motion_file:
        with open(args.motion_file, "r") as f:
            motions = [x.strip() for x in f.readlines()]
    else:
        motions = args.motion
    print(motions)

    configs["env"]["claim_pool_size"] = args.pool_size

    assert len(args.model) <= 2
    model1, model2 = args.model[0], args.model[-1]
    template_name = args.template.split(".yml")[0]

    for i, motion in enumerate(motions):
        save_path = f"{args.save_dir}/case{i+1}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model_name1 = model1.split("/")[-1]
        model_name2 = model2.split("/")[-1]
        motion_name = motion.replace(" ", "_").lower()

        configs["env"]["motion"] = motion
        configs["debater"][0] = {
            "side": "for",
            "model": model1,
            "type": "treedebater",
            "temperature": 1.0,
            "use_retrieval": True,
            "pool_file": f"../results{args.pool_version}/{model_name1}/{motion_name}_pool_for.json",
        }
        configs["debater"][1] = {
            "side": "against",
            "model": model2,
            "type": "baseline",
            "temperature": 1.0,
        }
        if model_name1 == model_name2:
            save_file = f"{save_path}/{template_name}_{model_name1}.yml"
        else:
            save_file = f"{save_path}/{template_name}_{model_name1}_{model_name2}.yml"
        dict_order_preserving_yaml_dump(configs, save_file)

        configs["debater"][0] = {
            "side": "for",
            "model": model2,
            "type": "baseline",
        }
        configs["debater"][1] = {
            "side": "against",
            "model": model1,
            "type": "treedebater",
            "temperature": 1.0,
            "use_retrieval": True,
            "add_retrieval_feedback": True,
            "pool_file": f"../results{args.pool_version}/{model_name2}/{motion_name}_pool_against.json",
        }

        if model_name1 == model_name2:
            save_file = f"{save_path}/{template_name}_{model_name1}_re.yml"
        else:
            save_file = f"{save_path}/{template_name}_{model_name1}_{model_name2}_re.yml"
        dict_order_preserving_yaml_dump(configs, save_file)
