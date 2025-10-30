import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

from tqdm import tqdm

from debate_tree import DebateTree
from utils.helper import extract_statement
from utils.model import HelperClient
from utils.tool import logger


def analyze_statement(llm, motion, status, statements, side, pro_debate_tree, con_debate_tree):
    """
    Analyze the statements:
    1. Extract the claims from the statements
    2. Match the opponent's claims with the debater's claims
    3. Update the claim status
    """
    if side == "for":  # statement is from the for side
        tree, oppo_tree = pro_debate_tree, con_debate_tree
    else:
        tree, oppo_tree = con_debate_tree, pro_debate_tree
    claims = extract_statement(
        llm,
        motion,
        statements,
        tree=[tree.print_tree(include_status=True), oppo_tree.print_tree(include_status=True)],
        side=side,
        stage=status,
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
        if status == "opening":
            if "propose" not in [p["action"] for p in purpose]:
                purpose.append({"action": "propose", "targeted_debate_tree": "you", "target": claim})
        for p in purpose:
            action = p["action"]
            target = p["target"]
            target_tree = p["targeted_debate_tree"]
            if action == "propose" or action == "reinforce" or action == "rebut":
                if target_tree != "you":
                    logger.warning(
                        f"Propose or reinforce action is not allowed for the opponent's debate tree: {target}"
                    )
                    continue
            elif action == "attack":
                if target_tree == "you":
                    logger.warning(f"Attack action is not allowed for your own debate tree: {target}")
                    continue
            else:
                logger.warning(f"Unknown action: {action}")
                continue

            target_tree = tree if target_tree == "you" else oppo_tree
            target_tree.update_node(action, new_claim=claim, new_argument=arguments, target=target)

    return claims


def process_debate_file(input_path: str, output_path: str) -> None:
    """
    Process a single debate JSON file and extract structured arguments.

    Args:
        input_path: Path to input JSON file
        output_path: Path to save processed JSON file
        api_key: Google API key
    """
    with open(input_path, "r") as f:
        debate_data = json.load(f)

    if len(debate_data["debate_process"]) < 6:
        logger.info(f"Skipping debate {input_path} because it has less than 3 stages")
        return

    motion = debate_data["motion"] if "motion" in debate_data else debate_data["config"]["env"]["motion"]
    processed_debate = {"motion": motion, "structured_arguments": []}

    llm = partial(HelperClient, model="gemini-2.0-flash", temperature=0, max_tokens=8192, n=1)

    pro_debate_tree = DebateTree(motion, "for")
    con_debate_tree = DebateTree(motion, "against")

    # Create progress bar for arguments within the file
    with tqdm(
        total=len(debate_data["debate_process"]), desc=f"Processing arguments in {Path(input_path).name}", leave=False
    ) as pbar:
        for stage in debate_data["debate_process"]:
            claims = analyze_statement(
                llm, motion, stage["stage"], stage["content"], stage["side"], pro_debate_tree, con_debate_tree
            )

            processed_debate["structured_arguments"].append(
                {"stage": stage["stage"], "side": stage["side"], "claims": claims}
            )
            pbar.update(1)

    processed_debate["pro_debate_tree"] = pro_debate_tree.get_tree_info()
    processed_debate["con_debate_tree"] = con_debate_tree.get_tree_info()

    logger.info(f"Pro_debate_tree: {pro_debate_tree.print_tree(include_status=True)}")
    logger.info(f"Con_debate_tree: {con_debate_tree.print_tree(include_status=True)}")

    with open(output_path, "w") as f:
        json.dump(processed_debate, f, indent=2)


def process_input(input_path: str, output_path: str) -> None:
    """
    Process either a single file or directory of JSON files.

    Args:
        input_path: Path to input file or directory
        output_path: Path to output file or directory
        api_key: Google API key
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.is_file():
        # Process single file
        if not output_path.parent.exists():
            os.makedirs(output_path.parent, exist_ok=True)
        print(f"\nProcessing file: {input_path.name}")
        process_debate_file(str(input_path), str(output_path))
        print(f"✓ Saved processed file to {output_path}")

    elif input_path.is_dir():
        # Process directory
        if not output_path.exists():
            os.makedirs(output_path, exist_ok=True)

        # Get list of JSON files
        json_files = [
            f
            for f in input_path.glob("*.json")
            if (not (output_path / f"processed_{f.name}").exists()) and (not f.name.startswith("processed_"))
        ]
        print(json_files)

        # Create progress bar for files
        with tqdm(total=len(json_files), desc="Processing files", unit="file") as pbar:
            for json_file in json_files:
                out_file = output_path / f"processed_{json_file.name}"
                process_debate_file(str(json_file), str(out_file))
                pbar.update(1)


def main():
    parser = argparse.ArgumentParser(description="Process debate JSON files to extract structured arguments")
    parser.add_argument("input", help="Input JSON file or directory containing JSON files")
    parser.add_argument("output", help="Output file path (if input is file) or directory (if input is directory)")

    args = parser.parse_args()

    try:
        process_input(args.input, args.output)
        print("\n✓ Processing completed successfully!")
    except (FileNotFoundError, KeyError) as e:
        print(f"Error in debate_tree_extract.py: {e}")
        return


if __name__ == "__main__":
    main()
