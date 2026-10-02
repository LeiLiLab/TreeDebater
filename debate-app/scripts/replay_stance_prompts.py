"""Replay saved stance failures without audio. Makes four paid model calls.

Run from the repository root: python debate-app/scripts/replay_stance_prompts.py
Results retain complete prompts and responses for manual semantic review.
"""
import ast
import argparse
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'debate-app/reports'


def literal(name):
    tree = ast.parse((ROOT / 'src/utils/prompts/others.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return ast.literal_eval(node.value)


def saved_blocks():
    text = (REPORTS / 'recorded-full-speech-2026-09-13-human-for-io.log').read_text()
    return re.split(r'(?=2026-\d\d-\d\d \d\d:\d\d:\d\d DEBUG \[io\])', text)


def body(block):
    return block.split('\n', 2)[2].rsplit('\n' + '=' * 60, 1)[0].strip()


def cases():
    blocks = saved_blocks()
    planner = literal('debate_flow_tree_action_eval_prompt')
    motion = 'Learning to be a good writer still matters in the age of AI'
    for label, call in [('opening_plan', 30), ('rebuttal_plan', 67)]:
        # Find by call ID and title, keeping the original tree and action data.
        candidates = [b for b in blocks if f'call_id={call} ' in b.split('\n')[0]
                      and 'title=Debate-Flow-Tree-Action-Eval-Prompt' in b.split('\n')[0]]
        if len(candidates) != 1:
            raise ValueError(f'Expected one saved {label} prompt, found {len(candidates)}')
        old = body(candidates[0])
        own = old.split('**Your Tree**:', 1)[1].split("**Opponent's Tree**:", 1)[0].strip()
        other = old.split("**Opponent's Tree**:", 1)[1].split('### Debate points:', 1)[0].strip()
        other = other.replace('Your Main Claim', "Opponent's Main Claim").replace("Opponent's Attack", 'Your Attack').replace('Your Rebuttal', "Opponent's Rebuttal")
        actions = json.JSONDecoder().raw_decode(old.split('### Debate points:', 1)[1].lstrip())[0]
        for action in actions:
            support = action['action'] in ('propose', 'reinforce', 'defend')
            action.update(claim_owner='us' if support else 'opponent',
                          desired_direction='support' if support else 'challenge',
                          unclassified_arguments=[action.get('target_argument', '')], counterarguments=[])
        yield label, [{'role': 'user', 'content': planner.format(
            motion=motion, side='against', act='OPPOSE', tree=own, oppo_tree=other, actions=json.dumps(actions))}]
    old = body(next(b for b in blocks if 'title=Get-Expert-Audience-Revision-Prompt_iter1' in b.split('\n')[0]))
    constraints = literal('post_process_prompt').split('## Non-negotiable meaning constraints', 1)[1].split('### Workflow', 1)[0]
    yield 'opening_revision', [{'role': 'user', 'content':
        '## Non-negotiable meaning constraints' + constraints.format(side='against', motion=motion) + old}]
    # Capture the actual timing-rewrite messages without importing the audio stack.
    module = ast.parse((ROOT / 'src/tts_streaming.py').read_text())
    function = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == '_revise_to_n_words')
    captured = {}
    def capture(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='captured'))])
    scope = {'List': list, 'OutputConfig': SimpleNamespace(refinement_model='gpt-4o-mini')}
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'timing_prompt', 'exec'), scope)
    draft = body(next(b for b in blocks if 'title=Response-Before-Post-Process' in b.split('\n')[0]))
    paragraph = draft[draft.rfind('In conclusion,'):]
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=capture)))
    scope['_revise_to_n_words'](client, paragraph, 65, [], motion=motion, side='against')
    yield 'timing_conclusion', captured['messages']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=REPORTS / 'stance-prompt-replay-2026-09-13.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite replay evidence: {args.output}')
    for key, value in json.loads((ROOT / 'src/configs/api_key.json').read_text()).items():
        os.environ.setdefault(key, value)
    client = OpenAI(timeout=90, max_retries=1)
    output = args.output
    results = []
    for name, messages in cases():
        response = client.chat.completions.create(model='gpt-4o-mini', messages=messages, temperature=0)
        results.append({'case': name, 'model': response.model, 'messages': messages,
                        'response': response.choices[0].message.content,
                        'usage': response.usage.model_dump()})
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
        print(f'{name}: saved', flush=True)


if __name__ == '__main__':
    main()
