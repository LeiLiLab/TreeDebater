"""Offline paired matching on trees actually shown in historical extraction prompts.

No models, embeddings or network calls. The legacy metric is exact lookup before
its embedding fallback; absence there is NOT claimed to be a final legacy miss.
"""
import hashlib
import json
from pathlib import Path
import re
import sys
from collections import Counter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'experiments/retrieval_all_motions_v1'))
from data import blocks
from debate_tree import DebateTree
from streaming.target_matching import resolve_target
from streaming.tree_updates import apply_statements
from streaming.grounding import normalize


def parse_tree(body, motion, side, seed):
    tree = DebateTree(motion, side)
    stack = [tree.root]
    for line in body.splitlines():
        match = re.match(r'^\s*Level-(\d+) .*?(\{"claim".*)', line)
        if not match:
            continue
        depth = int(match[1])
        value, _ = json.JSONDecoder().raw_decode(match[2])
        if depth < 1 or depth > len(stack):
            raise ValueError('Incomplete printed ancestry')
        stack = stack[:depth]
        node = stack[-1].add_node(new_claim=value['claim'], new_argument=value.get('argument', []),
                                 side=side if depth % 2 else ('against' if side == 'for' else 'for'))
        node.node_id = hashlib.sha256(f'{seed}:{len(tree.get_all_nodes())}'.encode()).hexdigest()[:32]
        stack.append(node)
    return tree


def parse_prompt(body, source):
    motion = body.split('**Debate Topic**:', 1)[1].split('\n', 1)[0].strip()
    side = body.split('**Your Stance**:', 1)[1].split('\n', 1)[0].strip()
    transcript = body.split('**Statement**:', 1)[1].split('**Your Debate Tree**:', 1)[0].strip()
    owntext, othertext = body.split('**Your Debate Tree**:', 1)[1].split("**Opponent's Debate Tree**:", 1)
    othertext = othertext.split('##Response Format', 1)[0]
    opposite = 'against' if side == 'for' else 'for'
    return motion, side, transcript, (parse_tree(owntext, motion, side, source + ':own'),
                                    parse_tree(othertext, motion, opposite, source + ':other'))


def main():
    counts = Counter(); reasons = Counter(); per_motion = {}; records = []; files = {}; seen = set()
    for file in sorted((ROOT / 'emnlp_res').glob('*_io.log')):
        text = file.read_text()
        files[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
        pending = None
        for header, body, offset in blocks(text):
            if 'title=Analyze-Helper-Prompt ' in header:
                pending = (header, body, offset)
                counts['prompt_blocks'] += 1
            elif 'title=Analyze-Helper-Response ' in header:
                if pending is None:
                    counts['unpaired_response_blocks'] += 1
                    continue
                ph, prompt, poffset = pending
                pending = None
                assert ph.split(' stage=', 1)[1] == header.split(' stage=', 1)[1]
                key = hashlib.sha256((prompt + body).encode()).hexdigest()
                if key in seen:
                    counts['duplicate_prompt_response_pairs'] += 1
                    continue
                seen.add(key)
                source = f'{file.relative_to(ROOT)}#offset={poffset}'
                payload, _ = json.JSONDecoder().raw_decode(body.strip().removeprefix('```json').lstrip())
                motion, side, speech, trees = parse_prompt(prompt, source)
                counts['paired_batches'] += 1
                stats = per_motion.setdefault(motion, Counter())
                stats['batches'] += 1
                initial = [(t, n) for t in trees for n in t.get_all_nodes() if n.parent is not None]
                for ci, item in enumerate(payload['statements']):
                    counts['extracted_claims'] += 1
                    quote = item.get('content')
                    grounded = isinstance(quote, str) and bool(quote.strip()) and normalize(quote) in normalize(speech)
                    counts['grounded_claims' if grounded else 'ungrounded_claims'] += 1
                    purposes = item.get('purpose') or []
                    if isinstance(purposes, dict): purposes = [purposes]
                    for pi, purpose in enumerate(purposes):
                        if purpose['action'] not in ('attack', 'rebut', 'reinforce'):
                            continue
                        counts['relations'] += 1; stats['relations'] += 1
                        preferred = trees[0] if purpose['targeted_debate_tree'] == 'you' else trees[1]
                        root_side = preferred.root.side
                        legacy_side = ('against' if root_side == 'for' else 'for') if purpose['action'] == 'rebut' else root_side
                        exact = preferred.get_node_by_claim(purpose['target'], side=legacy_side)
                        expected = side if purpose['action'] == 'reinforce' else ('against' if side == 'for' else 'for')
                        matched, info = resolve_target(initial, purpose, expected_side=expected, preferred_tree=preferred)
                        okay = info['reason'] is None
                        counts['legacy_primary_exact' if exact else 'legacy_requires_fallback'] += 1
                        stats['legacy_primary_exact' if exact else 'legacy_requires_fallback'] += 1
                        counts['new_resolved' if okay else 'new_unresolved'] += 1
                        stats['new_resolved' if okay else 'new_unresolved'] += 1
                        if exact and exact.side != expected: counts['legacy_exact_wrong_owner'] += 1
                        if okay and not exact: counts['newly_resolved_without_semantic_lookup'] += 1
                        if not okay: reasons[info['reason']] += 1
                        if grounded:
                            counts['grounded_relations'] += 1
                            counts['grounded_new_resolved' if okay else 'grounded_new_unresolved'] += 1
                        records.append(dict(source=source, response_offset=offset, motion=motion, claim_index=ci,
                                            purpose_index=pi, purpose=purpose, grounded=grounded,
                                            legacy_primary_exact=exact.node_id if exact else None,
                                            legacy_primary_correct_owner=exact.side == expected if exact else None,
                                            resolution=info))
                # Verify safe application separately, using the exact logged source.
                events = apply_statements(trees, payload['statements'], speech, side, allow_corrections=False)
                counts['preserved_unlinked_claim_events'] += sum(e['action'] == 'UNLINKED_CLAIM' for e in events)
                counts['applied_response_links'] += sum(e['action'] == 'LINK_RESPONSE' for e in events)
                counts['applied_reinforcements'] += sum(e['action'] == 'REINFORCE' for e in events)
    report = dict(method='Logged pre-extraction tree snapshots paired with original extracted purposes. No regenerated queries, no final-tree lookahead.',
                  limitations=['Legacy primary exact lookup only; original paid embedding fallback is not replayed.',
                               'Historical extractions contain no target IDs; future ID selection accuracy is not measured here.',
                               'Resolvable identity is not proof that the model chose the semantically correct target.',
                               'Only logs with paired analysis prompts/responses provide update-validation evidence.'],
                  new_api_cost_usd=0, counts=dict(counts), unresolved_reasons=dict(reasons),
                  per_motion={m: dict(c) for m,c in per_motion.items()}, input_sha256=files)
    (HERE / 'historical_records.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    (HERE / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
