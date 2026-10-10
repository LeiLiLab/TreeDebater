"""Immutable source projection and speaker-owned speech history."""
import re


class SpeechRejected(RuntimeError):
    """A draft or review cannot safely proceed to speech delivery."""


def speech_history(history, side, *, opponent_statement=None, opponent_stage=None):
    """Keep our delivered speeches as assistant turns and heard speech as user turns.

    Domain history carries speaker sides; filtered Debater history carries chat
    roles and Opponent's Statement headings. Private prompts are never speech.
    Replace only the current opponent turn when a newer partial/corrected ASR
    snapshot is supplied. Identical speech in another stage remains history.
    """
    opponent = 'against' if side == 'for' else 'for'
    entries = []
    for entry in history:
        content = entry['content']
        header = re.match(r"\*\*Opponent's (\w+) Statement\*\*\n", content, re.I)
        owner = entry.get('side')
        if owner is None:
            if entry.get('role') == 'assistant':
                owner = side
            elif entry.get('role') == 'user' and header:
                owner = opponent
            else:
                continue
        if owner not in (side, opponent):
            raise ValueError('Speech history has an unknown speaker side')
        stage = entry.get('stage') or (header.group(1).lower() if header else 'previous')
        if header and owner == opponent:
            content = content[header.end():]
        if content.strip():
            entries.append(dict(side=owner, stage=stage, content=content))
    if opponent_statement:
        last = entries[-1] if entries else None
        if (last and last['side'] == opponent
                and (last['stage'] == opponent_stage
                     or (opponent_stage is None or last['stage'] == 'previous')
                     and last['content'] == opponent_statement)):
            last['content'] = opponent_statement
        else:
            entries.append(dict(side=opponent, stage=opponent_stage or 'current',
                                content=opponent_statement))
    return [dict(role='assistant' if e['side'] == side else 'user',
                 content=e['content'] if e['side'] == side else
                     f"**Opponent's {e['stage']} Statement**\n" + e['content'])
            for e in entries]


def authoring_history(data):
    turn = str(data.get('turn') or '').partition(':')
    opponent = 'against' if data['our_side'] == 'for' else 'for'
    stage = turn[2] if turn[0] == opponent else None
    transcript = data.get('final_transcript') if data.get('endpoint') else data.get('heard_transcript')
    return speech_history(data.get('debate_history', []), data['our_side'],
                          opponent_statement=transcript, opponent_stage=stage)


def authoring_sources(data):
    """Project spoken evidence and owned preparation, never listener-authored tactics."""
    side = data['our_side']
    opponent = 'against' if side == 'for' else 'for'
    if any(t.get('side', opponent) != opponent for t in data.get('current_targets', [])):
        raise ValueError('Opponent authoring sources contain a target owned by our side')
    selected = set(data.get('selected_target_ids', []))
    return dict(motion=data['motion'], our_side=side, stage=data['stage'],
        our_definition=data.get('our_definition', ''),
        our_main_claims=data.get('our_main_claims', []),
        **{key: data[key] for key in ('our_tree', 'opponent_tree') if data.get(key)},
        clash_records=data.get('clash_records', []),
        opponent_statement=dict(side=opponent, complete=bool(data.get('endpoint', False))),
        opponent_targets=[dict(node_id=t['node_id'], side=opponent, claim=t['claim'],
            sources=t.get('sources', []), constraints=t.get('constraints', []))
            for t in data.get('current_targets', []) if not selected or t['node_id'] in selected],
        supplied_evidence=data.get('supplied_evidence', []),
        prepared_materials=data.get('prepared_rehearsal_materials', []),
        **({'prepared_revision_evidence': data['prepared_revision_evidence']}
           if 'prepared_revision_evidence' in data else {}))
