"""Readable planning-only projection; canonical sources stay local and intact.

Keep source text in its exchange, with qualifications beside it. Remove another
view's copy only when the source and owner match. No quote dictionary, generated
summary, new model call, or truncation of source text is involved.
"""
import copy


def readable_planning_context(context):
    view = copy.deepcopy(context)
    sources = []
    for record in view.get('clash_records', []):
        for chain in record.get('response_chains', []):
            # These pairs commonly describe the very same source-bearing node.
            for earlier, later, flag in (
                ('target', 'latest_opponent_reply', 'also_selected_target'),
                ('our_prior_position', 'our_objection', 'also_our_prior_position'),
            ):
                if chain.get(earlier) is not None and chain[earlier] == chain.get(later):
                    chain[later][flag] = True
                    del chain[earlier]
            for value in chain.values():
                candidates = value if isinstance(value, list) else [value]
                sources.extend(s for s in candidates if isinstance(s, dict)
                               and isinstance(s.get('quote'), str) and s.get('node_id'))
        # Small historical entries remain unless the chain contains their exact
        # owner, relation, status and source (including any clipped continuation).
        retained = []
        for entry in record.get('entries', []):
            covered = any(all(entry.get(k) == s.get(k) for k in
                              ('node_id', 'side', 'relation', 'responds_to', 'status'))
                          and (s['quote'] == entry.get('excerpt') or
                               (entry.get('excerpt_truncated') and
                                s['quote'].startswith(entry.get('excerpt', '').removesuffix('…'))))
                          for s in sources)
            if not covered:
                retained.append(entry)
        if 'entries' in record:
            record['entries_in_response_chains'] = len(record['entries']) - len(retained)
            record['entries'] = retained

    opponent = 'against' if context.get('our_side') == 'for' else 'for'
    targets = view.get('tree_targets', [])
    by_node = {}
    for source in sources:
        by_node.setdefault(source['node_id'], []).append(source)

    # Keep target order and indices exactly as the canonical parser expects.
    for target_index, target in enumerate(targets):
        target['target_index'] = target_index
        matching = [s for s in by_node.get(target['node_id'], [])
                    if s.get('side') == target.get('side', opponent)
                    and s.get('status') == 'current']
        original_quotes = target.get('sources', [])
        displayed = [s['quote'] for s in matching]
        target['sources'] = [q for q in original_quotes if q not in displayed]
        # Supporting prose already reproduced verbatim in this node's sources
        # adds no information; genuinely different explanations remain visible.
        target['arguments'] = [a for a in target.get('arguments', [])
                               if not (isinstance(a, str) and len(a) >= 32
                                       and any(a in q for q in original_quotes))]
        target['constraints'] = [condition for condition in target.get('constraints', [])
                                 if not any(condition in s.get('constraints', []) for s in matching)]

    # The flat ledger can duplicate an already inline, claim-owned condition.
    # Never merge equal quotations belonging to different nodes or speakers.
    supplemental = []
    for condition in view.get('constraints', []):
        owner = condition.get('node_id')
        target = next((t for t in targets if t['node_id'] == owner
                       and t.get('claim') == condition.get('claim')), None)
        attached = ([*target.get('constraints', [])] if target else [])
        if target:
            attached += [c for s in by_node.get(owner, [])
                         if s.get('side') == target.get('side', opponent)
                         and s.get('status') == 'current' for c in s.get('constraints', [])]
        plain = {k: v for k, v in condition.items() if k not in ('node_id', 'claim')}
        if plain not in attached:
            supplemental.append(condition)
    view['constraints'] = supplemental

    # Indexes remain output selection labels, not references replacing prose.
    # Put each boundary option beside its text/owner. Unmatched options retain
    # their verbatim text in an explicit supplemental list.
    additional = []
    for index, quote in enumerate(view.pop('position_limits', [])):
        opponent_sources = [s for s in sources if s.get('side') == opponent]
        containers = [c for s in opponent_sources for c in s.get('constraints', [])]
        containers += [c for t in targets for c in t.get('constraints', [])]
        containers += supplemental + opponent_sources
        exact = next((x for x in containers if x.get('quote') == quote), None)
        if exact is not None:
            exact.setdefault('boundary_indexes', []).append(index)
            continue
        source = next((s for s in opponent_sources if quote and quote in s['quote']), None)
        if source is not None:
            source.setdefault('boundary_options', []).append(dict(index=index, quote=quote))
            continue
        target = next((t for t in targets if any(quote and quote in q
                                               for q in t.get('sources', []))), None)
        option = dict(index=index, quote=quote)
        if target is not None:
            target.setdefault('boundary_options', []).append(option)
        else:
            additional.append(option)
    view['additional_boundaries'] = additional
    return view
