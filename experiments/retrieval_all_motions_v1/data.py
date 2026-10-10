"""Offline provenance-aware extraction of motions, prepared trees and live queries."""
import ast
import json
import re


def slug(motion):
    return motion.strip().casefold().replace(' ', '_')


def blocks(text):
    headers=list(re.finditer(r'^\d{4}-\d\d-\d\d .*?\[io\] (?P<header>[^\n]+)\n-+\n',text,re.M))
    for i,m in enumerate(headers):
        body=text[m.end():headers[i+1].start() if i+1<len(headers) else len(text)]
        yield m.group('header'),body,m.start()


def configs(text):
    for line,text in enumerate(text.splitlines(),1):
        if 'Config: ' in text:
            value=ast.literal_eval(text.split('Config: ',1)[1])
            yield line,value


def recover_trees(body, motion, side):
    roots=[];stack=[];count=0
    for line in body.splitlines():
        match=re.match(r'^\s*Level-(\d+) [^:]+:\s*(\{.*)',line)
        if not match:continue
        level=int(match[1]);value,end=json.JSONDecoder().raw_decode(match[2]);suffix=match[2][end:]
        def score(name):
            m=re.search(name+r' Score: ([-+\d.]+)',suffix)
            return float(m[1]) if m else 0.
        node=dict(side=side if level%2==0 else ('for' if side=='against' else 'against'),
                  level=level,claim=value['claim'],argument=value['argument'],evidence=[],status='prepared',
                  visit_count=0,scores=dict(defense=score('Attack'),support=score('Support')),children=[])
        if level==0:
            roots.append(dict(motion=motion,side=side,structure=node));stack=[node]
        else:
            if level>len(stack):raise ValueError('Missing parent in printed tree')
            stack=stack[:level];stack[-1]['children'].append(node);stack.append(node)
        count+=1
    if not roots:raise ValueError('No recoverable trees in claim-selection prompt')
    return roots,count


def historical_queries(document, source):
    """Arguments come only from preceding analyses in the same player's history."""
    result=[]
    for player,thoughts in document['debate_thoughts'].items():
        seen={}
        for i,t in enumerate(thoughts):
            if t.get('mode')=='analyze_statement':
                for claim in t.get('claims',[]):
                    if not claim.get('claim'):continue
                    argument=claim.get('arguments',[])
                    if isinstance(argument,list):argument='\n'.join(str(a) for a in argument)
                    seen[t['side'],claim['claim'].strip()]=(argument,f'{source}#debate_thoughts/{player}/{i}')
            if t.get('mode')!='retrieve_on_prepared_tree' or t['action_type'] not in {'attack','reinforce','rebut'}:continue
            target_side=('for' if player=='against' else 'against') if t['action_type'] in {'attack','rebut'} else player
            argument,argument_source=seen.get((target_side,t['target_claim'].strip()),('',None))
            result.append(dict(slug=slug(document['motion']),action=t['action_type'],side=player,
                               target=t['target_claim'],target_argument=argument,argument_source=argument_source,
                               stage=t['stage'],source=f'{source}#debate_thoughts/{player}/{i}',query_origin='historical'))
    return result


def validate_generated(result):
    qs=result.get('queries') if isinstance(result,dict) else None
    if not isinstance(qs,list) or len(qs)!=15 or any(not isinstance(q,dict) for q in qs):raise ValueError('Expected15 generated queries')
    for action in ['attack','reinforce','rebut']:
        if sum(q.get('action')==action for q in qs)!=5:raise ValueError('Expected5 per action')
    if len({q.get('target_claim') for q in qs})!=15:raise ValueError('Duplicate generated targets')
    for q in qs:
        if any(not isinstance(q.get(k),str) or not q[k].strip() for k in ['target_claim','target_argument']):
            raise ValueError('Missing generated target/context')
    return qs
