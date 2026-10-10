"""Exercise the real artifact-writing method without loading the model stack."""
import ast
import json
import logging
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from debate_app.config import ENGINE_ROOT


def claim_generation():
    import hashlib

    source = ENGINE_ROOT / 'src' / 'ouragents.py'
    module = ast.parse(source.read_text())
    cls = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == 'TreeDebater')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'claim_generation')
    scope = {
        'json': json, 'os': os, 're': re, 'hashlib': hashlib,
        'logger': logging.getLogger('artifact-test'), 'log_llm_io': Mock(),
        'propose_definition_prompt': '{motion} {act}',
        'ClaimPool': lambda side, **kw: SimpleNamespace(
            create_claim=lambda **kw: [[{'claim': side, 'definition': 'A test definition'}]],
        ),
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
    return scope['claim_generation']


class ArtifactTests(unittest.TestCase):
    def test_saved_pools_load_own_and_opponent_claims_without_generation(self):
        generate = claim_generation()
        pool_dir = ENGINE_ROOT / 'results' / 'deepseek-chat'
        motion = 'Learning to be a good writer still matters in the age of AI'
        for side in ('for', 'against'):
            with self.subTest(side=side):
                opponent = 'against' if side == 'for' else 'for'
                pool = pool_dir / f"{motion.replace(' ', '_').lower()}_pool_{side}.json"
                other = pool_dir / f"{motion.replace(' ', '_').lower()}_pool_{opponent}.json"
                player = SimpleNamespace(
                    pool_file=str(pool), side=side, oppo_side=opponent,
                    config=SimpleNamespace(claim_pool_limit=8),
                    _listening_prefix_enabled=lambda: False,
                    helper_client=Mock(side_effect=AssertionError('Must not generate claims')),
                )
                generate(player, 4)
                self.assertEqual(player.claim_pool, json.loads(pool.read_text())[:8])
                self.assertEqual(player.oppo_claim_pool, json.loads(other.read_text()))
                player.helper_client.assert_not_called()

    def test_claim_pools_contain_the_generated_side_and_support_arbitrary_motions(self):
        generate = claim_generation()
        for motion in ('A normal motion', 'Should AI/ML be taught?', 'Writing ' + '观点' * 900):
            with self.subTest(motion=motion[:40]), tempfile.TemporaryDirectory() as root:
                cwd = os.getcwd()
                player = SimpleNamespace(
                    motion=motion, side='for', oppo_side='against', act='support',
                    config=SimpleNamespace(model='test'), pool_file=None,
                    _listening_prefix_enabled=lambda: False,
                    claim_pool=[], oppo_claim_pool=[], helper_client=lambda **kw: ['None'],
                )
                try:
                    os.chdir(root)
                    generate(player, 1)
                finally:
                    os.chdir(cwd)
                files = list(Path(root).glob('*.json'))
                self.assertEqual(len(files), 2)
                for side in ('for', 'against'):
                    path = next(p for p in files if p.name.endswith(f'_pool_{side}.json'))
                    self.assertLessEqual(len(path.name.encode()), 255)
                    self.assertEqual(json.loads(path.read_text())[0][0]['claim'], side)
