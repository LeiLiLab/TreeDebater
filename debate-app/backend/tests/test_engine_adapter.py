import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from debate_app.engine_adapter import TreeDebaterEngine
from debate_app.schemas import SessionSettings
from debate_app.config import ENGINE_ROOT


class AdapterTests(unittest.TestCase):
    def test_preparation_uses_api_claim_scoring_for_each_ai_side(self):
        agents = ModuleType('agents')
        agents.DebaterConfig = SimpleNamespace
        ouragents = ModuleType('ouragents')
        ouragents.TreeDebater = Mock()
        for mode, side, expected_count in [('human_ai', None, 1), ('ai_ai', None, 2),
                                           ('ai_ai', 'for', 1), ('ai_ai', 'against', 1)]:
            with self.subTest(mode=mode, side=side), tempfile.TemporaryDirectory() as root:
                player = Mock()
                ouragents.TreeDebater.reset_mock()
                ouragents.TreeDebater.return_value = player
                settings = SessionSettings(motion='Test API scoring', mode=mode)
                engine = TreeDebaterEngine({**settings.model_dump(), "worker_side": side}, Path(root))
                engine.trees = Mock(return_value={})
                cwd = os.getcwd()
                try:
                    with patch.dict(sys.modules, {'agents': agents, 'ouragents': ouragents}):
                        engine.prepare()
                finally:
                    os.chdir(cwd)
                self.assertEqual(player.claim_generation.call_count, expected_count)
                if side:
                    self.assertEqual(list(engine.players), [side])
                for call in player.claim_generation.call_args_list:
                    self.assertIs(call.kwargs['use_rm_model'], False)
                self.assertIsNone(ouragents.TreeDebater.call_args.args[0].pool_file)

    def test_preparation_selects_saved_pool_for_each_ai_side(self):
        agents = ModuleType('agents')
        agents.DebaterConfig = SimpleNamespace
        ouragents = ModuleType('ouragents')
        ouragents.TreeDebater = Mock()
        motion = 'Learning to be a good writer still matters in the age of AI'
        with tempfile.TemporaryDirectory() as root:
            settings = SessionSettings(motion=motion, mode='ai_ai')
            engine = TreeDebaterEngine(settings.model_dump(), Path(root))
            engine.trees = Mock(return_value={})
            cwd = os.getcwd()
            try:
                with patch.dict(sys.modules, {'agents': agents, 'ouragents': ouragents}):
                    engine.prepare()
            finally:
                os.chdir(cwd)
            configs = [call.args[0] for call in ouragents.TreeDebater.call_args_list]
            self.assertEqual([cfg.side for cfg in configs], ['for', 'against'])
            for cfg in configs:
                expected = ENGINE_ROOT / 'results' / 'deepseek-chat' / (
                    motion.replace(' ', '_').lower() + f'_pool_{cfg.side}.json'
                )
                self.assertEqual(Path(cfg.pool_file), expected)
                self.assertTrue(expected.is_file())
