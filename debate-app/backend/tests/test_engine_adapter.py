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
from streaming.planning import IncrementalPlanner, PlanningConfig


class AdapterTests(unittest.TestCase):
    def test_streamed_input_uses_policy_and_restore_discards_speculative_work(self):
        with tempfile.TemporaryDirectory() as root:
            engine = TreeDebaterEngine({}, root)
            policy = IncrementalPlanner(PlanningConfig(mode="linear"))
            policy.start("for:opening")
            player = SimpleNamespace(planner=policy, _planning_turn_snapshot=None,
                                     observe_opponent=Mock(), status="opening",
                                     conversation=[], used_evidence=set())
            engine.players = {"against": player}
            engine.trees = Mock(return_value={})
            engine.checkpoint()
            policy.observe("All cars should be banned.", llm=lambda *a: "Attack the broad scope",
                           analyze=Mock(), context=lambda: {})
            engine.analyze("Only private cars downtown.", "for", "opening")
            player.observe_opponent.assert_called_once_with("Only private cars downtown.", "for", "opening")
            self.assertEqual(player.conversation, [])
            self.assertEqual(player.used_evidence, set())
            engine.restore()
            self.assertEqual(player.planner.chunks, [])
            self.assertEqual(player.planner.plan, "")

    def test_planning_settings_validate_and_reach_engine_config(self):
        for mode in ("legacy", "end_of_turn", "linear", "adaptive_linear", "branch_tree", "flat_tree"):
            settings = SessionSettings(motion="Limit cars downtown", planning={"mode": mode})
            self.assertEqual(settings.model_dump()["planning"], {"mode": mode})
        with self.assertRaises(ValueError):
            SessionSettings(motion="Limit cars downtown", planning={"mode": "unknown"})

    def test_rerecord_discards_running_prefix_before_restoring_input(self):
        with tempfile.TemporaryDirectory() as root:
            engine = TreeDebaterEngine({}, root)
            player = SimpleNamespace(conversation=[], discard_listening_prefix=Mock())
            engine.players = {'against': player}
            engine.trees = Mock(return_value={})
            engine.checkpoint()
            player.conversation.append({'role': 'user', 'content': 'Discarded recording.'})
            engine.restore()
            player.discard_listening_prefix.assert_called_once_with()
            self.assertEqual(player.conversation, [])

    def test_listening_prefix_preset_validates_with_full_stage_budgets(self):
        import yaml
        raw = yaml.safe_load((ENGINE_ROOT / 'debate-app/configs/gemma-flat-listening-prefix.yml').read_text())
        settings = SessionSettings(**raw)
        self.assertEqual(settings.planning['mode'], 'flat_tree')
        self.assertEqual(settings.streaming['output']['speech_mode'], 'listening_prefix')
        self.assertEqual(settings.budgets, {'opening': 240, 'rebuttal': 240, 'closing': 120})
        self.assertFalse(settings.streaming['output']['first_chunk_local_tempo'])
        self.assertEqual(settings.ai_model, 'google.gemma-4-26b-a4b')
        self.assertEqual(settings.helper_model, settings.ai_model)
        self.assertTrue(settings.rehearsal.enabled)
        self.assertEqual(settings.rehearsal.pool_name, 'gemma-4-26b-a4b')
        self.assertEqual(settings.rehearsal.mode, 'hybrid')
        self.assertNotIn('listening_body_review_model', settings.streaming['output'])
        self.assertTrue(settings.streaming['output']['allow_expansion'])
        self.assertNotIn('verify_rewrites', settings.streaming['output'])
        self.assertEqual(settings.streaming['output']['early_max_refinements'], 3)
        self.assertEqual(settings.streaming['output']['max_refinements'], 10)
        self.assertEqual(settings.streaming['output']['max_parallel_tts'], 8)
        self.assertEqual(settings.streaming['output']['refinement_model'], settings.ai_model)

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
                settings = SessionSettings(motion='Test API scoring', mode=mode,
                    claim_selection_strategy='saved_scores',
                    budgets={'opening': 46, 'rebuttal': 92, 'closing': 69})
                engine = TreeDebaterEngine({**settings.model_dump(), "worker_side": side}, Path(root))
                engine.trees = Mock(return_value={})
                cwd = os.getcwd()
                try:
                    with patch.dict(sys.modules, {'agents': agents, 'ouragents': ouragents}):
                        engine.prepare()
                finally:
                    os.chdir(cwd)
                self.assertEqual(player.claim_generation.call_count, expected_count)
                self.assertEqual((player.speech_budgets.opening, player.speech_budgets.rebuttal,
                                  player.speech_budgets.closing), (46, 92, 69))
                if side:
                    self.assertEqual(list(engine.players), [side])
                for call in player.claim_generation.call_args_list:
                    self.assertIs(call.kwargs['use_rm_model'], False)
                self.assertIsNone(ouragents.TreeDebater.call_args.args[0].pool_file)
                self.assertEqual(ouragents.TreeDebater.call_args.args[0].claim_selection_strategy, 'saved_scores')

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

    def test_enabled_retrieval_loads_gemma_pools_and_warms_before_speaking(self):
        agents = ModuleType('agents')
        agents.DebaterConfig = SimpleNamespace
        ouragents = ModuleType('ouragents')
        ouragents.TreeDebater = Mock()
        motion = 'Social media should be required to verify user identities'
        with tempfile.TemporaryDirectory() as root:
            settings = SessionSettings(motion=motion, mode='ai_ai', rehearsal={'enabled': True})
            engine = TreeDebaterEngine(settings.model_dump(), Path(root))
            engine.trees = Mock(return_value={})
            cwd = os.getcwd()
            try:
                with patch.dict(sys.modules, {'agents': agents, 'ouragents': ouragents}):
                    engine.prepare()
            finally:
                os.chdir(cwd)
            self.assertEqual(ouragents.TreeDebater.call_count, 2)
            for call in ouragents.TreeDebater.call_args_list:
                cfg = call.args[0]
                self.assertTrue(cfg.use_rehearsal_tree)
                self.assertFalse(cfg.use_retrieval)
                self.assertEqual(Path(cfg.pool_file).parent.name, 'gemma-4-26b-a4b')
                self.assertEqual(Path(cfg.rehearsal_index_cache_dir), Path(cfg.pool_file).parent / 'retrieval_indexes')
            self.assertEqual(ouragents.TreeDebater.return_value._warm_rehearsal_indexes.call_count, 2)

    def test_enabled_retrieval_requires_saved_pools_without_paid_generation_fallback(self):
        agents = ModuleType('agents')
        agents.DebaterConfig = SimpleNamespace
        ouragents = ModuleType('ouragents')
        ouragents.TreeDebater = Mock()
        with tempfile.TemporaryDirectory() as root:
            settings = SessionSettings(motion='Missing Gemma rehearsal motion', rehearsal={'enabled': True})
            engine = TreeDebaterEngine(settings.model_dump(), Path(root))
            cwd = os.getcwd()
            try:
                with patch.dict(sys.modules, {'agents': agents, 'ouragents': ouragents}):
                    with self.assertRaisesRegex(ValueError, 'pools for both sides'):
                        engine.prepare()
            finally:
                os.chdir(cwd)
            ouragents.TreeDebater.assert_not_called()
