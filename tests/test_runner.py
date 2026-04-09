"""Tests for the benchmark runner and end-to-end pipeline."""

import numpy as np
import pytest

from dreambench.adapters.mock import MockAdapter
from dreambench.envs.base import BaseEnvWrapper, Scenario, Trajectory
from dreambench.probes.reward_fidelity import RewardFidelityProbe
from dreambench.runner import BenchmarkRunner, BenchmarkResult


class FakeEnvWrapper(BaseEnvWrapper):
    """Deterministic env wrapper for testing."""

    def run_ground_truth(self, scenario: Scenario) -> Trajectory:
        obs = np.zeros((84, 84, 3), dtype=np.uint8)
        observations = [obs]
        rewards = []
        dones = []
        for i, action in enumerate(scenario.actions):
            observations.append(obs.copy())
            rewards.append(1.0 if action == 1 else 0.0)
            dones.append(False)
        return Trajectory(observations=observations, rewards=rewards, dones=dones)

    def get_action_space_size(self) -> int:
        return 4


class TestMockAdapter:
    def test_reset_and_step(self):
        adapter = MockAdapter(noise_std=0.01)
        obs = np.zeros((84, 84, 3), dtype=np.uint8)
        adapter.reset(obs)

        next_obs, reward, done = adapter.step(0)
        assert next_obs.shape == (84, 84, 3)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_raises_without_reset(self):
        adapter = MockAdapter()
        with pytest.raises(RuntimeError):
            adapter.step(0)

    def test_get_latent(self):
        adapter = MockAdapter()
        obs = np.ones((10, 10), dtype=np.uint8)
        adapter.reset(obs)
        latent = adapter.get_latent()
        assert latent.shape == (100,)


class TestRewardFidelityProbe:
    def test_perfect_match(self):
        probe = RewardFidelityProbe()
        result = probe(
            predicted_frames=[],
            gt_frames=[],
            predicted_rewards=[1.0, 0.0, 1.0],
            gt_rewards=[1.0, 0.0, 1.0],
            metadata={"name": "test"},
        )
        assert result.score == 1.0
        assert result.probe_name == "reward_fidelity"

    def test_complete_mismatch(self):
        probe = RewardFidelityProbe()
        result = probe(
            predicted_frames=[],
            gt_frames=[],
            predicted_rewards=[1.0, 1.0, 1.0],
            gt_rewards=[0.0, 0.0, 0.0],
            metadata={"name": "test"},
        )
        assert result.score == 0.0

    def test_partial_match(self):
        probe = RewardFidelityProbe()
        result = probe(
            predicted_frames=[],
            gt_frames=[],
            predicted_rewards=[1.0, 0.0, 1.0, 0.0],
            gt_rewards=[1.0, 1.0, 0.0, 0.0],
            metadata={"name": "test"},
        )
        assert 0.0 < result.score < 1.0

    def test_empty_rewards(self):
        probe = RewardFidelityProbe()
        result = probe([], [], [], [], {"name": "test"})
        assert result.score == 1.0

    def test_length_mismatch_penalty(self):
        probe = RewardFidelityProbe()
        result = probe(
            predicted_frames=[],
            gt_frames=[],
            predicted_rewards=[1.0, 0.0],
            gt_rewards=[1.0, 0.0, 0.0, 0.0],
            metadata={"name": "test"},
        )
        # Perfect match on overlapping part, but penalized for length
        assert result.score < 1.0


class TestBenchmarkRunner:
    def test_end_to_end(self):
        adapter = MockAdapter(noise_std=0.0)
        env_wrapper = FakeEnvWrapper()
        probes = {"reward_fidelity": RewardFidelityProbe()}

        runner = BenchmarkRunner(
            adapter=adapter, env_wrapper=env_wrapper, probes=probes
        )

        scenarios = [
            Scenario(
                name="test_scenario",
                env_id="TestEnv-v0",
                actions=[0, 1, 0, 1],
                probe="reward_fidelity",
            ),
        ]

        result = runner.run(scenarios, model_name="mock")
        assert isinstance(result, BenchmarkResult)
        assert result.model_name == "mock"
        assert len(result.results) == 1
        assert 0.0 <= result.overall_score <= 1.0

    def test_multiple_scenarios(self):
        adapter = MockAdapter(noise_std=0.0)
        env_wrapper = FakeEnvWrapper()
        probes = {"reward_fidelity": RewardFidelityProbe()}
        runner = BenchmarkRunner(
            adapter=adapter, env_wrapper=env_wrapper, probes=probes
        )

        scenarios = [
            Scenario(name="s1", env_id="T-v0", actions=[0, 1], probe="reward_fidelity"),
            Scenario(name="s2", env_id="T-v0", actions=[1, 0, 1], probe="reward_fidelity"),
        ]

        result = runner.run(scenarios, model_name="mock")
        assert len(result.results) == 2
        assert "s1" in result.scores_by_scenario()
        assert "s2" in result.scores_by_scenario()

    def test_scores_by_probe(self):
        result = BenchmarkResult(model_name="test")
        from dreambench.probes.base import ProbeResult
        result.results = [
            ProbeResult(probe_name="reward_fidelity", scenario_name="s1", score=0.8),
            ProbeResult(probe_name="reward_fidelity", scenario_name="s2", score=0.6),
        ]
        by_probe = result.scores_by_probe()
        assert abs(by_probe["reward_fidelity"] - 0.7) < 1e-6

    def test_failed_scenario_gets_zero(self):
        """Scenarios that throw exceptions should get score 0."""
        class FailingWrapper(BaseEnvWrapper):
            def run_ground_truth(self, scenario):
                raise RuntimeError("env crashed")
            def get_action_space_size(self):
                return 4

        adapter = MockAdapter()
        runner = BenchmarkRunner(
            adapter=adapter,
            env_wrapper=FailingWrapper(),
            probes={"reward_fidelity": RewardFidelityProbe()},
        )

        scenarios = [
            Scenario(name="fail", env_id="T-v0", actions=[0], probe="reward_fidelity"),
        ]
        result = runner.run(scenarios, model_name="mock")
        assert result.results[0].score == 0.0
        assert "error" in result.results[0].details
