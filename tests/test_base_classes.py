"""Tests for base abstract classes and data structures."""

import numpy as np
import pytest

from dreambench.adapters.base import WorldModelAdapter
from dreambench.envs.base import BaseEnvWrapper, Scenario, Trajectory
from dreambench.probes.base import BaseProbe, ProbeResult


class TestWorldModelAdapter:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            WorldModelAdapter()

    def test_concrete_subclass(self):
        class DummyAdapter(WorldModelAdapter):
            def reset(self, initial_obs):
                self.state = initial_obs

            def step(self, action):
                return self.state, 0.0, False

            def get_latent(self):
                return self.state

        adapter = DummyAdapter()
        obs = np.zeros((84, 84, 3))
        adapter.reset(obs)
        next_obs, reward, done = adapter.step(0)
        assert next_obs.shape == (84, 84, 3)
        assert reward == 0.0
        assert done is False


class TestBaseProbe:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseProbe()

    def test_concrete_subclass(self):
        class DummyProbe(BaseProbe):
            name = "dummy"

            def __call__(self, predicted_frames, gt_frames,
                         predicted_rewards, gt_rewards, metadata):
                return ProbeResult(
                    probe_name=self.name,
                    scenario_name=metadata.get("name", ""),
                    score=1.0,
                )

        probe = DummyProbe()
        result = probe([], [], [], [], {"name": "test"})
        assert result.score == 1.0
        assert result.probe_name == "dummy"


class TestBaseEnvWrapper:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseEnvWrapper()


class TestProbeResult:
    def test_valid_score(self):
        result = ProbeResult(probe_name="test", scenario_name="s1", score=0.5)
        assert result.score == 0.5

    def test_invalid_score_raises(self):
        with pytest.raises(ValueError):
            ProbeResult(probe_name="test", scenario_name="s1", score=1.5)

        with pytest.raises(ValueError):
            ProbeResult(probe_name="test", scenario_name="s1", score=-0.1)


class TestScenario:
    def test_from_dict(self):
        data = {
            "name": "test_scenario",
            "env_id": "Breakout-v4",
            "actions": [0, 1, 2],
            "probe": "reward_fidelity",
            "description": "A test scenario",
        }
        scenario = Scenario.from_dict(data)
        assert scenario.name == "test_scenario"
        assert scenario.env_id == "Breakout-v4"
        assert scenario.actions == [0, 1, 2]
        assert scenario.probe == "reward_fidelity"


class TestTrajectory:
    def test_creation(self):
        traj = Trajectory(
            observations=[np.zeros((84, 84, 3))],
            rewards=[1.0],
            dones=[False],
        )
        assert len(traj.observations) == 1
        assert traj.rewards == [1.0]
