"""Tests for environment wrappers, scenario loading, and registries."""

from pathlib import Path

import pytest

from dreambench.envs.base import Scenario, load_scenarios
from dreambench.envs.registry import ENV_REGISTRY, get_env_wrapper
from dreambench.probes.registry import PROBE_REGISTRY, get_probe


SCENARIOS_DIR = Path(__file__).parent.parent / "dreambench" / "envs"


class TestScenarioLoading:
    def test_load_atari_scenarios(self):
        scenarios = load_scenarios(SCENARIOS_DIR / "atari" / "scenarios.yaml")
        assert len(scenarios) > 0
        for s in scenarios:
            assert isinstance(s, Scenario)
            assert s.name
            assert s.env_id
            assert len(s.actions) > 0
            assert s.probe

    def test_load_minigrid_scenarios(self):
        scenarios = load_scenarios(SCENARIOS_DIR / "minigrid" / "scenarios.yaml")
        assert len(scenarios) > 0
        for s in scenarios:
            assert isinstance(s, Scenario)
            assert "MiniGrid" in s.env_id or "minigrid" in s.env_id.lower()

    def test_load_crafter_scenarios(self):
        scenarios = load_scenarios(SCENARIOS_DIR / "crafter" / "scenarios.yaml")
        assert len(scenarios) > 0
        for s in scenarios:
            assert isinstance(s, Scenario)

    def test_all_scenarios_reference_known_probes(self):
        known_probes = {
            "reward_fidelity", "object_permanence", "physics_consistency",
            "entity_integrity", "temporal_coherence",
        }
        for env_dir in ["atari", "minigrid", "crafter"]:
            path = SCENARIOS_DIR / env_dir / "scenarios.yaml"
            scenarios = load_scenarios(path)
            for s in scenarios:
                assert s.probe in known_probes, (
                    f"Scenario '{s.name}' references unknown probe '{s.probe}'"
                )


class TestEnvRegistry:
    def test_all_envs_registered(self):
        assert "atari" in ENV_REGISTRY
        assert "minigrid" in ENV_REGISTRY
        assert "crafter" in ENV_REGISTRY

    def test_get_env_wrapper(self):
        wrapper = get_env_wrapper("atari")
        assert wrapper is not None

    def test_unknown_env_raises(self):
        with pytest.raises(KeyError):
            get_env_wrapper("nonexistent")


class TestProbeRegistry:
    def test_all_probes_registered(self):
        expected = [
            "reward_fidelity", "object_permanence", "physics_consistency",
            "entity_integrity", "temporal_coherence",
        ]
        for name in expected:
            assert name in PROBE_REGISTRY

    def test_get_probe(self):
        probe = get_probe("reward_fidelity")
        assert probe is not None

    def test_unknown_probe_raises(self):
        with pytest.raises(KeyError):
            get_probe("nonexistent")
