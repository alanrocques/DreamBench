"""Central orchestrator that connects adapters, environments, and probes."""

import importlib
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np

from dreambench.adapters.base import WorldModelAdapter
from dreambench.envs.base import BaseEnvWrapper, Scenario, Trajectory, load_scenarios
from dreambench.probes.base import BaseProbe, ProbeResult

logger = logging.getLogger(__name__)

# Registry of built-in probes
PROBE_REGISTRY: Dict[str, Type[BaseProbe]] = {}


def register_probe(name: str, probe_cls: Type[BaseProbe]) -> None:
    PROBE_REGISTRY[name] = probe_cls


def _load_default_probes() -> None:
    """Register built-in probes."""
    from dreambench.probes.reward_fidelity import RewardFidelityProbe
    from dreambench.probes.object_permanence import ObjectPermanenceProbe
    from dreambench.probes.physics_consistency import PhysicsConsistencyProbe
    from dreambench.probes.entity_integrity import EntityIntegrityProbe
    from dreambench.probes.temporal_coherence import TemporalCoherenceProbe

    register_probe("reward_fidelity", RewardFidelityProbe)
    register_probe("object_permanence", ObjectPermanenceProbe)
    register_probe("physics_consistency", PhysicsConsistencyProbe)
    register_probe("entity_integrity", EntityIntegrityProbe)
    register_probe("temporal_coherence", TemporalCoherenceProbe)


# Auto-register on import
_load_default_probes()


def import_class(dotted_path: str) -> type:
    """Import a class from a dotted module path like 'package.module.ClassName'."""
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


@dataclass
class BenchmarkResult:
    """Aggregated results from a full benchmark run."""

    model_name: str
    results: List[ProbeResult] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        if not self.results:
            return 0.0
        return float(np.mean([r.score for r in self.results]))

    def scores_by_probe(self) -> Dict[str, float]:
        """Average score per probe type."""
        by_probe: Dict[str, List[float]] = {}
        for r in self.results:
            by_probe.setdefault(r.probe_name, []).append(r.score)
        return {k: float(np.mean(v)) for k, v in by_probe.items()}

    def scores_by_scenario(self) -> Dict[str, float]:
        """Score per scenario."""
        return {r.scenario_name: r.score for r in self.results}


class BenchmarkRunner:
    """Runs diagnostic scenarios through a world model and evaluates with probes."""

    def __init__(
        self,
        adapter: WorldModelAdapter,
        env_wrapper: BaseEnvWrapper,
        probes: Optional[Dict[str, BaseProbe]] = None,
    ):
        self.adapter = adapter
        self.env_wrapper = env_wrapper
        self.probes = probes or {name: cls() for name, cls in PROBE_REGISTRY.items()}

    def run_model_rollout(
        self, scenario: Scenario, gt_trajectory: Trajectory
    ) -> Trajectory:
        """Run the world model on a scenario using the ground-truth initial obs."""
        # If the adapter supports GT replay (e.g. MockAdapter), provide GT data
        if hasattr(self.adapter, "set_ground_truth"):
            self.adapter.set_ground_truth(
                gt_trajectory.observations, gt_trajectory.rewards
            )

        initial_obs = gt_trajectory.observations[0]
        self.adapter.reset(initial_obs)

        observations = [initial_obs]
        rewards = []
        dones = []

        for action in scenario.actions:
            obs, reward, done = self.adapter.step(action)
            observations.append(obs)
            rewards.append(reward)
            dones.append(done)
            if done:
                break

        return Trajectory(observations=observations, rewards=rewards, dones=dones)

    def evaluate_scenario(self, scenario: Scenario) -> List[ProbeResult]:
        """Run a single scenario and evaluate with the appropriate probe(s)."""
        logger.info("Running scenario: %s", scenario.name)

        # Get ground truth
        gt_trajectory = self.env_wrapper.run_ground_truth(scenario)

        # Get model prediction
        pred_trajectory = self.run_model_rollout(scenario, gt_trajectory)

        # Run probe(s)
        results = []
        probe_names = (
            [scenario.probe] if scenario.probe in self.probes else list(self.probes)
        )

        metadata = {
            "name": scenario.name,
            "env_id": scenario.env_id,
            "description": scenario.description,
            **scenario.metadata,
        }

        for probe_name in probe_names:
            if probe_name not in self.probes:
                logger.warning("Probe '%s' not found, skipping", probe_name)
                continue

            probe = self.probes[probe_name]
            result = probe(
                predicted_frames=pred_trajectory.observations,
                gt_frames=gt_trajectory.observations,
                predicted_rewards=pred_trajectory.rewards,
                gt_rewards=gt_trajectory.rewards,
                metadata=metadata,
            )
            results.append(result)
            logger.info(
                "  %s / %s: score=%.3f", scenario.name, probe_name, result.score
            )

        return results

    def run(
        self, scenarios: List[Scenario], model_name: str = "unknown"
    ) -> BenchmarkResult:
        """Run the full benchmark across all scenarios."""
        benchmark_result = BenchmarkResult(model_name=model_name)

        for scenario in scenarios:
            try:
                results = self.evaluate_scenario(scenario)
                benchmark_result.results.extend(results)
            except Exception as e:
                logger.error("Scenario '%s' failed: %s", scenario.name, e)
                benchmark_result.results.append(
                    ProbeResult(
                        probe_name=scenario.probe,
                        scenario_name=scenario.name,
                        score=0.0,
                        details={"error": str(e)},
                    )
                )

        logger.info(
            "Benchmark complete. Overall score: %.3f", benchmark_result.overall_score
        )
        return benchmark_result
