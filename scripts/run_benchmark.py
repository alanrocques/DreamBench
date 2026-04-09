"""Main entry point for running the DreamBench benchmark."""

import json
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig

from dreambench.envs.base import load_scenarios
from dreambench.metrics.per_probe import summarize_probe_results
from dreambench.runner import BenchmarkRunner, import_class

logger = logging.getLogger(__name__)


@hydra.main(config_path="../dreambench/configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    # Instantiate adapter
    adapter_cls = import_class(cfg.model.adapter)
    adapter_kwargs = {
        k: v for k, v in cfg.model.items() if k not in ("name", "adapter")
    }
    adapter = adapter_cls(**adapter_kwargs)
    logger.info("Loaded adapter: %s", cfg.model.name)

    # Instantiate env wrapper
    wrapper_cls = import_class(cfg.env.wrapper)
    env_wrapper = wrapper_cls()
    logger.info("Loaded env wrapper: %s", cfg.env.name)

    # Load scenarios
    scenarios_path = Path(cfg.env.scenarios_path)
    scenarios = load_scenarios(scenarios_path)
    logger.info("Loaded %d scenarios from %s", len(scenarios), scenarios_path)

    # Filter scenarios by probe if specified
    if cfg.probes:
        probe_filter = set(cfg.probes)
        scenarios = [s for s in scenarios if s.probe in probe_filter]
        logger.info("Filtered to %d scenarios for probes: %s", len(scenarios), probe_filter)

    # Run benchmark
    runner = BenchmarkRunner(adapter=adapter, env_wrapper=env_wrapper)
    result = runner.run(scenarios, model_name=cfg.model.name)

    # Output results
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results
    results_data = {
        "model": cfg.model.name,
        "overall_score": result.overall_score,
        "scores_by_probe": result.scores_by_probe(),
        "scores_by_scenario": result.scores_by_scenario(),
        "details": [
            {
                "probe": r.probe_name,
                "scenario": r.scenario_name,
                "score": r.score,
                "details": r.details,
            }
            for r in result.results
        ],
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # Print summary
    summaries = summarize_probe_results(result.results)
    print(f"\n{'=' * 60}")
    print(f"DreamBench Results: {cfg.model.name}")
    print(f"{'=' * 60}")
    print(f"Overall Score: {result.overall_score:.3f}")
    print(f"\nPer-Probe Breakdown:")
    for name, summary in summaries.items():
        print(f"  {name}: {summary.mean:.3f} (std={summary.std:.3f}, n={summary.num_scenarios})")
        for scenario, score in summary.per_scenario.items():
            print(f"    {scenario}: {score:.3f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
