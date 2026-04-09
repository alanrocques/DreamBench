"""Per-probe score aggregation and breakdown."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from dreambench.probes.base import ProbeResult


@dataclass
class ProbeScoreSummary:
    """Aggregated statistics for a single probe across scenarios."""

    probe_name: str
    mean: float
    std: float
    min: float
    max: float
    num_scenarios: int
    per_scenario: Dict[str, float]


def summarize_probe_results(results: List[ProbeResult]) -> Dict[str, ProbeScoreSummary]:
    """Group results by probe and compute summary statistics."""
    by_probe: Dict[str, List[ProbeResult]] = {}
    for r in results:
        by_probe.setdefault(r.probe_name, []).append(r)

    summaries = {}
    for probe_name, probe_results in by_probe.items():
        scores = np.array([r.score for r in probe_results])
        summaries[probe_name] = ProbeScoreSummary(
            probe_name=probe_name,
            mean=float(np.mean(scores)),
            std=float(np.std(scores)),
            min=float(np.min(scores)),
            max=float(np.max(scores)),
            num_scenarios=len(scores),
            per_scenario={r.scenario_name: r.score for r in probe_results},
        )

    return summaries
