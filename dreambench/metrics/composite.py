"""Composite scoring: aggregate probe scores into an overall benchmark result."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from dreambench.probes.base import ProbeResult


@dataclass
class CompositeScore:
    """Aggregated benchmark score across all probes and scenarios."""

    model_name: str
    probe_scores: Dict[str, float]  # probe_name -> mean score
    overall: float
    weights: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_results(
        cls,
        results: List[ProbeResult],
        model_name: str = "unknown",
        weights: Optional[Dict[str, float]] = None,
    ) -> "CompositeScore":
        """Compute composite score from a list of probe results.

        Args:
            results: List of ProbeResult from a benchmark run.
            model_name: Name of the model being evaluated.
            weights: Optional per-probe weights. If None, equal weighting.
        """
        by_probe: Dict[str, List[float]] = {}
        for r in results:
            by_probe.setdefault(r.probe_name, []).append(r.score)

        probe_scores = {k: float(np.mean(v)) for k, v in by_probe.items()}

        if weights:
            # Weighted average (only over probes that have results)
            total_weight = sum(weights.get(k, 1.0) for k in probe_scores)
            overall = sum(
                probe_scores[k] * weights.get(k, 1.0) for k in probe_scores
            ) / total_weight if total_weight > 0 else 0.0
        else:
            overall = float(np.mean(list(probe_scores.values()))) if probe_scores else 0.0

        return cls(
            model_name=model_name,
            probe_scores=probe_scores,
            overall=overall,
            weights=weights or {},
        )
