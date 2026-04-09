"""Tests for metrics: composite scoring, visualization, and report generation."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dreambench.metrics.composite import CompositeScore
from dreambench.metrics.per_probe import summarize_probe_results
from dreambench.metrics.visualization import create_radar_chart
from dreambench.probes.base import ProbeResult


def _make_results():
    return [
        ProbeResult(probe_name="reward_fidelity", scenario_name="s1", score=0.9),
        ProbeResult(probe_name="reward_fidelity", scenario_name="s2", score=0.7),
        ProbeResult(probe_name="object_permanence", scenario_name="s3", score=0.5),
        ProbeResult(probe_name="physics_consistency", scenario_name="s4", score=0.8),
        ProbeResult(probe_name="entity_integrity", scenario_name="s5", score=0.6),
        ProbeResult(probe_name="temporal_coherence", scenario_name="s6", score=1.0),
    ]


class TestCompositeScore:
    def test_from_results(self):
        results = _make_results()
        cs = CompositeScore.from_results(results, model_name="test")
        assert cs.model_name == "test"
        assert 0.0 <= cs.overall <= 1.0
        assert len(cs.probe_scores) == 5
        assert abs(cs.probe_scores["reward_fidelity"] - 0.8) < 1e-6

    def test_weighted(self):
        results = _make_results()
        weights = {"reward_fidelity": 2.0, "object_permanence": 1.0}
        cs = CompositeScore.from_results(results, weights=weights)
        assert cs.overall > 0

    def test_empty_results(self):
        cs = CompositeScore.from_results([])
        assert cs.overall == 0.0
        assert cs.probe_scores == {}


class TestPerProbeSummary:
    def test_summarize(self):
        results = _make_results()
        summaries = summarize_probe_results(results)
        assert "reward_fidelity" in summaries
        rf = summaries["reward_fidelity"]
        assert rf.num_scenarios == 2
        assert abs(rf.mean - 0.8) < 1e-6
        assert rf.min == 0.7
        assert rf.max == 0.9


class TestVisualization:
    def test_radar_chart_creates_figure(self):
        scores = {
            "reward_fidelity": 0.8,
            "object_permanence": 0.5,
            "physics_consistency": 0.7,
            "entity_integrity": 0.6,
            "temporal_coherence": 0.9,
        }
        fig = create_radar_chart(scores, model_name="test")
        assert fig is not None

    def test_radar_chart_saves_to_file(self):
        scores = {
            "reward_fidelity": 0.8,
            "object_permanence": 0.5,
            "physics_consistency": 0.7,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "radar.png"
            create_radar_chart(scores, output_path=path)
            assert path.exists()
            assert path.stat().st_size > 0


class TestReportGeneration:
    def test_generate_html(self):
        from scripts.generate_report import generate_html_report

        results_data = {
            "model": "mock",
            "overall_score": 0.75,
            "scores_by_probe": {"reward_fidelity": 0.8, "object_permanence": 0.5, "physics_consistency": 0.9},
            "scores_by_scenario": {"s1": 0.8, "s2": 0.5, "s3": 0.9},
            "details": [
                {"probe": "reward_fidelity", "scenario": "s1", "score": 0.8, "details": {}},
                {"probe": "object_permanence", "scenario": "s2", "score": 0.5, "details": {}},
                {"probe": "physics_consistency", "scenario": "s3", "score": 0.9, "details": {}},
            ],
        }

        with tempfile.TemporaryDirectory() as tmp:
            results_path = Path(tmp) / "results.json"
            with open(results_path, "w") as f:
                json.dump(results_data, f)

            output_path = Path(tmp) / "report.html"
            result = generate_html_report(results_path, output_path)
            assert result.exists()
            html = output_path.read_text()
            assert "mock" in html
            assert "DreamBench" in html
            assert "Reward Fidelity" in html
