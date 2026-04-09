"""Generate an HTML diagnostic report from benchmark results."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from dreambench.metrics.composite import CompositeScore
from dreambench.metrics.visualization import create_radar_chart
from dreambench.probes.base import ProbeResult

logger = logging.getLogger(__name__)


def generate_html_report(results_path: Path, output_path: Path) -> Path:
    """Generate an HTML report from a results.json file.

    Args:
        results_path: Path to results.json from run_benchmark.py.
        output_path: Path for the output HTML file.

    Returns:
        Path to the generated HTML file.
    """
    try:
        from jinja2 import Environment, FileSystemLoader
    except ImportError:
        raise ImportError(
            "Report generation requires jinja2. Install with: pip install dreambench[viz]"
        )

    with open(results_path) as f:
        data = json.load(f)

    # Reconstruct ProbeResults for composite scoring
    probe_results = [
        ProbeResult(
            probe_name=d["probe"],
            scenario_name=d["scenario"],
            score=d["score"],
            details=d.get("details", {}),
        )
        for d in data["details"]
    ]

    composite = CompositeScore.from_results(probe_results, model_name=data["model"])

    # Ensure output directory exists before generating assets
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate radar chart
    radar_path = None
    if composite.probe_scores:
        radar_file = output_path.parent / f"{data['model']}_radar.png"
        fig = create_radar_chart(
            composite.probe_scores,
            model_name=data["model"],
            output_path=radar_file,
        )
        if fig is not None:
            radar_path = radar_file.name

    # Render HTML
    template_dir = Path(__file__).parent.parent / "dreambench" / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("report.html.j2")

    html = template.render(
        model_name=data["model"],
        overall_score=composite.overall,
        probe_scores=composite.probe_scores,
        details=data["details"],
        radar_chart_path=radar_path,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
        version="0.1.0",
    )

    output_path.write_text(html)
    logger.info("Report written to %s", output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate DreamBench HTML report")
    parser.add_argument(
        "--results", required=True, help="Path to results directory or results.json"
    )
    parser.add_argument(
        "--output", required=True, help="Output HTML file path"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    results_path = Path(args.results)
    if results_path.is_dir():
        results_path = results_path / "results.json"

    output_path = Path(args.output)
    generate_html_report(results_path, output_path)
    print(f"Report generated: {output_path}")


if __name__ == "__main__":
    main()
