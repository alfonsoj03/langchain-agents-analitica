from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

from ml_pipeline.report import _fig_to_base64

matplotlib.use("Agg")

logger = logging.getLogger(__name__)


def _tree_figure_size(model: DecisionTreeRegressor) -> tuple[float, float]:
    n_nodes = model.tree_.node_count
    depth = model.get_depth()
    return (max(12.0, n_nodes * 0.35), max(8.0, depth * 1.8))


def _tree_image_html(model: DecisionTreeRegressor, feature_names: list[str]) -> str:
    fig, ax = plt.subplots(figsize=_tree_figure_size(model))
    plot_tree(
        model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=9,
        ax=ax,
    )
    ax.set_title("Decision Tree Structure")
    return f'<img src="data:image/png;base64,{_fig_to_base64(fig)}" />'


def _params_html(best_params: dict[str, Any] | None) -> str:
    if not best_params:
        return ""
    rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"
        for k, v in best_params.items()
    )
    return (
        "<h2>Best Hyperparameters</h2>"
        f"<table class='table'><tr><th>Parameter</th><th>Value</th></tr>{rows}</table>"
    )


def _metrics_html(metrics: dict[str, float] | None) -> str:
    if not metrics:
        return ""
    rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{v:.6f}</td></tr>" for k, v in metrics.items()
    )
    return (
        "<h2>Model Metrics (30% holdout)</h2>"
        f"<table class='table'><tr><th>Metric</th><th>Value</th></tr>{rows}</table>"
    )


def generate_tree_html_report(
    model: DecisionTreeRegressor,
    feature_names: list[str],
    target: str,
    output_path: str | Path,
    *,
    metrics: dict[str, float] | None = None,
    best_params: dict[str, Any] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Building tree HTML report: %d features, target=%s, nodes=%d, depth=%d",
        len(feature_names),
        target,
        model.tree_.node_count,
        model.get_depth(),
    )

    tree_text = export_text(model, feature_names=feature_names)
    tree_image_html = _tree_image_html(model, feature_names)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Decision Tree Model</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; }}
    .table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; }}
    .table th, .table td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: left; }}
    img {{ max-width: 100%; margin-bottom: 2rem; }}
    pre {{ background: #f5f5f5; padding: 1rem; overflow-x: auto; border: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>Decision Tree Model</h1>
  <h2>Model Overview</h2>
  <table class="table">
    <tr><th>Target</th><td>{html.escape(target)}</td></tr>
    <tr><th>Features</th><td>{len(feature_names)}</td></tr>
    <tr><th>Tree depth</th><td>{model.get_depth()}</td></tr>
    <tr><th>Node count</th><td>{model.tree_.node_count}</td></tr>
  </table>
  {_params_html(best_params)}
  {_metrics_html(metrics)}
  <h2>Tree Visualization</h2>
  {tree_image_html}
  <h2>Tree Rules (text)</h2>
  <pre>{html.escape(tree_text)}</pre>
</body>
</html>"""

    output_path.write_text(html_doc, encoding="utf-8")
    return output_path
