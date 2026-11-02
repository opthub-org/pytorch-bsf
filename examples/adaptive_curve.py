"""Write a self-contained visual example of budgeted input-chart discovery.

Run ``python examples/adaptive_curve.py --output build/adaptive_curve.html``
after installing ``pytorch-bsf[adaptive,plot]``. The known formula supplies a
demonstration, not a benchmark or a boundary label available to the learner.
"""

import argparse
import html
import io
import json
from pathlib import Path

import torch

from torch_bsf import BezierSpace, Samples, fit, fit_adaptive_curve


def response(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.sin(3 * x), torch.cos(5 * x)), 1) + (x >= 0.413).double() * x.new_tensor([1.0, 0.3])


def write_example(output: Path, budget: int) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    batches = []

    def oracle(x):
        values = response(x)
        batches.append((x.clone(), values.clone()))
        return values

    result = fit_adaptive_curve(oracle, (0, 1), budget=budget)
    initial_x, initial_y = batches[0]
    initial = fit(BezierSpace.simplex(1, degree=3), Samples(torch.cat((1 - initial_x, initial_x), 1), initial_y))
    initial.require_success()
    # These display-only responses are obtained after learning has finished.
    # They are never supplied to the learner or used to select a predictor.
    grid = torch.linspace(0, 1, 513, dtype=torch.float64)[:, None]
    truth = response(grid)
    connected = initial.model(torch.cat((1 - grid, grid), 1)).detach()
    predicted = result.predict(grid)
    chronology = torch.cat([x for x, _ in batches])[:, 0]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, values, title in [
        (axes[0, 0], connected, "Initial single simplex, degree 3"),
        (axes[0, 1], predicted, "Final BSF charts"),
    ]:
        # Break the known display curve at its jump instead of drawing a
        # fictitious connecting line. This truth is for rendering only.
        for use in (grid[:, 0] < 0.413, grid[:, 0] >= 0.413):
            axis.plot(truth[use, 0], truth[use, 1], color="0.7", linewidth=3)
        axis.plot(values[:, 0], values[:, 1], color="tab:blue", label="BSF prediction")
        observed = initial_y if axis is axes[0, 0] else result.values
        axis.scatter(observed[:, 0], observed[:, 1], s=12, color="tab:orange", label="Measured responses")
        axis.set(xlabel="Response 1", ylabel="Response 2", title=title)
        axis.legend(fontsize=8)
    axes[1, 0].scatter(chronology, torch.arange(len(chronology)), c=torch.arange(len(chronology)), s=14, cmap="viridis")
    axes[1, 0].axvline(0.413, color="0.6", linestyle="--", label="Known demonstration boundary")
    axes[1, 0].set(xlabel="Queried parameter", ylabel="Charged response row", title="Acquisition order")
    error = (predicted - truth).norm(dim=1)
    axes[1, 1].plot(grid[:, 0], error)
    axes[1, 1].set(xlabel="Parameter", ylabel="Vector error norm", title="Display-grid error and excluded brackets")
    for gap in result.gaps:
        for axis in axes[1]:
            axis.axvspan(gap.left, gap.right, color="tab:red", alpha=0.3)
    image = io.StringIO()
    figure.savefig(image, format="svg", metadata={"Date": None})
    plt.close(figure)
    summary = {
        "learning_response_rows": result.cost,
        "display_response_rows_after_learning": len(grid),
        "initial_response_rows_reused_from_learning": len(initial_x),
        "input_chart_components": result.domain.betti_numbers()[0],
        "polynomial_intervals": len(result.intervals),
        "excluded_brackets": [[gap.left, gap.right] for gap in result.gaps],
        "status": result.status,
        "profile": result.diagnostics["profile"],
    }
    assert result.cost == sum(len(x) for x, _ in batches)
    svg = image.getvalue()[image.getvalue().index("<svg") :]
    content = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Adaptive BSF curve example</title><style>
body{{font:16px sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}}
svg{{max-width:100%;height:auto}}pre{{white-space:pre-wrap}}</style>
<h1>From one connected guess to provisional input charts</h1>
<p>The learner receives only queried parameter/response pairs. The initial cubic
uses the first charged measurements again, without another response query.
The grey curve and dashed boundary are known demonstration truth used only
after learning for this plot. The display-grid error is not a benchmark.</p>
{svg}<pre>{html.escape(json.dumps(summary, indent=2))}</pre>
<p>Red brackets are unresolved and predictions inside them remain NaN.
The component count describes the proposed parameter charts, not certified
topology of their output image. A finite budget can miss other features.</p>
</html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/adaptive_curve.html"))
    parser.add_argument("--budget", type=int, default=64)
    arguments = parser.parse_args()
    if arguments.budget < 9:
        parser.error("The example requires at least nine initial measurements.")
    torch.set_num_threads(1)
    print(json.dumps(write_example(arguments.output, arguments.budget), indent=2))


if __name__ == "__main__":
    main()
