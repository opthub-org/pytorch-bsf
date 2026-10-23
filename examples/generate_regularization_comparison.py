"""Generate reproducible geometry comparisons using the public fitting API.

Run: python examples/generate_regularization_comparison.py
Requires the plot and sparse extras. Writes docs/_static/fit_comparison.svg.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "fit-comparison"
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from torch_bsf import BezierSpace, FixedControls, QuadraticPenalty, Samples, fit  # noqa: E402
from torch_bsf.solvers import SparseLeastSquares  # noqa: E402


def main() -> None:
    space = BezierSpace.simplex(1, degree=5)
    x = torch.linspace(0, 1, 18, dtype=torch.float64)
    t = torch.stack([1 - x, x], 1)
    generator = torch.Generator().manual_seed(17)
    y = torch.sin(x * torch.pi) + 0.13 * torch.randn(18, generator=generator, dtype=torch.float64)
    samples = Samples(t, torch.stack([x, y], 1))
    fixed = FixedControls.from_cells(space, [(0, (5, 0), [0.0, 0.0]), (0, (0, 5), [1.0, 0.0])])
    settings = [
        ("Before: no penalty / 6 free controls", QuadraticPenalty(), None),
        ("After: ridge=0.01 / same space", QuadraticPenalty(ridge=0.01), None),
        ("After: smoothness=0.02 / same space", QuadraticPenalty(smoothness=0.02), None),
        ("After: fixed endpoints / 4 free controls", QuadraticPenalty(), fixed),
    ]
    q = torch.linspace(0, 1, 401, dtype=torch.float64)
    evaluation = torch.stack([1 - q, q], 1)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True, layout="constrained")
    for ax, (title, penalty, constraints) in zip(axes.flat, settings, strict=True):
        result = fit(space, samples, penalty=penalty, fixed=constraints).require_success()
        sparse = fit(
            space, samples, penalty=penalty, fixed=constraints, solver=SparseLeastSquares(atol=1e-12, btol=1e-12, max_iter=200)
        ).require_success()
        assert torch.allclose(result.model(evaluation), sparse.model(evaluation), atol=1e-8)
        with torch.no_grad():
            p, curve = result.model.coefficients.numpy(), result.model(evaluation).numpy()
            ax.plot(q, torch.sin(q * torch.pi), ":", color=".3", label="noise-free target")
            ax.scatter(x, y, color=".5", s=15, label="same observations")
            ax.plot(p[:, 0], p[:, 1], "o--", color="#d97706", alpha=0.7, label="control net")
            ax.plot(curve[:, 0], curve[:, 1], color="#0369a1", lw=2, label="fitted image (dense = sparse)")
        ax.set(title=title, xlabel="output 1", ylabel="output 2", xlim=(-0.12, 1.12), ylim=(-0.3, 2.0))
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    destination = Path(__file__).resolve().parents[1] / "docs/_static/fit_comparison.svg"
    fig.savefig(destination, metadata={"Date": None})
    destination.write_text(
        "\n".join(line.rstrip() for line in destination.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
