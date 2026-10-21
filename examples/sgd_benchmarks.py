"""Compare SGD and preconditioned SGD on the benchmark problems of Hikima et al. (AISTATS 2025).

The script fits Bézier simplices to the Pareto sets of the following problems with
:func:`torch_bsf.fit_sgd`, by plain stochastic gradient descent (SGD, ``preconditioner="none"``) and
by the preconditioned one of the paper (PSGD, ``preconditioner="gram"``):

* ``bergou2020``: 2 objectives of 2 variables (Bergou et al., 2020);
* ``scaled_med``: the scaled-MED problem, 3 objectives of 3 variables;
* ``skew_mmd``: the skew-MMD problem, 3 objectives of 3 variables;
* ``facility_location``: the generalized location problem, 3 objectives of 4 variables.

The problem definitions are implemented from Section 5 of the paper.  The settings are those of the
paper: the initial control points ``P0 = O``, ``n = 100`` samples, minibatches of ``m = 20``,
``K = 100`` iterations with the step size ``alpha = 0.05``, the degrees ``D = 2, 3, 4`` and 5 trials
with different samples.  The generalization loss is evaluated on 10,000 other samples.

The loss of the scaled-MED, the location problem and Bergou2020 is quadratic in the control points, so
its minimizer ``P*`` over the Bézier simplices of degree ``D`` is computed exactly by Newton's method,
and the figures show the optimality gap ``L(P) - L(P*)``.  The skew-MMD has no closed form, so its
figure shows the loss ``L(P)`` itself.

Usage (matplotlib and SciPy are required, e.g., by the ``plot`` extra)::

    pip install -e ".[plot]"
    python examples/sgd_benchmarks.py

The script writes, for every problem ``<name>`` above, ``docs/_static/sgd_benchmark_<name>_loss.png``
(the history of the empirical and generalization losses: the mean and the range over the trials) and
``docs/_static/sgd_benchmark_<name>_pareto.png`` (the Pareto set and front approximated by PSGD in the
median trial, with the control points).  The median trial is the one of the median final empirical value
in the loss figure: the optimality gap, or the loss for the skew-MMD.  For the problems with a known Pareto
set (all but the skew-MMD), it also writes ``docs/_static/sgd_benchmark_<name>_mse.png``: the history of the
mean squared error ``MSE(P) = mean_i ||b(t_i | P) - x*(t_i)||^2`` to the Pareto set on the 10,000 samples of
the generalization loss (Figure 4 of the paper).  It runs in less than a minute on a CPU.
"""

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.tri import Triangulation  # noqa: E402

import torch_bsf  # noqa: E402
from torch_bsf.plotting import plot_bezier_simplex  # noqa: E402
from torch_bsf.sampling import simplex_grid  # noqa: E402
from torch_bsf.simploidal import BezierSimploid, linear_scalarization, sample_domain  # noqa: E402

DEGREES = (2, 3, 4)
N_TRIALS = 5
N_SAMPLES = 100  # n
BATCH_SIZE = 20  # m
MAX_ITER = 100  # K
STEP_SIZE = 0.05  # alpha
N_TEST_SAMPLES = 10_000
METHODS = {"SGD": "none", "PSGD": "gram"}
OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "_static"

Objective = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Problem:
    """A multi-objective optimization problem.

    ``objective`` maps points of shape ``(batch, n_variables)`` to the objective values of shape
    ``(batch, n_objectives)``.  ``pareto_set`` is the closed-form minimizer ``x*(t)`` of the linear
    scalarization ``t^T f(x)``, if known.  ``quadratic`` tells whether the objectives are quadratic, so
    that the loss is quadratic in the control points.
    """

    name: str
    title: str
    n_objectives: int
    n_variables: int
    objective: Objective
    quadratic: bool
    pareto_set: Callable[[torch.Tensor], torch.Tensor] | None = None


def bergou2020(x: torch.Tensor) -> torch.Tensor:
    """The two objectives of Bergou et al. (2020), Eq. (10) of the paper."""
    x1, x2 = x[:, 0], x[:, 1]
    return torch.stack([(x1 - 1) ** 2 + (x1 - x2) ** 2, (x2 - 3) ** 2 + (x1 - x2) ** 2], dim=1)


def bergou2020_pareto_set(t: torch.Tensor) -> torch.Tensor:
    """The minimizer of ``t1 f1(x) + t2 f2(x)`` given in the paper."""
    t1 = t[:, 0]
    denominator = t1**2 - t1 - 1
    return torch.stack([(t1**2 + t1 - 3) / denominator, (3 * t1**2 - t1 - 3) / denominator], dim=1)


# The scaled-MED: f_i(x) = sum_l W[i, l] (x_l - C[i, l])^2.
SCALED_MED_WEIGHTS = torch.tensor([[1.0, 3.0, 2.0], [2.0, 1.0, 3.0], [3.0, 2.0, 1.0]], dtype=torch.float64)
SCALED_MED_CENTERS = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, -1.0]], dtype=torch.float64)


def scaled_med(x: torch.Tensor) -> torch.Tensor:
    """The scaled-MED problem: three axis-aligned quadratics with different scales."""
    w, c = SCALED_MED_WEIGHTS.to(x), SCALED_MED_CENTERS.to(x)
    return (w * (x.unsqueeze(1) - c) ** 2).sum(dim=-1)


def scaled_med_pareto_set(t: torch.Tensor) -> torch.Tensor:
    """The minimizer of ``t^T f(x)``: the weighted mean of the centers, coordinate by coordinate."""
    w, c = SCALED_MED_WEIGHTS.to(t), SCALED_MED_CENTERS.to(t)
    return (t @ (w * c)) / (t @ w)


# The skew-MMD: f_i(x) = ||A_i (x - c_i)||_2^{p_i} with A_i diagonal, c_i = e_i and
# p_i = exp(2 (i - 1) / (M - 1) - 1).
SKEW_MMD_SCALES = torch.tensor([[0.6, 0.8, 0.8], [0.8, 0.6, 0.8], [0.8, 0.8, 0.6]], dtype=torch.float64)
SKEW_MMD_POWERS = torch.exp(2 * torch.arange(3, dtype=torch.float64) / 2 - 1)


def skew_mmd(x: torch.Tensor) -> torch.Tensor:
    """The skew-MMD problem with ``L = M = 3``."""
    a, c, p = SKEW_MMD_SCALES.to(x), torch.eye(3).to(x), SKEW_MMD_POWERS.to(x)
    return ((a * (x.unsqueeze(1) - c)) ** 2).sum(dim=-1).sqrt() ** p


def facility_location(x: torch.Tensor) -> torch.Tensor:
    """The generalized location problem: the squared distances from ``e_1, e_2, e_3`` in ``R^4``."""
    return ((x.unsqueeze(1) - torch.eye(3, x.shape[1]).to(x)) ** 2).sum(dim=-1)


def facility_location_pareto_set(t: torch.Tensor) -> torch.Tensor:
    """The minimizer of ``t^T f(x)``: ``x = t_1 e_1 + t_2 e_2 + t_3 e_3``."""
    return torch.cat([t, torch.zeros(len(t), 1).to(t)], dim=1)


PROBLEMS = (
    Problem("bergou2020", "Bergou2020", 2, 2, bergou2020, True, bergou2020_pareto_set),
    Problem("scaled_med", "Scaled-MED", 3, 3, scaled_med, True, scaled_med_pareto_set),
    Problem("skew_mmd", "Skew-MMD", 3, 3, skew_mmd, False),
    Problem("facility_location", "Location problem", 3, 4, facility_location, True, facility_location_pareto_set),
)


def initial_model(problem: Problem, degree: int) -> BezierSimploid:
    """The Bézier simplex with all control points at the origin (``P0 = O``), in double precision."""
    return BezierSimploid.zeros((problem.n_objectives - 1,), degree, n_values=problem.n_variables).double()


def mean_loss(loss: Any, model: BezierSimploid, inputs: list[torch.Tensor]) -> float:
    """The mean loss of ``model`` on ``inputs``."""
    with torch.no_grad():
        return float(loss(model(*inputs), *inputs).mean())


def optimal_loss(problem: Problem, init: BezierSimploid, inputs: list[torch.Tensor]) -> float:
    """The minimum mean loss over the Bézier simplices of the degree of ``init``, for a quadratic problem.

    The mean loss is quadratic in the control points, so one Newton step from any point reaches its
    minimizer; a second step removes the rounding errors.
    """
    loss = linear_scalarization(problem.objective)
    z = init.design_matrix(*inputs)
    shape = init.control_points.shape

    def mean(p: torch.Tensor) -> torch.Tensor:
        return loss(z @ p.view(shape), *inputs).mean()

    p = torch.zeros(math.prod(shape), dtype=z.dtype)
    for _ in range(2):
        hessian = torch.autograd.functional.hessian(mean, p)
        (grad,) = torch.autograd.grad(mean(p.requires_grad_()), p)
        p = (p - torch.linalg.solve(hessian, grad)).detach()
    return float(mean(p))


def mean_squared_error(problem: Problem, model: BezierSimploid, inputs: list[torch.Tensor]) -> float:
    """The mean of ``||b(t | P) - x*(t)||^2`` over the parameters ``t`` in ``inputs``."""
    assert problem.pareto_set is not None
    with torch.no_grad():
        return float(((model(*inputs) - problem.pareto_set(*inputs)) ** 2).sum(dim=1).mean())


def run_trial(
    problem: Problem, degree: int, preconditioner: Any, trial: int, test_inputs: list[torch.Tensor]
) -> tuple[BezierSimploid, dict[str, list[float]]]:
    """Fits a Bézier simplex once and records the losses at ``P^(0), ..., P^(K)``.

    If the Pareto set is known, it also records the MSE to it on ``test_inputs``.
    """
    init = initial_model(problem, degree)
    loss = linear_scalarization(problem.objective)
    inputs = [t.double() for t in sample_domain(init, N_SAMPLES, seed=trial)]
    history = {"empirical": [mean_loss(loss, init, inputs)], "generalization": [mean_loss(loss, init, test_inputs)]}
    if problem.pareto_set is not None:
        history["mse"] = [mean_squared_error(problem, init, test_inputs)]

    def record(k: int, model: BezierSimploid, batch_loss: float) -> None:
        history["empirical"].append(mean_loss(loss, model, inputs))
        history["generalization"].append(mean_loss(loss, model, test_inputs))
        if problem.pareto_set is not None:
            history["mse"].append(mean_squared_error(problem, model, test_inputs))

    model = torch_bsf.fit_sgd(
        init,
        loss,
        inputs=inputs,
        batch_size=BATCH_SIZE,
        max_iter=MAX_ITER,
        step_size=STEP_SIZE,
        preconditioner=preconditioner,
        seed=trial,
        callback=record,
    )
    if problem.quadratic:  # record the optimality gap instead of the loss
        for key, samples in (("empirical", inputs), ("generalization", test_inputs)):
            best = optimal_loss(problem, init, samples)
            history[key] = [v - best for v in history[key]]
    return model, history


def run_problem(problem: Problem) -> dict[tuple[int, str], Any]:
    """Runs every degree, method and trial.

    Returns the histories of shape ``(trial, iteration)`` and the model of the median trial, i.e., the one
    with the median final value of the empirical history (the optimality gap for a quadratic problem, the
    loss otherwise).
    """
    init = initial_model(problem, 1)
    test_inputs = [t.double() for t in sample_domain(init, N_TEST_SAMPLES, seed=10_000)]
    results: dict[tuple[int, str], Any] = {}
    for degree in DEGREES:
        for method, preconditioner in METHODS.items():
            models, histories = [], []
            for trial in range(N_TRIALS):
                model, history = run_trial(problem, degree, preconditioner, trial, test_inputs)
                models.append(model)
                histories.append(history)
            result = {key: np.array([h[key] for h in histories]) for key in histories[0]}
            median = np.argsort(result["empirical"][:, -1])[N_TRIALS // 2]
            results[degree, method] = {"model": models[median], **result}
    return results


def plot_losses(problem: Problem, results: dict[tuple[int, str], Any]) -> Path:
    """Plots the mean and the range of the losses over the trials, for every degree and method."""
    fig, axes = plt.subplots(2, len(DEGREES), figsize=(11, 6), sharex=True, sharey="row")
    quantity = r"$\mathcal{L}(P^{(k)}) - \mathcal{L}(P^*)$" if problem.quadratic else r"$\mathcal{L}(P^{(k)})$"
    for row, key in enumerate(("empirical", "generalization")):
        for col, degree in enumerate(DEGREES):
            ax = axes[row, col]
            for method in METHODS:
                values = results[degree, method][key]
                iterations = np.arange(values.shape[1])
                (line,) = ax.plot(iterations, values.mean(axis=0), label=method)
                ax.fill_between(iterations, values.min(axis=0), values.max(axis=0), color=line.get_color(), alpha=0.25)
            if problem.quadratic:
                ax.set_yscale("log")
            ax.grid(alpha=0.3)
            if row == 0:
                ax.set_title(f"$D = {degree}$")
            else:
                ax.set_xlabel("iteration $k$")
            if col == 0:
                ax.set_ylabel(f"{key.capitalize()}\n{quantity}")
    axes[0, 0].legend()
    fig.suptitle(f"{problem.title}: SGD vs. PSGD (mean and range over {N_TRIALS} trials)")
    fig.tight_layout()
    return save(fig, f"sgd_benchmark_{problem.name}_loss.png")


def plot_mse(problem: Problem, results: dict[tuple[int, str], Any]) -> Path:
    """Plots the mean and the range of the MSE to the Pareto set over the trials, for every degree and method."""
    fig, axes = plt.subplots(1, len(DEGREES), figsize=(11, 3.5), sharey=True)
    for ax, degree in zip(axes, DEGREES, strict=True):
        for method in METHODS:
            values = results[degree, method]["mse"]
            iterations = np.arange(values.shape[1])
            (line,) = ax.plot(iterations, values.mean(axis=0), label=method)
            ax.fill_between(iterations, values.min(axis=0), values.max(axis=0), color=line.get_color(), alpha=0.25)
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.set_title(f"$D = {degree}$")
        ax.set_xlabel("iteration $k$")
    axes[0].set_ylabel(r"$\mathrm{MSE}(P^{(k)})$")
    axes[0].legend()
    fig.suptitle(f"{problem.title}: MSE to the Pareto set (mean and range over {N_TRIALS} trials)")
    fig.tight_layout()
    return save(fig, f"sgd_benchmark_{problem.name}_mse.png")


def boundary(n_objectives: int, num: int = 100) -> list[torch.Tensor]:
    """The edges of the simplex of weights (the whole simplex for two objectives)."""
    s = torch.linspace(0, 1, num, dtype=torch.float64).unsqueeze(1)
    if n_objectives == 2:
        return [torch.cat([s, 1 - s], dim=1)]
    eye = torch.eye(3, dtype=torch.float64)
    return [(1 - s) * eye[i] + s * eye[(i + 1) % 3] for i in range(3)]


def draw(ax: Any, points: torch.Tensor, **kwargs: Any) -> None:
    """Draws a curve in 2D or 3D, given by the first coordinates of ``points``."""
    ax.plot(*points[:, : 3 if ax.name == "3d" else 2].T.numpy(), **kwargs)


def plot_pareto_front(ax: Any, problem: Problem, model: BezierSimploid) -> None:
    """Plots the image of the Bézier simplex under the objectives."""
    t = simplex_grid(n_params=problem.n_objectives, degree=30).double()
    with torch.no_grad():
        front = problem.objective(model(t)).numpy()
    if problem.n_objectives == 2:
        order = np.argsort(t[:, 0].numpy())
        ax.plot(front[order, 0], front[order, 1], label="PSGD")
    else:
        u = t.numpy()
        tri = Triangulation(u[:, 1] + 0.5 * u[:, 2], np.sqrt(3) / 2 * u[:, 2])
        ax.plot_trisurf(front[:, 0], front[:, 1], front[:, 2], triangles=tri.triangles, alpha=0.5)


def plot_pareto(problem: Problem, results: dict[tuple[int, str], Any]) -> Path:
    """Plots the Pareto set and front approximated by PSGD (the median trial) with the control points."""
    fig = plt.figure(figsize=(13, 8))
    set_3d, front_3d = problem.n_variables >= 3, problem.n_objectives >= 3
    for col, degree in enumerate(DEGREES):
        model = results[degree, "PSGD"]["model"]
        ax_set = fig.add_subplot(2, len(DEGREES), col + 1, projection="3d" if set_3d else None)
        ax_front = fig.add_subplot(2, len(DEGREES), len(DEGREES) + col + 1, projection="3d" if front_3d else None)
        # The Pareto set of the location problem lies in x_4 = 0: plot the first three coordinates.
        shown = BezierSimploid(model.simploid_type, degree, model.control_points.detach()[:, :3])
        plot_bezier_simplex(shown, num=30, ax=ax_set)
        plot_pareto_front(ax_front, problem, model)
        if problem.pareto_set is not None:
            for k, t in enumerate(boundary(problem.n_objectives)):
                label = "true Pareto set" if k == 0 else None
                x = problem.pareto_set(t)
                draw(ax_set, x, color="gray", linestyle=":", label=label)
                draw(ax_front, problem.objective(x), color="gray", linestyle=":", label=label and "true Pareto front")
        ax_set.set_title(f"Pareto set, $D = {degree}$")
        ax_front.set_title(f"Pareto front, $D = {degree}$")
        labeled: tuple[tuple[Any, str], ...] = ((ax_set, "x"), (ax_front, "f"))
        for ax, symbol in labeled:
            ax.set_xlabel(f"${symbol}_1$")
            ax.set_ylabel(f"${symbol}_2$")
            if ax.name == "3d":
                ax.set_zlabel(f"${symbol}_3$")
        if col == 0 and problem.pareto_set is not None:
            ax_set.legend()
            ax_front.legend()
    fig.suptitle(f"{problem.title}: Bézier simplices fitted by PSGD, the median of {N_TRIALS} trials (red: control points)")
    fig.tight_layout()
    return save(fig, f"sgd_benchmark_{problem.name}_pareto.png")


def save(fig: Any, name: str) -> Path:
    """Saves a figure in ``docs/_static`` and closes it."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    return path


def summarize(problem: Problem, results: dict[tuple[int, str], Any]) -> None:
    """Prints the mean losses at the last iteration."""
    quantity = "L(P) - L(P*)" if problem.quadratic else "L(P)"
    for degree in DEGREES:
        cells = [
            f"{method} {key[:3]}. {results[degree, method][key][:, -1].mean():.2e}"
            for method in METHODS
            for key in ("empirical", "generalization")
        ]
        print(f"  D = {degree}, {quantity} at k = {MAX_ITER}: " + ", ".join(cells))
        if problem.pareto_set is not None:
            cells = [f"{method} {results[degree, method]['mse'][:, -1].mean():.2e}" for method in METHODS]
            print(f"  D = {degree}, MSE at k = {MAX_ITER}: " + ", ".join(cells))


def main() -> None:
    for problem in PROBLEMS:
        start = time.perf_counter()
        results = run_problem(problem)
        print(f"{problem.title} ({time.perf_counter() - start:.1f} s)")
        summarize(problem, results)
        paths = [plot_losses(problem, results), plot_pareto(problem, results)]
        if problem.pareto_set is not None:
            paths.append(plot_mse(problem, results))
        for path in paths:
            print(f"  Saved {path}")


if __name__ == "__main__":
    main()
