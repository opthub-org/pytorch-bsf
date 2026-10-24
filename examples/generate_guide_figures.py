"""Reproduce the unified API tutorials' synthetic comparisons.

Run from a checkout with the plot extra installed:
    MPLCONFIGDIR=/tmp/bsf-mpl python examples/generate_guide_figures.py

Each experiment uses fixed inputs and local seeds. JSON records the measured
errors and settings; none of these experiments is a hardware benchmark.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from torch_bsf import (  # noqa: E402
    BezierModel,
    BezierSpace,
    Ensemble,
    FixedControls,
    Points,
    PreconditionedSGD,
    Samples,
    SimploidalSet,
    fit,
    minimize,
)
from torch_bsf.active_learning import farthest_points, query_by_committee  # noqa: E402
from torch_bsf.model_selection import KFold, cross_validate, select_degree  # noqa: E402
from torch_bsf.preprocessing import StdScaler  # noqa: E402
from torch_bsf.sampling import grid, sample  # noqa: E402
from torch_bsf.splitting import EdgeSplit, subdivide  # noqa: E402

DESTINATION = Path(__file__).resolve().parents[1] / "docs/_static/guides"


def coordinates(x):
    return torch.stack((1 - x, x), dim=1)


def target(x):
    return torch.sin(torch.pi * x) + 0.2 * x


def save(fig, name):
    fig.savefig(DESTINATION / f"{name}.png", dpi=140)
    plt.close(fig)


def training_data():
    x = torch.linspace(0, 1, 18, dtype=torch.float64)
    noise = torch.randn(18, generator=torch.Generator().manual_seed(17), dtype=torch.float64) * 0.1
    return x, Samples(coordinates(x), (target(x) + noise)[:, None])


def degree_comparison(records):
    x, data = training_data()
    q = torch.linspace(0, 1, 401, dtype=torch.float64)
    search = select_degree(BezierSpace.simplex(1, degree=1), data, degrees=[1, 2, 3, 5, 9], cv=KFold(3, seed=7))
    assert search.success
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for degree in [1, search.best_degree, 9]:
        result = fit(BezierSpace.simplex(1, degree=degree), data).require_success()
        prediction = result.model(coordinates(q)).detach()[:, 0]
        axes[0].plot(q, prediction, label=f"degree {degree}")
    axes[0].plot(q, target(q), "k:", label="noise-free target")
    axes[0].scatter(x, data.values[:, 0], c=".5", s=14, label="observations")
    axes[0].set(xlabel="x (second barycentric coordinate)", ylabel="output", ylim=(-0.3, 1.5))
    axes[0].legend(fontsize=8)
    errors = [c.validation.metrics["pooled_mse"] for c in search.candidates]
    axes[1].semilogy([1, 2, 3, 5, 9], errors, "o-")
    axes[1].set(xlabel="degree", ylabel="held-out pooled MSE", title=f"After selection: degree {search.best_degree}")
    records["degree"] = {"seed": 17, "fold_seed": 7, "folds": 3, "degrees": [1, 2, 3, 5, 9], "pooled_mse": errors}
    save(fig, "degree")


def sampling_comparison(records):
    triangle = BezierSpace.simplex(2, degree=2)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    for ax, (name, points) in zip(
        axes,
        [
            ("random, n=64", sample(triangle, 64, seed=7)),
            ("Sobol, n=64", sample(triangle, 64, seed=7, method="sobol")),
            ("grid, resolution=10 (66 points)", grid(triangle, 10)),
        ],
        strict=True,
    ):
        t = points.parameters
        ax.scatter(t[:, 1] + t[:, 2] / 2, t[:, 2] * 3**0.5 / 2, s=12)
        ax.plot([0, 1, 0.5, 0], [0, 0, 3**0.5 / 2, 0], c=".5")
        ax.set(title=name, aspect="equal", xlabel="triangle horizontal coordinate", ylabel="triangle vertical coordinate")
    records["sampling"] = {"seed": 7, "random_n": 64, "sobol_n": 64, "grid_resolution": 10, "grid_n": 66}
    save(fig, "sampling")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    for ax, resolution in zip(axes, [3, 12], strict=True):
        points = grid(BezierSpace.simplex(1, degree=3), resolution)
        x = points.parameters[:, 1]
        ax.plot(x, target(x), "o-", label=f"resolution={resolution}")
        q = torch.linspace(0, 1, 401)
        ax.plot(q, target(q), "k:", label="same continuous function")
        ax.set(xlabel="x", ylabel="output", title=f"{len(points)} evaluation points")
        ax.legend(fontsize=8)
    save(fig, "grid")


def fixed_and_scaling(records):
    x, data = training_data()
    space = BezierSpace.simplex(1, degree=3)
    before = fit(space, data).require_success()
    fixed = FixedControls.from_cells(space, [(0, (3, 0), [0.0]), (0, (0, 3), [0.2])])
    after = fit(space, data, fixed=fixed).require_success()
    q = torch.linspace(0, 1, 401, dtype=torch.float64)
    fig, ax = plt.subplots(figsize=(7, 3.5), layout="constrained")
    ax.scatter(x, data.values[:, 0], c=".5", s=14, label="same observations")
    for label, result in [("before: all controls free", before), ("after: exact endpoint controls", after)]:
        ax.plot(q, result.model(coordinates(q)).detach()[:, 0], label=label)
    ax.scatter([0, 1], [0, 0.2], marker="s", c="k", label="known endpoint values")
    ax.set(xlabel="x", ylabel="output")
    ax.legend(fontsize=8)
    records["fixed"] = {"before_data_mse": before.metrics["data_mse"], "after_data_mse": after.metrics["data_mse"]}
    save(fig, "fixed")
    y = torch.cat((data.values, 1000 + 100 * data.values), dim=1)
    scaler = StdScaler()
    scaled = scaler.fit_transform(y)
    raw = fit(space, Samples(data.points, y)).require_success()
    normalized = fit(space, Samples(data.points, scaled)).require_success()
    restored = scaler.inverse_transform(normalized.model(coordinates(q))).detach()
    direct = raw.model(coordinates(q)).detach()
    assert torch.allclose(restored, direct, atol=1e-10, rtol=1e-10)
    records["scaling"] = {"max_prediction_difference": (restored - direct).abs().max().item()}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for column in range(2):
        axes[0].plot(x, y[:, column], "o-", label=f"output {column + 1}")
        axes[1].plot(x, scaled[:, column], "o-" if column == 0 else "x--", label=f"output {column + 1}")
    for ax in axes:
        ax.legend(fontsize=8)
    axes[0].set(title="Before: different units", xlabel="x", ylabel="original outputs")
    axes[1].set(title="After: explicit standardization", xlabel="x", ylabel="standardized outputs")
    save(fig, "scaling")


def subdivision(records):
    space = BezierSpace.simplex(1, degree=3)
    x, data = training_data()
    model = fit(space, data).require_success().model
    parts = subdivide(model, policy=EdgeSplit(0, 1, fraction=0.4))
    points = grid(space, 100)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True, layout="constrained")
    axes[0].plot(points.parameters[:, 1], model(points).detach()[:, 0], label="parent")
    errors = []
    for k, (child, mapping) in enumerate(zip(parts.models, parts.maps, strict=True)):
        local = grid(child.space, 100)
        parent = mapping.to_parent(local)
        values = child(local).detach()
        error = (values - model(parent)).abs().max().item()
        assert error < 1e-12
        errors.append(error)
        axes[1].plot(parent.parameters[:, 1], values[:, 0], label=f"child {k}")
    for ax, title in zip(axes, ["Before: one parent", "After: two exact pieces"], strict=True):
        ax.set(title=title, xlabel="parent x", ylabel="output")
        ax.legend()
    records["subdivision"] = {"fraction": 0.4, "maximum_errors": errors}
    save(fig, "subdivision")


def active_learning(records):
    space = BezierSpace.simplex(1, degree=3)
    x = torch.tensor([0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0], dtype=torch.float64)
    data = Samples(coordinates(x), target(x)[:, None])
    validation = cross_validate(space, data, cv=KFold(4, seed=7))
    assert validation.success
    committee = Ensemble(validation.models)
    q = torch.linspace(0.001, 0.999, 200, dtype=torch.float64)
    candidates = Points(coordinates(q))
    selected = query_by_committee(committee, candidates, n=3)
    spread = farthest_points(candidates, data.points, n=3)
    expanded = Samples(
        torch.cat([data.points.parameters, selected.parameters]),
        torch.cat([data.values, target(selected.parameters[:, 1])[:, None]]),
    )
    before = fit(space, data).require_success().model
    after = fit(space, expanded).require_success().model
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained")
    members = committee.members(candidates).detach()[:, :, 0]
    for member in members:
        axes[0].plot(q, member, alpha=0.5)
    axes[0].scatter(x, data.values[:, 0], c="k", label="observed")
    axes[0].scatter(selected.parameters[:, 1], target(selected.parameters[:, 1]), marker="x", s=70, label="committee queries")
    axes[0].scatter(spread.parameters[:, 1], target(spread.parameters[:, 1]), marker="+", s=70, label="coverage queries")
    for label, model in [("before acquisition", before), ("after querying exact target", after)]:
        error = (model(candidates).detach()[:, 0] - target(q)).abs()
        axes[1].plot(q, error, label=label)
        records[label] = {"test_mse": error.square().mean().item()}
    axes[0].set(title="Fold models and query locations", xlabel="x", ylabel="output")
    axes[1].set(title="Independent dense-grid error", xlabel="x", ylabel="absolute error")
    for ax in axes:
        ax.legend(fontsize=8)
    records["acquisition"] = {
        "fold_seed": 7,
        "candidates": 200,
        "committee_x": selected.parameters[:, 1].tolist(),
        "coverage_x": spread.parameters[:, 1].tolist(),
    }
    save(fig, "acquisition")


def initialization(records):
    space = BezierSpace.simplex(1, degree=2)
    points = grid(space, 80)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained")
    for constant in [0.0, 2.0]:
        initial = BezierModel(space, torch.full((space.n_controls, 1), constant, dtype=torch.float64))
        result = minimize(
            initial,
            lambda y, p: (y[:, 0] - p.parameters[:, 1].square()).square(),
            points=points,
            seed=7,
            method=PreconditionedSGD(max_iter=400, step_size=0.1, gradient_tolerance=1e-9),
        )
        result.require_success()
        axes[0].plot(points.parameters[:, 1], initial(points).detach()[:, 0], "--", label=f"before: constant {constant:g}")
        axes[0].plot(points.parameters[:, 1], result.model(points).detach()[:, 0], label=f"after: start {constant:g}")
        axes[1].semilogy([h.iteration for h in result.history], [h.loss for h in result.history], label=f"start {constant:g}")
        records[f"initial_{constant:g}"] = {
            "status": result.status,
            "iterations": result.n_iterations,
            "objective": result.objective,
        }
    axes[0].set(xlabel="x", ylabel="output", title="Same convex objective, different starts")
    axes[1].set(xlabel="iteration (pre-update)", ylabel="full-batch loss", title="Convergence histories")
    for ax in axes:
        ax.legend(fontsize=8)
    save(fig, "initialization")


def topology(records):
    domain = SimploidalSet()
    left, right = domain.add_simplex(1), domain.add_simplex(1)
    domain.identify(left.face(0, 0), right.face(0, 1))
    space = BezierSpace.on_set(domain, degree=3)
    u = torch.linspace(0, 1, 12, dtype=torch.float64)
    t = coordinates(u)
    xs = [0.5 * u, 0.5 + 0.5 * u]
    ys = [target(xs[0]) + 0.08, target(xs[1]) - 0.08]
    data = Samples.from_cells(space, [(left, t, ys[0][:, None]), (right, t, ys[1][:, None])])
    joined = fit(space, data).require_success().model
    standalone = BezierSpace.simplex(1, degree=3)
    independent = [fit(standalone, Samples(t, y[:, None])).require_success().model for y in ys]
    seam = Points.from_cells(space, [(left, [[0.0, 1.0]]), (right, [[1.0, 0.0]])])
    seam_values = joined(seam).detach()
    assert torch.equal(seam_values[0], seam_values[1])
    before_gap = (independent[0](torch.tensor([[0.0, 1.0]])) - independent[1](torch.tensor([[1.0, 0.0]]))).abs().item()
    records["topology"] = {"independent_seam_gap": before_gap, "joined_seam_gap": 0.0}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), sharex=True, sharey=True, layout="constrained")
    q = torch.linspace(0, 1, 201, dtype=torch.float64)
    for k, handle in enumerate([left, right]):
        physical_x = 0.5 * q + 0.5 * k
        local = coordinates(q)
        for ax in axes:
            ax.scatter(xs[k], ys[k], s=12, color=f"C{k}", alpha=0.6)
        axes[0].plot(physical_x, independent[k](local).detach()[:, 0], label=f"patch {k}")
        tagged = Points.from_cells(space, [(handle, local)])
        axes[1].plot(physical_x, joined(tagged).detach()[:, 0], label=f"patch {k}")
    for ax, title in zip(axes, [f"Before: independent (gap={before_gap:.3f})", "After: shared endpoint (gap=0)"], strict=True):
        ax.axvline(0.5, c=".5", ls=":")
        ax.set(title=title, xlabel="physical x", ylabel="output")
        ax.legend(fontsize=8)
    save(fig, "topology")


def main():
    torch.set_num_threads(1)
    DESTINATION.mkdir(parents=True, exist_ok=True)
    records = {"torch_version": torch.__version__, "dtype": "float64", "device": "cpu", "optimizer_seed": 7}
    for experiment in [
        degree_comparison,
        sampling_comparison,
        fixed_and_scaling,
        subdivision,
        active_learning,
        initialization,
        topology,
    ]:
        experiment(records)
    (DESTINATION / "metrics.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
