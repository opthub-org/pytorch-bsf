import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.optimize import minimize
from torch_bsf.sklearn import BezierSimplexRegressor
from torch_bsf.preprocessing import NoneScaler


def build_problem(n=8, seed=42):
    rng = np.random.default_rng(seed)
    K = rng.standard_normal((6, n))
    L = np.zeros((n - 1, n))
    for i in range(n - 1):
        L[i, i] = -1.0
        L[i, i + 1] = 1.0

    x_true = np.array([0.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.0])
    y_obs = K @ x_true + 0.05 * rng.standard_normal(6)
    return K, L, x_true, y_obs


def fidelity(x, K, y_obs):
    return np.sum((K @ x - y_obs) ** 2) + 0.01 * np.sum(x ** 2)


def smoothness(x, L):
    return np.sum((L @ x) ** 2) + 0.01 * np.sum(x ** 2)


def weighted_objective(x, w, K, L, y_obs):
    return w[0] * fidelity(x, K, y_obs) + w[1] * smoothness(x, L)


def compute_pareto_points(K, L, y_obs, n_samples=1000):
    n = K.shape[1]
    weights = np.linspace(0.0, 1.0, n_samples)
    optimals = []

    for t in weights:
        w = np.array([1.0 - t, t])
        result = minimize(
            lambda x: weighted_objective(x, w, K, L, y_obs),
            x0=np.zeros(n),
            method="L-BFGS-B",
        )
        x_opt = result.x
        optimals.append((w, x_opt, fidelity(x_opt, K, y_obs), smoothness(x_opt, L)))

    return np.array(weights), optimals


def prepare_training_data(optimals):
    w1 = np.array([p[0][0] for p in optimals])
    w2 = np.array([p[0][1] for p in optimals])
    X = np.column_stack([w1, w2])
    y = np.column_stack([
        np.array([p[2] for p in optimals]),
        np.array([p[3] for p in optimals]),
    ])
    return X, y


def init_bezier_endpoints(y, degree):
    init = {(i, degree - i): ((i / degree) * y[0] + (1 - i / degree) * y[-1]).tolist()
            for i in range(degree + 1)}
    init[(degree, 0)] = y[0].tolist()
    init[(0, degree)] = y[-1].tolist()
    return init


def fit_bezier_simplex(X, y, degree, freeze, smoothness_weight, max_epochs):
    init = init_bezier_endpoints(y, degree)
    regressor = BezierSimplexRegressor(
        degree=None,
        init=init,
        freeze=freeze,
        smoothness_weight=smoothness_weight,
        max_epochs=max_epochs,
    )
    scaler = NoneScaler()
    y_scaled = scaler.fit_transform(y)
    regressor.fit(X, y_scaled)
    return regressor, scaler


def weighted_colors(weights):
    return [(1.0 - t, t, 0.0) for t in weights]


def plot_colored_line(ax, points, weights, label):
    colors = weighted_colors(weights)
    segments = np.array([points[:-1], points[1:]]).transpose(1, 0, 2)
    lc = LineCollection(segments, colors=colors, linewidth=2, label=label)
    ax.add_collection(lc)


def plot_pareto_front(y, y_smooth, optimals, t_smooth, output_path):
    point_colors = [(p[0][0], p[0][1], 0.0) for p in optimals]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y[:, 0], y[:, 1], color=point_colors, label="Optimization-derived Pareto points")
    plot_colored_line(ax, y_smooth, t_smooth, "Fitted Bézier simplex")
    ax.set_xlabel("Data Fidelity (f₁)")
    ax.set_ylabel("Smoothness Regularization (f₂)")
    ax.set_title("CT Reconstruction L-Curve: Optimization vs Bézier Simplex")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")


def plot_pareto_set(z_targets, z_smooth, optimals, x_true, t_smooth, output_path):
    point_colors = [(p[0][0], p[0][1], 0.0) for p in optimals]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(z_targets[:, 0], z_targets[:, 1], color=point_colors,
               label="Optimal reconstructions (samples)", s=50)

    plot_colored_line(ax, z_smooth[:, :2], t_smooth, "Bézier simplex approximation")
    ax.axhline(x_true[1], color="gray", linestyle="--", alpha=0.5,
               label=f"True pixel 2 = {x_true[1]:.1f}")
    ax.axvline(x_true[0], color="gray", linestyle=":", alpha=0.5,
               label=f"True pixel 1 = {x_true[0]:.1f}")
    ax.set_xlabel("Pixel 1 (x₁)")
    ax.set_ylabel("Pixel 2 (x₂)")
    ax.set_title("Pareto Set: CT Reconstruction Parameters with Bézier Approximation")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")


def main():
    K, L, x_true, y_obs = build_problem()
    _, optimals = compute_pareto_points(K, L, y_obs, n_samples=1000)
    X, y = prepare_training_data(optimals)

    regressor, scaler = fit_bezier_simplex(
        X,
        y,
        degree=100,
        freeze=[[0, 100], [100, 0]],
        smoothness_weight=1e-4,
        max_epochs=3,
    )

    t_smooth = np.linspace(0.0, 1.0, 200)
    X_smooth = np.column_stack([1.0 - t_smooth, t_smooth])
    y_smooth = regressor.predict(X_smooth)
    y_smooth = scaler.inverse_transform(y_smooth)

    pred_train = regressor.predict(X)
    max_error = np.max(np.linalg.norm(pred_train - y, axis=1))
    print(f"Max training approximation error: {max_error:.4f}")

    pathlib.Path("docs/_static").mkdir(parents=True, exist_ok=True)
    plot_pareto_front(y, y_smooth, optimals, t_smooth,
                      "docs/_static/medical_imaging_pareto.png")
    print("Pareto front plot saved.")

    z_targets = np.array([p[1] for p in optimals])
    regressor_set, _ = fit_bezier_simplex(
        X,
        z_targets,
        degree=128,
        freeze=[[0, 128], [128, 0]],
        smoothness_weight=1e-6,
        max_epochs=3,
    )
    z_smooth = regressor_set.predict(X_smooth)

    plot_pareto_set(z_targets, z_smooth, optimals, x_true, t_smooth,
                    "docs/_static/medical_imaging_pareto_set.png")
    print("Pareto set plot saved.")


if __name__ == "__main__":
    main()
