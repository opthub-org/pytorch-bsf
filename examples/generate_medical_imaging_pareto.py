import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.collections import LineCollection
from scipy.optimize import minimize

from torch_bsf import BezierSpace, Samples, fit

# CT reconstruction: fidelity vs. regularization (L-curve experiment)
# 2D image: 16x16 pixels for realistic 2D reconstruction
height, width = 16, 16
n_pixels = height * width
rng = np.random.default_rng(42)

# Number of projection angles and rays per angle
n_angles = 8
n_rays = 16

# Generate projection matrix K (shape n_angles*n_rays x n_pixels)
# Simplified 2D Radon transform simulation
K = np.zeros((n_angles * n_rays, n_pixels))
angle_step = np.pi / n_angles
for a in range(n_angles):
    theta = a * angle_step
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    for r in range(n_rays):
        # Ray position
        t = (r - n_rays // 2) * 0.1
        for i in range(height):
            for j in range(width):
                x = (i - height // 2) * 0.1
                y = (j - width // 2) * 0.1
                dist = abs(x * cos_theta + y * sin_theta - t)
                if dist < 0.05:  # Simple line integral approximation
                    K[a * n_rays + r, i * width + j] = 1.0

# finite-difference matrix L for 2D total variation (anisotropic)
L = np.zeros((2 * n_pixels, n_pixels))
idx = 0
for i in range(height):
    for j in range(width):
        p = i * width + j
        if j < width - 1:  # horizontal difference
            L[idx, p] = -1
            L[idx, p + 1] = 1
            idx += 1
        if i < height - 1:  # vertical difference
            L[idx, p] = -1
            L[idx, p + width] = 1
            idx += 1
L = L[:idx]  # Trim to actual size

# True image: simple 2D phantom (circle)
x_true = np.zeros((height, width))
center = (height // 2, width // 2)
radius = min(height, width) // 4
for i in range(height):
    for j in range(width):
        if (i - center[0]) ** 2 + (j - center[1]) ** 2 <= radius**2:
            x_true[i, j] = 1.0
x_true = x_true.flatten()

# Noisy measurement
y_obs = K @ x_true + 0.05 * rng.standard_normal(K.shape[0])


def f1(x):
    # Data fidelity + small L2 regularization
    return np.sum((K @ x - y_obs) ** 2) + 0.01 * np.sum(x**2)


def f2(x):
    # Smoothness (total variation via finite differences) + small L2
    return np.sum((L @ x) ** 2) + 0.01 * np.sum(x**2)


def f(x, w):
    # Weighted combination of fidelity and smoothness
    return w[0] * f1(x) + w[1] * f2(x)


def grad_f(x, w):
    # Analytic gradient of f, which lets L-BFGS-B converge to high accuracy
    grad_f1 = 2 * K.T @ (K @ x - y_obs) + 0.02 * x
    grad_f2 = 2 * L.T @ (L @ x) + 0.02 * x
    return w[0] * grad_f1 + w[1] * grad_f2


# The trade-off between fidelity and smoothness is governed by the regularization
# parameter lam = w2 / w1. As in the classical L-curve analysis, we sweep lam on a
# logarithmic scale, lam(t) = 10^(-2 + 3t) for t in [0, 1], and use (1 - t, t) as the
# parameter on the 1-simplex. Sampling the weights uniformly instead would crowd all the
# interesting reconstructions into a tiny neighborhood of w = (1, 0), where the Pareto set
# changes too abruptly for a low-degree Bézier simplex.
def weights_of(t):
    lam = 10.0 ** (-2.0 + 3.0 * t)
    return np.array([1.0, lam]) / (1.0 + lam)


def solve(t):
    w = weights_of(t)
    result = minimize(
        lambda x: f(x, w),
        x0=np.zeros(n_pixels),
        jac=lambda x: grad_f(x, w),
        method="L-BFGS-B",
        options={"maxiter": 10000, "ftol": 1e-14, "gtol": 1e-10},
    )
    x_opt = result.x
    return (t, x_opt, f1(x_opt), f2(x_opt))


def fit_bezier_simplex(X, Y, degree):
    """Fit a Bézier simplex by exact linear least squares and return its predictor.

    The control points enter the model linearly, so ``fit`` solves for them exactly, which
    is more accurate than gradient-based training for the 256-dimensional Pareto set.
    """
    model = fit(BezierSpace.simplex(1, degree=degree), Samples(torch.tensor(X), torch.tensor(Y))).model

    def predict(X_new):
        with torch.no_grad():
            return model(torch.tensor(X_new)).numpy()

    return predict


n_samples = 41
ts = np.linspace(0.0, 1.0, n_samples)
optimals = []
for idx, t in enumerate(ts):
    print(f"Computing Pareto front... {idx}/{n_samples} weights optimized", end="\r", flush=True)
    optimals.append(solve(t))
print(f"Computing Pareto front... {n_samples}/{n_samples} weights optimized")

pareto_colors = [(1 - t, t, 0) for t, _, _, _ in optimals]

# Prepare training data for Bézier simplex fitting
X = np.column_stack([1.0 - ts, ts])
y = np.column_stack(
    [
        np.array([p[2] for p in optimals]),
        np.array([p[3] for p in optimals]),
    ]
)

# Fit a degree-10 Bézier simplex to the Pareto front (objective space)
pf_bezier = fit_bezier_simplex(X, y, degree=10)

# Generate smooth curve for visualization
t_smooth = np.linspace(0.0, 1.0, 200)
line_colors = [(1 - t, t, 0) for t in t_smooth[:-1]]
X_smooth = np.column_stack([1.0 - t_smooth, t_smooth])
y_smooth = pf_bezier(X_smooth)

# Compute approximation error on training points
pred_train = pf_bezier(X)
max_error = np.max(np.linalg.norm(pred_train - y, axis=1))
print(f"Max training approximation error: {max_error:.4f}")

# Compute approximation error on 20 test parameters lying between the training ones
t_test = (ts[:-1] + ts[1:])[::2] / 2
test_optimals = [solve(t) for t in t_test]
X_test = np.column_stack([1.0 - t_test, t_test])
y_test = np.array([[p[2], p[3]] for p in test_optimals])
print(f"Max test approximation error: {np.max(np.linalg.norm(pf_bezier(X_test) - y_test, axis=1)):.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y[:, 0], y[:, 1], color=pareto_colors, label="Optimization-derived Pareto points")
segments = np.array([y_smooth[:-1], y_smooth[1:]]).transpose(1, 0, 2)
lc = LineCollection(segments, colors=line_colors, linewidth=2, label="Fitted Bézier simplex")
ax.add_collection(lc)
ax.set_xlabel("Data Fidelity (f₁)")
ax.set_ylabel("Smoothness Regularization (f₂)")
ax.set_title("CT Reconstruction L-Curve: Optimization vs Bézier Simplex")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
pathlib.Path("docs/_static").mkdir(parents=True, exist_ok=True)
plt.savefig("docs/_static/medical_imaging_pareto.png", dpi=150, bbox_inches="tight")
print("Pareto front plot saved.")

# Fit a degree-10 Bézier simplex to Pareto set (reconstruction space, 2D images)
z_targets = np.array([p[1] for p in optimals])
ps_bezier = fit_bezier_simplex(X, z_targets, degree=10)
z_test = np.array([p[1] for p in test_optimals])
print(f"Max test approximation error (Pareto set): {np.max(np.abs(ps_bezier(X_test) - z_test)):.4f}")
z_smooth = ps_bezier(X_smooth)

fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.scatter(z_targets[:, 0], z_targets[:, 1], color=pareto_colors, label="Optimal reconstructions (samples)", s=50)
segments = np.array([z_smooth[:-1, 0:2], z_smooth[1:, 0:2]]).transpose(1, 0, 2)
lc = LineCollection(segments, colors=line_colors, linewidth=2, label="Fitted Bézier simplex")
ax2.add_collection(lc)
ax2.axhline(x_true[1], color="gray", linestyle="--", alpha=0.5, label=f"True pixel 2 = {x_true[1]:.1f}")
ax2.axvline(x_true[0], color="gray", linestyle=":", alpha=0.5, label=f"True pixel 1 = {x_true[0]:.1f}")
ax2.set_xlabel("Pixel 1 (x₁)")
ax2.set_ylabel("Pixel 2 (x₂)")
ax2.set_title("CT Reconstruction Parameters with Bézier Approximation")
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
pathlib.Path("docs/_static").mkdir(parents=True, exist_ok=True)
plt.savefig("docs/_static/medical_imaging_pareto_set.png", dpi=150, bbox_inches="tight")
print("Pareto set plot saved.")

# Plot selected 2D reconstructions along the Pareto set
fig3, axes = plt.subplots(1, 5, figsize=(15, 3))
selected_indices = [0, 49, 99, 149, 199]  # Select 5 points along the curve
for idx, ax_idx in enumerate(selected_indices):
    img = z_smooth[ax_idx].reshape(height, width)
    axes[idx].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[idx].set_title(f"λ={10.0 ** (-2.0 + 3.0 * t_smooth[ax_idx]):.3g}")
    axes[idx].axis("off")
plt.tight_layout()
plt.savefig("docs/_static/medical_imaging_pareto_samples.png", dpi=150, bbox_inches="tight")
print("Pareto set samples plot saved.")
