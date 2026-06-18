import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.optimize import minimize
from torch_bsf.sklearn import BezierSimplexRegressor

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
                x = (i - height//2) * 0.1
                y = (j - width//2) * 0.1
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
center = (height//2, width//2)
radius = min(height, width) // 4
for i in range(height):
    for j in range(width):
        if (i - center[0])**2 + (j - center[1])**2 <= radius**2:
            x_true[i, j] = 1.0
x_true = x_true.flatten()

# Noisy measurement
y_obs = K @ x_true + 0.05 * rng.standard_normal(K.shape[0])


def f1(x):
    # Data fidelity + small L2 regularization
    return np.sum((K @ x - y_obs) ** 2) + 0.01 * np.sum(x ** 2)


def f2(x):
    # Smoothness (total variation via finite differences) + small L2
    return np.sum((L @ x) ** 2) + 0.01 * np.sum(x ** 2)


def f(x, w):
    # Weighted combination of fidelity and smoothness
    return w[0] * f1(x) + w[1] * f2(x)


# Sample weights on the 1-simplex from w=(1,0) to w=(0,1)
n_samples = 200
weights = np.linspace(0.0, 1.0, n_samples)
optimals = []
for idx, t in enumerate(weights):
    print(f"Computing Pareto front... {idx}/{n_samples} weights optimized", end="\r", flush=True)
    w = np.array([1.0 - t, t])
    result = minimize(
        lambda x: f(x, w),
        x0=np.zeros(n_pixels),
        method="L-BFGS-B",
    )
    x_opt = result.x
    optimals.append((w, x_opt, f1(x_opt), f2(x_opt)))
print(f"Computing Pareto front... {n_samples}/{n_samples} weights optimized", end="\r", flush=True)

pareto_colors = [(w[0], w[1], 0) for w, _, _, _ in optimals]

# Prepare training data for Bézier simplex fitting
w1 = np.array([p[0][0] for p in optimals])
w2 = np.array([p[0][1] for p in optimals])
X = np.column_stack([w1, w2])
y = np.column_stack([
    np.array([p[2] for p in optimals]),
    np.array([p[3] for p in optimals]),
])

# Initialize control points along the line between the two extreme Pareto points
degree = 100
init = {
    (i, degree - i): ((i / degree) * y[0] + (1 - i / degree) * y[-1]).tolist()
    for i in range(degree + 1)
}

# Fit Bézier simplex to Pareto front (objective space)
pf_bezier = BezierSimplexRegressor(
    init=init,  # initialize control points along the line between the two extreme Pareto points
    freeze=[[0, degree], [degree, 0]],  # freeze endpoints to match true extremes
    smoothness_weight=1e-4  # small smoothness to stabilize fitting
)
pf_bezier.fit(X, y)

# Generate smooth curve for visualization
t_smooth = np.linspace(0.0, 1.0, 200)
line_colors = [(1 - t, t, 0) for t in t_smooth[:-1]]
X_smooth = np.column_stack([1.0 - t_smooth, t_smooth])
y_smooth = pf_bezier.predict(X_smooth)

# Compute approximation error on training points
pred_train = pf_bezier.predict(X)
max_error = np.max(np.linalg.norm(pred_train - y, axis=1))
print(f"Max training approximation error: {max_error:.4f}")

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

# Fit Bézier simplex to Pareto set (reconstruction space, 2D images)
z_targets = np.array([p[1] for p in optimals])
degree = 128
init = {
    (i, degree - i): ((i / degree) * z_targets[0] + (1 - i / degree) * z_targets[-1]).tolist()
    for i in range(degree + 1)
}

ps_bezier = BezierSimplexRegressor(
    init=init,
    freeze=[[0, degree], [degree, 0]],
    smoothness_weight=1e-6
)
ps_bezier.fit(X, z_targets)
z_smooth = ps_bezier.predict(X_smooth)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(z_targets[:, 0], z_targets[:, 1], color=pareto_colors,
            label="Optimal reconstructions (samples)", s=50)
segments = np.array([z_smooth[:-1, 0:2], z_smooth[1:, 0:2]]).transpose(1, 0, 2)
lc = LineCollection(segments, colors=line_colors, linewidth=2, label="Fitted Bézier simplex")
ax.add_collection(lc)
ax.axhline(x_true[1], color="gray", linestyle="--", alpha=0.5,
            label=f"True pixel 2 = {x_true[1]:.1f}")
ax.axvline(x_true[0], color="gray", linestyle=":", alpha=0.5,
            label=f"True pixel 1 = {x_true[0]:.1f}")
ax.set_xlabel("Pixel 1 (x₁)")
ax.set_ylabel("Pixel 2 (x₂)")
ax.set_title("CT Reconstruction Parameters with Bézier Approximation")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
pathlib.Path("docs/_static").mkdir(parents=True, exist_ok=True)
plt.savefig("docs/_static/medical_imaging_pareto_set.png", dpi=150, bbox_inches="tight")
print("Pareto set plot saved.")

# Plot selected 2D reconstructions along the Pareto set
fig2, axes = plt.subplots(1, 5, figsize=(15, 3))
selected_indices = [0, 49, 99, 149, 199]  # Select 5 points along the curve
for idx, ax_idx in enumerate(selected_indices):
    img = z_smooth[ax_idx].reshape(height, width)
    axes[idx].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[idx].set_title(f"w=({1 - t_smooth[ax_idx]:.2f}, {t_smooth[ax_idx]:.2f})")
    axes[idx].axis('off')
plt.tight_layout()
plt.savefig("docs/_static/medical_imaging_pareto_samples.png", dpi=150, bbox_inches="tight")
print("Pareto set plot saved.")
