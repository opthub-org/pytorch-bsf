import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from torch_bsf.sklearn import BezierSimploidRegressor

# 3-generator economic-emission dispatch: generation cost vs. emissions
# P = [P1, P2, P3] in R^3 (power outputs), subject to the demand balance P1 + P2 + P3 = D
# Generator 1 is cheap but dirty, generator 3 is expensive but clean.
demand = 5.0
cost_a = np.array([0.10, 0.12, 0.15])  # quadratic cost coefficients
cost_b = np.array([0.20, 0.30, 0.40])  # linear cost coefficients
emis_a = np.array([0.20, 0.10, 0.05])  # quadratic emission coefficients
emis_b = np.array([0.30, 0.20, 0.10])  # linear emission coefficients


def f1(P):
    # Generation cost (strongly convex)
    return cost_a @ P ** 2 + cost_b @ P


def f2(P):
    # Emissions (strongly convex)
    return emis_a @ P ** 2 + emis_b @ P


def f(P, w):
    return w[0] * f1(P) + w[1] * f2(P)


def solve(t):
    w = np.array([1.0 - t, t])
    result = minimize(
        lambda P: f(P, w),
        x0=np.full(3, demand / 3),
        method="SLSQP",
        bounds=[(0.0, None)] * 3,
        constraints=[{"type": "eq", "fun": lambda P: np.sum(P) - demand}],
        options={"ftol": 1e-12},
    )
    P_opt = result.x
    return (w, P_opt, f1(P_opt), f2(P_opt))


# Sample 10 weights on the 1-simplex from w=(1,0) to w=(0,1)
n_samples = 10
weights = np.linspace(0.0, 1.0, n_samples)
optimals = [solve(t) for t in weights]

# Prepare training data for Bézier simplex fitting
w1 = np.array([p[0][0] for p in optimals])
w2 = np.array([p[0][1] for p in optimals])
X = np.column_stack([w1, w2])
y = np.column_stack([
    np.array([p[2] for p in optimals]),
    np.array([p[3] for p in optimals]),
])

# Fit a degree-4 Bézier simplex to the Pareto front
regressor = BezierSimploidRegressor(degree=4)
regressor.fit(X, y)

# Generate smooth curve for visualization
t_smooth = np.linspace(0.0, 1.0, 200)
X_smooth = np.column_stack([1.0 - t_smooth, t_smooth])
y_smooth = regressor.predict(X_smooth)

# Compute approximation error on training points
pred_train = regressor.predict(X)
max_error = np.max(np.linalg.norm(pred_train - y, axis=1))
print(f"Max training approximation error: {max_error:.4f}")

# Compute approximation error on 101 test weights
t_test = np.linspace(0.0, 1.0, 101)
test_optimals = [solve(t) for t in t_test]
y_test = np.array([[p[2], p[3]] for p in test_optimals])
pred_test = regressor.predict(np.column_stack([1.0 - t_test, t_test]))
print(f"Max test approximation error: {np.max(np.linalg.norm(pred_test - y_test, axis=1)):.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y[:, 0], y[:, 1], color="blue", label="Optimization-derived Pareto points")
ax.plot(y_smooth[:, 0], y_smooth[:, 1], color="red", label="Fitted Bézier simplex")
ax.set_xlabel("Generation Cost (f₁)")
ax.set_ylabel("Emissions (f₂)")
ax.set_title("Smart Grid Pareto Front: Optimization vs Bézier Simplex")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
pathlib.Path("docs/_static").mkdir(parents=True, exist_ok=True)
plt.savefig("docs/_static/smart_grids_pareto.png", dpi=150, bbox_inches="tight")
print("Pareto front plot saved.")

# Fit Bézier simplex to Pareto set (power output space, showing P1 and P2; P3 = D - P1 - P2)
P_targets = np.array([p[1] for p in optimals])
regressor_set = BezierSimploidRegressor(degree=4)
regressor_set.fit(X, P_targets)
P_smooth = regressor_set.predict(X_smooth)

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(P_targets[:, 0], P_targets[:, 1], color="green", label="Optimal power outputs (samples)", s=50)
ax2.plot(P_smooth[:, 0], P_smooth[:, 1], color="orange", label="Bézier simplex approximation", linewidth=2)
ax2.set_xlabel("P₁ (Generator 1 Output)")
ax2.set_ylabel("P₂ (Generator 2 Output)")
ax2.set_title("Pareto Set: Generator Power Outputs with Bézier Approximation")
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("docs/_static/smart_grids_pareto_set.png", dpi=150, bbox_inches="tight")
print("Pareto set plot saved.")
