import pathlib

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from scipy.optimize import minimize
from torch_bsf.simploidal import fit_simploid

# Parametric 2-objective MPC: tracking vs. control effort
# 1D stable system: x(k+1) = a x(k) + u(k), target r = 1, horizon H = 5
# U = [u0, u1, u2, u3, u4] in R^5
# The optimal inputs U*(w, x0) depend on the objective weights w and on the measured
# initial state x0. The parameter space is the square (1-simplex) x (1-simplex),
# so we fit a Bézier simploid of type (1, 1) instead of a Bézier simplex.
a = 0.5
r = 1.0
H = 5
x0_min, x0_max = -1.0, 1.0


def simulate(U, x0):
    """Return state trajectory x(0)..x(H) given x(0)=x0 and inputs U."""
    x = np.zeros(H + 1)
    x[0] = x0
    for k in range(H):
        x[k + 1] = a * x[k] + U[k]
    return x


def f1(U, x0):
    # Tracking error: sum_{k=1}^{H} (x(k) - r)^2
    x = simulate(U, x0)
    return np.sum((x[1:] - r) ** 2)


def f2(U):
    # Control effort: sum_{k=0}^{H-1} u(k)^2
    return np.sum(U ** 2)


def f(U, w, x0):
    return w[0] * f1(U, x0) + w[1] * f2(U)


def solve(t, s):
    """Solve the MPC problem for weight w=(1-t, t) and initial state x0 = x0_min + s (x0_max - x0_min)."""
    w = np.array([1.0 - t, t])
    x0 = x0_min + s * (x0_max - x0_min)
    result = minimize(
        lambda U: f(U, w, x0),
        x0=np.zeros(H),
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-10},
    )
    U_opt = result.x
    return U_opt, np.array([f1(U_opt, x0), f2(U_opt)])


def to_params(t, s):
    """Concatenate barycentric coordinates (1-t, t) of the weight and (1-s, s) of the initial state."""
    t, s = np.broadcast_arrays(np.asarray(t, dtype=float), np.asarray(s, dtype=float))
    return torch.tensor(np.column_stack([1.0 - t, t, 1.0 - s, s]))


# Sample an 11 x 5 grid on the square: 11 weights from w=(1,0) to w=(0,1)
# and 5 initial states from x0=-1 to x0=1
t_grid, s_grid = np.meshgrid(np.linspace(0.0, 1.0, 11), np.linspace(0.0, 1.0, 5), indexing="ij")
t_train, s_train = t_grid.ravel(), s_grid.ravel()
solutions = [solve(t, s) for t, s in zip(t_train, s_train)]
params = to_params(t_train, s_train)
U_targets = np.array([sol[0] for sol in solutions])
y = np.array([sol[1] for sol in solutions])

# Fit Bézier simploids of type (1, 1). The optimal inputs are affine in x0 and the
# objectives are quadratic in x0, so degree 1 and 2 suffice for the x0 factor.
front_model = fit_simploid(params, torch.tensor(y), simploid_type=(1, 1), degree=(6, 2))
set_model = fit_simploid(params, torch.tensor(U_targets), simploid_type=(1, 1), degree=(6, 1))

# Compute approximation errors on training points and on a 41 x 21 test grid
with torch.no_grad():
    max_error = np.max(np.linalg.norm(front_model(params).numpy() - y, axis=1))
print(f"Max training approximation error: {max_error:.4f}")
t_test, s_test = (g.ravel() for g in np.meshgrid(np.linspace(0, 1, 41), np.linspace(0, 1, 21), indexing="ij"))
test_solutions = [solve(t, s) for t, s in zip(t_test, s_test)]
with torch.no_grad():
    pred_test = front_model(to_params(t_test, s_test)).numpy()
    pred_U_test = set_model(to_params(t_test, s_test)).numpy()
y_test = np.array([sol[1] for sol in test_solutions])
U_test = np.array([sol[0] for sol in test_solutions])
print(f"Max test approximation error (Pareto front): {np.max(np.linalg.norm(pred_test - y_test, axis=1)):.4f}")
print(f"Max test approximation error (Pareto set): {np.max(np.linalg.norm(pred_U_test - U_test, axis=1)):.4f}")

# Draw one Pareto front / Pareto set curve per sampled initial state
t_smooth = np.linspace(0.0, 1.0, 200)
s_levels = np.unique(s_train)
colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(s_levels)))

fig, ax = plt.subplots(figsize=(8, 6))
for s, color in zip(s_levels, colors):
    x0 = x0_min + s * (x0_max - x0_min)
    mask = s_train == s
    with torch.no_grad():
        y_smooth = front_model(to_params(t_smooth, s)).numpy()
    ax.scatter(y[mask, 0], y[mask, 1], color=color, s=30)
    ax.plot(y_smooth[:, 0], y_smooth[:, 1], color=color, label=f"x₀ = {x0:+.1f}")
ax.scatter([], [], color="gray", s=30, label="Optimization-derived Pareto points")
ax.plot([], [], color="gray", label="Fitted Bézier simploid")
ax.set_xlabel("Tracking Error (f₁)")
ax.set_ylabel("Control Effort (f₂)")
ax.set_title("Parametric MPC Pareto Fronts: Optimization vs Bézier Simploid")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
pathlib.Path("docs/_static").mkdir(parents=True, exist_ok=True)
plt.savefig("docs/_static/mpc_pareto.png", dpi=150, bbox_inches="tight")
print("Pareto front plot saved.")

fig2, ax2 = plt.subplots(figsize=(8, 6))
for s, color in zip(s_levels, colors):
    x0 = x0_min + s * (x0_max - x0_min)
    mask = s_train == s
    with torch.no_grad():
        U_smooth = set_model(to_params(t_smooth, s)).numpy()
    ax2.scatter(U_targets[mask, 0], U_targets[mask, 1], color=color, s=50)
    ax2.plot(U_smooth[:, 0], U_smooth[:, 1], color=color, linewidth=2, label=f"x₀ = {x0:+.1f}")
ax2.scatter([], [], color="gray", s=50, label="Optimal control inputs (samples)")
ax2.plot([], [], color="gray", linewidth=2, label="Bézier simploid approximation")
ax2.set_xlabel("u₀ (Control Input at Step 0)")
ax2.set_ylabel("u₁ (Control Input at Step 1)")
ax2.set_title("Pareto Sets: MPC Control Inputs with Bézier Simploid Approximation")
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("docs/_static/mpc_pareto_set.png", dpi=150, bbox_inches="tight")
print("Pareto set plot saved.")
