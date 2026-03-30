"""Generate a figure illustrating the leaf-space structure of the elastic-net
hyperparameter space and the corresponding grid sampling on the 2-simplex.

The figure has three panels:

1. The (α, λ) parameter half-plane with the identified edge λ=0 highlighted.
2. The elastic-net grid on the 2-simplex with vertices (1,0,0) at top,
   (0,1,0) at bottom-left, and (0,0,1) at bottom-right.
3. The quotient space obtained by collapsing the base edge of the simplex to a
   single point P* (the null model), shown as a leaf/eye-shaped CW complex.

All points are coloured by (w1, w2, w3) mapped to (R, G, B), so the same
weight vector has the same colour in every panel.

Usage::

    python examples/generate_leaf_space_plot.py

The script writes ``docs/_static/elastic_net_leaf_space.png``.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from torch_bsf.model_selection.elastic_net_grid import elastic_net_grid


# ---------------------------------------------------------------------------
# Color helper
# ---------------------------------------------------------------------------

def weights_to_rgb(w1, w2, w3):
    """Map weight vectors (w1, w2, w3) to RGB colours."""
    return np.clip(np.stack([w1, w2, w3], axis=-1), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Projection helpers
# New vertex layout: (1,0,0) at top, (0,1,0) at bottom-left,
#                    (0,0,1) at bottom-right.
# ---------------------------------------------------------------------------

def project_to_2d(points):
    """Project 3-D simplex points to 2-D Cartesian coordinates.

    Vertex mapping::

        (1, 0, 0)  ->  (0.5, sqrt3/2)  # data-only vertex (lambda=0), top
        (0, 1, 0)  ->  (0,   0      )  # pure-L1 vertex, bottom-left
        (0, 0, 1)  ->  (1,   0      )  # pure-L2 vertex, bottom-right
    """
    w1, w2, w3 = points[:, 0], points[:, 1], points[:, 2]
    px = 0.5 * w1 + w3
    py = np.sqrt(3) / 2 * w1
    return px, py


def draw_simplex_boundary(ax, **kwargs):
    """Draw the triangle boundary on *ax*."""
    triangle = plt.Polygon(
        [[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]],
        fill=False,
        edgecolor="black",
        linewidth=1.0,
        **kwargs,
    )
    ax.add_patch(triangle)


def to_eye_coords(w1, alpha, r=0.55):
    """Map (w1, alpha) in [0,1]^2 to 2-D leaf/eye coordinates.

    The mapping places vertex A (w1=1, lambda=0) at the top (0, 1) and the
    collapsed base point P* (w1=0, lambda->inf) at the bottom (0, -1).  The
    width of the shape at height corresponding to w1 is proportional to
    sin(pi*(1-w1)), so it tapers to zero at both endpoints producing a
    leaf/eye silhouette.

    Parameters
    ----------
    w1 : float or ndarray
        Data-fidelity weight in [0, 1].
    alpha : float or ndarray
        L1 mixing ratio in [0, 1].
    r : float
        Half-width of the widest cross-section.
    """
    t = 1.0 - w1
    width = r * np.sin(np.pi * t)
    y = np.cos(np.pi * t)
    x = (2.0 * alpha - 1.0) * width
    return x, y


# ---------------------------------------------------------------------------
# Build a coarse grid for illustration (n_lambdas=12, n_alphas=8)
# ---------------------------------------------------------------------------

N_LAMBDAS = 12
N_ALPHAS = 8
BASE = 10

grid = elastic_net_grid(n_lambdas=N_LAMBDAS, n_alphas=N_ALPHAS, base=BASE)
px, py = project_to_2d(grid)

w1_all = grid[:, 0]
w2_all = grid[:, 1]
w3_all = grid[:, 2]
rgb_all = weights_to_rgb(w1_all, w2_all, w3_all)

# Main grid points (excluding vertex copies added for CV compatibility)
n_main = (N_LAMBDAS - 1) * N_ALPHAS

# Unique λ levels and corresponding w1 values
# w1=0 corresponds to λ→∞ (base edge) – filter to finite λ only for the
# left panel and leaf lines.
unique_w1 = np.unique(grid[:n_main, 0])
unique_w1_finite = unique_w1[unique_w1 > 1e-10]   # finite-λ levels only
lambda_unique = (1.0 - unique_w1_finite) / unique_w1_finite  # λ = (1-w1)/w1
lambda_max = lambda_unique.max() * 1.15                       # axis upper limit

alphas_per_row = np.linspace(0.0, 1.0, N_ALPHAS, endpoint=True)


# ---------------------------------------------------------------------------
# Figure layout: three panels side by side
# ---------------------------------------------------------------------------

fig, (ax_rect, ax_simplex, ax_leaf) = plt.subplots(
    1, 3, figsize=(16, 5.2), gridspec_kw={"wspace": 0.45}
)


# ===========================================================================
# Left panel – (α, λ) hyperparameter space
# Points are coloured by (w1, w2, w3) = (R, G, B).
# ===========================================================================
ax = ax_rect

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-lambda_max * 0.04, lambda_max)
ax.set_xlabel(r"$\alpha$  (L1 mixing ratio)", fontsize=11)
ax.set_ylabel(r"$\lambda$  (regularisation strength)", fontsize=11)
ax.set_title("Hyperparameter space\n(before identification)", fontsize=11)

# Background gradient: colour each pixel by (w1, w2, w3)
n_bg = 200
a_bg = np.linspace(0.0, 1.0, n_bg)
l_bg = np.linspace(0.0, lambda_max, n_bg)
A_bg, L_bg = np.meshgrid(a_bg, l_bg)
W1_bg = 1.0 / (1.0 + L_bg)
W2_bg = L_bg * A_bg / (1.0 + L_bg)
W3_bg = L_bg * (1.0 - A_bg) / (1.0 + L_bg)
rgb_bg = np.stack([W1_bg, W2_bg, W3_bg], axis=-1)
ax.imshow(rgb_bg, origin="lower", extent=[0.0, 1.0, 0.0, lambda_max],
          aspect="auto", alpha=0.4)

# Horizontal leaf lines at each λ level
for lam in lambda_unique:
    ax.axhline(y=lam, color="gray", lw=0.5, alpha=0.4, zorder=2)

# Grid scatter coloured by (w1, w2, w3)
for lam, w1 in zip(lambda_unique, unique_w1_finite):
    w1_row = np.full(N_ALPHAS, w1)
    w2_row = lam * alphas_per_row / (1.0 + lam)
    w3_row = lam * (1.0 - alphas_per_row) / (1.0 + lam)
    ax.scatter(
        alphas_per_row,
        np.full(N_ALPHAS, lam),
        c=weights_to_rgb(w1_row, w2_row, w3_row),
        s=20,
        zorder=4,
        edgecolors="none",
    )

# Identified edge: λ=0 → all points collapse to w=(1,0,0) = red
ax.axhline(y=0.0, color=(1.0, 0.0, 0.0), lw=2.5, zorder=3)
ax.scatter([0.0, 1.0], [0.0, 0.0],
           c=[(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)], s=60, zorder=5,
           edgecolors="none")

# Annotation
ax.annotate(
    "identified to one point\n$(w_1, w_2, w_3) = (1, 0, 0)$",
    xy=(0.5, 0.0),
    xytext=(0.5, lambda_max * 0.14),
    fontsize=8.5,
    ha="center",
    color="darkred",
    arrowprops=dict(arrowstyle="-[", color="darkred", lw=1.5,
                    connectionstyle="arc3,rad=0"),
)

# Corner labels
ax.text(0.0, -lambda_max * 0.035, r"$\alpha=0$" "\n(pure L2)",
        fontsize=8, va="top", ha="center")
ax.text(1.0, -lambda_max * 0.035, r"$\alpha=1$" "\n(pure L1)",
        fontsize=8, va="top", ha="center")
ax.tick_params(axis="both", which="both", length=3)


# ===========================================================================
# Centre panel – 2-simplex with foliation (leaves) and grid points
# Vertices: (1,0,0) at top, (0,1,0) at bottom-left, (0,0,1) at bottom-right.
# Points are coloured by (w1, w2, w3) = (R, G, B).
# ===========================================================================
ax = ax_simplex

ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Elastic-net grid on the 2-simplex\n(after identification)", fontsize=11)

# Triangle boundary
draw_simplex_boundary(ax)

# Highlight the base edge (bottom, from (0,0) to (1,0)) with a green→blue gradient
t_base = np.linspace(0.0, 1.0, 100)
pts_base = np.column_stack([t_base, np.zeros(100)])
segs_base = np.stack([pts_base[:-1], pts_base[1:]], axis=1)
t_base_mids = 0.5 * (t_base[:-1] + t_base[1:])
# Along base: (0,1,0)→green at left (t=0), (0,0,1)→blue at right (t=1)
c_base = np.stack([np.zeros(99), 1.0 - t_base_mids, t_base_mids], axis=-1)
ax.add_collection(LineCollection(segs_base, colors=c_base, lw=2.5, zorder=2))

# Leaf lines: iso-w1 segments (horizontal in this projection), green→blue gradient
# Skip w1=0 (that is the base edge itself, already drawn above)
h3 = np.sqrt(3) / 2
for w1 in unique_w1_finite:
    y_lf = h3 * w1
    x_l = 0.5 * w1           # left end: (w1, 1-w1, 0)
    x_r = 1.0 - 0.5 * w1    # right end: (w1, 0, 1-w1)
    t_lf = np.linspace(0.0, 1.0, 30)
    x_lf = np.linspace(x_l, x_r, 30)
    w2_lf = (1.0 - t_lf) * (1.0 - w1)
    w3_lf = t_lf * (1.0 - w1)
    c_lf = weights_to_rgb(np.full(30, w1), w2_lf, w3_lf)
    pts_lf = np.column_stack([x_lf, np.full(30, y_lf)])
    segs_lf = np.stack([pts_lf[:-1], pts_lf[1:]], axis=1)
    c_lf_mids = 0.5 * (c_lf[:-1] + c_lf[1:])
    ax.add_collection(LineCollection(segs_lf, colors=c_lf_mids,
                                     lw=1.0, alpha=0.7, zorder=1))

# Scatter grid points coloured by (w1, w2, w3)
ax.scatter(px, py, c=rgb_all, s=20, zorder=3, edgecolors="none")

# Vertex markers and labels
ax.plot(0.5, h3, "o", color=(1.0, 0.0, 0.0), markersize=8, zorder=5)
ax.plot(0.0, 0.0, "o", color=(0.0, 1.0, 0.0), markersize=8, zorder=5)
ax.plot(1.0, 0.0, "o", color=(0.0, 0.0, 1.0), markersize=8, zorder=5)

offset = 0.05
ax.text(0.5, h3 + offset, r"$(1,0,0)$" "\n" r"$\lambda=0$",
        ha="center", va="bottom", fontsize=8.5, color=(0.8, 0.0, 0.0))
ax.text(0.0 - offset, 0.0, r"$(0,1,0)$" "\n" r"$(\lambda\!\to\!\infty,\ \alpha\!=\!1)$",
        ha="right", va="center", fontsize=8.5, color=(0.0, 0.65, 0.0))
ax.text(1.0 + offset, 0.0, r"$(0,0,1)$" "\n" r"$(\lambda\!\to\!\infty,\ \alpha\!=\!0)$",
        ha="left", va="center", fontsize=8.5, color=(0.0, 0.0, 0.9))
ax.text(0.5, -0.07, "base edge (null model)", fontsize=7.5, ha="center",
        color="dimgray")


# ===========================================================================
# Right panel – quotient (leaf/eye) space
#
# The 2-simplex has its base edge (connecting (0,1,0) and (0,0,1)) collapsed
# to a single point P*.  The resulting CW complex has:
#   - vertex A  = (1,0,0)  at the top          (lambda = 0)
#   - vertex P* = collapsed base edge           (lambda -> inf, null model)
#   - two 1-cells (edges) from A to P*, shown as curves
#   - one 2-cell (face) enclosed by the two edges
# The overall shape resembles a leaf or eye.
# Points are coloured by (w1, w2, w3) = (R, G, B).
# At P*, a large green dot (behind) and a smaller blue dot (in front) are
# overlaid to show that (0,1,0) and (0,0,1) are both mapped here.
# ===========================================================================
ax = ax_leaf

ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Quotient space\n(base edge identified to null model $P^*$)", fontsize=11)

R = 0.55  # half-width of the widest cross-section

# Boundary curves with colour gradient
# t parametrises the boundary from A (t=0) to P* (t=1)
t_bnd = np.linspace(0.0, 1.0, 300)
x_left_bnd  = -R * np.sin(np.pi * t_bnd)   # α=0 side
x_right_bnd =  R * np.sin(np.pi * t_bnd)   # α=1 side
y_bnd = np.cos(np.pi * t_bnd)

# Left boundary (α=0): colour transitions red→blue  (w1=1-t, w2=0, w3=t)
pts_lb = np.column_stack([x_left_bnd, y_bnd])
segs_lb = np.stack([pts_lb[:-1], pts_lb[1:]], axis=1)
t_lb_mids = 0.5 * (t_bnd[:-1] + t_bnd[1:])
c_lb = np.stack([1.0 - t_lb_mids, np.zeros_like(t_lb_mids), t_lb_mids], axis=-1)
ax.add_collection(LineCollection(segs_lb, colors=c_lb, lw=2.0, zorder=3))

# Right boundary (α=1): colour transitions red→green  (w1=1-t, w2=t, w3=0)
pts_rb = np.column_stack([x_right_bnd, y_bnd])
segs_rb = np.stack([pts_rb[:-1], pts_rb[1:]], axis=1)
c_rb = np.stack([1.0 - t_lb_mids, t_lb_mids, np.zeros_like(t_lb_mids)], axis=-1)
ax.add_collection(LineCollection(segs_rb, colors=c_rb, lw=2.0, zorder=3))

# Leaves: constant-λ (constant w1) horizontal segments
# Skip w1=0: it maps to P* (zero-width leaf) and is shown by the vertex marker
for w1 in unique_w1_finite:
    t = 1.0 - w1
    y_lf = np.cos(np.pi * t)
    width = R * np.sin(np.pi * t)
    ax.plot([-width, width], [y_lf, y_lf], color="gray", lw=0.7, alpha=0.45)

# Constant-α curves from A to P*
alpha_lines = np.linspace(0.0, 1.0, N_ALPHAS)
t_line = np.linspace(0.0, 1.0, 200)
for alpha in alpha_lines:
    x_line = (2.0 * alpha - 1.0) * R * np.sin(np.pi * t_line)
    y_line = np.cos(np.pi * t_line)
    ax.plot(x_line, y_line, color="gray", lw=0.5, alpha=0.35)

# Grid scatter coloured by (w1, w2, w3)
grid_main = grid[:n_main]
w1_m = grid_main[:, 0]
w2_m = grid_main[:, 1]
w3_m = grid_main[:, 2]
rgb_m = weights_to_rgb(w1_m, w2_m, w3_m)
w23_sum = w2_m + w3_m
# For the base-edge points (w1=0, w2+w3=1) alpha is defined; for the
# exact vertex (1,0,0) w2=w3=0, so α is undefined – use 0.5 (midpoint).
alpha_m = np.where(w23_sum > 1e-10, w2_m / w23_sum, 0.5)
x_eye, y_eye = to_eye_coords(w1_m, alpha_m, r=R)
ax.scatter(x_eye, y_eye, c=rgb_m, s=20, zorder=3, edgecolors="none")

# Vertex A: w=(1,0,0) = red
ax.plot(0, 1, "o", color=(1.0, 0.0, 0.0), markersize=8, zorder=5)

# Collapsed point P*: large green dot (behind) + smaller blue dot (in front)
# This visualises that both (0,1,0) and (0,0,1) map to P*.
ax.plot(0, -1, "o", color=(0.0, 1.0, 0.0), markersize=14, zorder=4)  # green, larger
ax.plot(0, -1, "o", color=(0.0, 0.0, 1.0), markersize=8,  zorder=5)  # blue, smaller

# Labels
ax.text(0,  1.10, r"$A$: $(1, 0, 0)$, $\lambda=0$",
        ha="center", va="bottom", fontsize=8.5, color=(0.8, 0.0, 0.0))
ax.text(0, -1.12,
        r"$P^*$: null model" "\n" r"$(0,1,0)$ and $(0,0,1)$ identified",
        ha="center", va="top", fontsize=8.0, color="dimgray")

# Edge labels at the widest point
ax.text(-R - 0.04, 0, r"$\alpha=0$" "\n(pure L2)",
        ha="right", va="center", fontsize=8)
ax.text( R + 0.04, 0, r"$\alpha=1$" "\n(pure L1)",
        ha="left",  va="center", fontsize=8)


# ===========================================================================
# Save
# ===========================================================================

fig.savefig(
    "docs/_static/elastic_net_leaf_space.png",
    dpi=150,
    bbox_inches="tight",
)
print("Saved docs/_static/elastic_net_leaf_space.png")
