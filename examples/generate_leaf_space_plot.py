"""Generate a figure illustrating the leaf-space structure of the elastic-net
hyperparameter space and the corresponding grid sampling on the 2-simplex.

The figure has three panels:

1. The (λ, α) parameter domain [0, ∞) × [0, 1] with the boundary λ=0 highlighted.
   x-axis: λ (regularization strength), y-axis: α (L1 mixing ratio).
2. The elastic-net grid on the 2-simplex with vertices (1,0,0) at bottom-left,
   (0,1,0) at the top, and (0,0,1) at bottom-right.
3. The quotient space obtained by collapsing the base edge of the simplex to a
   single point P* (the null model), shown as a leaf/eye-shaped CW complex,
   rotated 90° counterclockwise (A at left, P* at right).

All points are colored by (w1, w2, w3) mapped to (R, G, B), so the same
weight vector has the same color in every panel.

Usage::

    python examples/generate_leaf_space_plot.py

The script writes ``docs/_static/elastic_net_leaf_space.png``.
"""

import matplotlib.patches
import numpy as np
from pathlib import Path

from torch_bsf.model_selection.elastic_net_grid import elastic_net_grid


# ---------------------------------------------------------------------------
# Color helper
# ---------------------------------------------------------------------------

def weights_to_rgb(w1, w2, w3):
    """Map weight vectors (w1, w2, w3) to RGB colors."""
    return np.clip(np.stack([w1, w2, w3], axis=-1), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------

def project_to_2d(points):
    """Project 3-D simplex points to 2-D Cartesian coordinates.

    Vertex mapping::

        (1, 0, 0)  ->  (0,   0      )  # data-only vertex (lambda=0), bottom-left
        (0, 1, 0)  ->  (0.5, sqrt3/2)  # pure-L1 vertex, top
        (0, 0, 1)  ->  (1,   0      )  # pure-L2 vertex, bottom-right
    """
    w1, w2, w3 = points[:, 0], points[:, 1], points[:, 2]
    px = 0.5 * w2 + w3
    py = np.sqrt(3) / 2 * w2
    return px, py


def draw_simplex_boundary(ax, **kwargs):
    """Draw the triangle boundary on *ax*."""
    triangle = matplotlib.patches.Polygon(
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


def main():
    """Generate the elastic-net leaf-space figure and write it to disk."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    grid = elastic_net_grid(
        n_lambdas=N_LAMBDAS,
        n_alphas=N_ALPHAS,
        base=BASE,
    )
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
    h3 = np.sqrt(3) / 2   # used in both center and right panels

    # ---------------------------------------------------------------------------
    # Figure layout: three panels side by side
    # ---------------------------------------------------------------------------

    fig, (ax_rect, ax_simplex, ax_leaf) = plt.subplots(
        1, 3, figsize=(16, 5.2), gridspec_kw={"wspace": 0.45}
    )


    # ===========================================================================
    # Left panel – (λ, α) hyperparameter space
    # x-axis: λ (regularization strength), y-axis: α (L1 mixing ratio).
    # Points are colored by (w1, w2, w3) = (R, G, B).
    # ===========================================================================
    ax = ax_rect

    ax.set_xlim(-lambda_max * 0.04, lambda_max)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"$\lambda$  (regularization strength)", fontsize=11)
    ax.set_ylabel(r"$\alpha$  (L1 mixing ratio)", fontsize=11)
    ax.set_title("Hyperparameter space\n(before identification)", fontsize=11)

    # Vertical leaf lines at each λ level
    for lam in lambda_unique:
        ax.axvline(x=lam, color="gray", lw=0.5, alpha=0.4, zorder=2)

    # Grid scatter colored by (w1, w2, w3)
    for lam, w1 in zip(lambda_unique, unique_w1_finite):
        w1_row = np.full(N_ALPHAS, w1)
        w2_row = lam * alphas_per_row / (1.0 + lam)
        w3_row = lam * (1.0 - alphas_per_row) / (1.0 + lam)
        ax.scatter(
            np.full(N_ALPHAS, lam),
            alphas_per_row,
            c=weights_to_rgb(w1_row, w2_row, w3_row),
            s=20,
            zorder=4,
            edgecolors="none",
        )

    # Identified edge: λ=0 → all points collapse to w=(1,0,0) = red
    ax.axvline(x=0.0, color=(1.0, 0.0, 0.0), lw=2.5, zorder=3)
    ax.scatter([0.0, 0.0], [0.0, 1.0],
            c=[(1.0, 0.0, 0.0), (1.0, 0.0, 0.0)], s=60, zorder=5,
            edgecolors="none")

    # Annotation
    ax.annotate(
        "identified to one point\n$(w_1, w_2, w_3) = (1, 0, 0)$",
        xy=(0.0, 0.5),
        xytext=(lambda_max * 0.14, 0.5),
        fontsize=8.5,
        ha="left",
        va="center",
        color="darkred",
        arrowprops=dict(arrowstyle="-[", color="darkred", lw=1.5,
                        connectionstyle="arc3,rad=0"),
    )

    # Corner labels
    ax.text(-lambda_max * 0.035, -0.06, r"$\alpha=0$" "\n(pure L2)",
            fontsize=8, va="top", ha="right")
    ax.text(-lambda_max * 0.035, 1.06, r"$\alpha=1$" "\n(pure L1)",
            fontsize=8, va="bottom", ha="right")
    ax.tick_params(axis="both", which="both", length=3)
    ax.set_box_aspect(1)
    # Horizontal grid lines at each α level in the data
    for alpha_val in alphas_per_row:
        ax.axhline(y=alpha_val, color="gray", lw=0.5, alpha=0.4, zorder=1)


    # ===========================================================================
    # Center panel – 2-simplex with foliation (leaves) and grid points
    # Vertices: (1,0,0) at bottom-left, (0,1,0) at top, (0,0,1) at bottom-right.
    # Points are colored by (w1, w2, w3) = (R, G, B).
    # ===========================================================================
    ax = ax_simplex

    ax.set_aspect("equal")
    ax.axis("off")

    # Manual radial grid lines from vertex (1,0,0)=(0,0) toward each α value on
    # the base edge — only at α levels present in the data.
    # Base edge point for α: (0, α, 1-α)  →  (1 - 0.5*α, sqrt3/2 * α)
    for alpha_val in alphas_per_row:
        bx = 1.0 - 0.5 * alpha_val   # x of base-edge point
        by = h3 * alpha_val           # y of base-edge point
        ax.plot([0.0, bx], [0.0, by], color="gray", lw=0.5, alpha=0.4, zorder=0)

    ax.set_title("Elastic-net grid on the 2-simplex\n(after identification)", fontsize=11, pad=34)

    # Triangle boundary
    draw_simplex_boundary(ax)

    # Highlight the base edge connecting (0,1,0) and (0,0,1)
    # In this layout: (0,1,0) → (0.5, sqrt3/2) top; (0,0,1) → (1,0) bottom-right
    # This is the right side of the triangle.
    t_base = np.linspace(0.0, 1.0, 100)
    x_base = 0.5 + 0.5 * t_base      # from (0.5, sqrt3/2) to (1, 0)
    y_base = h3 * (1.0 - t_base)
    pts_base = np.column_stack([x_base, y_base])
    segs_base = np.stack([pts_base[:-1], pts_base[1:]], axis=1)
    t_base_mids = 0.5 * (t_base[:-1] + t_base[1:])
    # (0,1,0)→green at t=0, (0,0,1)→blue at t=1
    c_base = np.stack([np.zeros_like(t_base_mids), 1.0 - t_base_mids, t_base_mids], axis=-1)
    ax.add_collection(LineCollection(segs_base, colors=c_base, lw=2.5, zorder=2))

    # Leaf lines: iso-w1 segments (in this layout these are diagonal)
    # At fixed w1: from (w1, 1-w1, 0) → (0.5*(1-w1), h3*(1-w1)) to (w1, 0, 1-w1) → (1-w1, 0)
    # Color: from (w1, 1-w1, 0) [green end] to (w1, 0, 1-w1) [blue end]
    # Skip w1=0 (that is the base edge itself, already drawn above)
    for w1 in unique_w1_finite:
        x_t = 0.5 * (1.0 - w1)          # (w1, 1-w1, 0)  top end
        y_t = h3 * (1.0 - w1)
        x_r = 1.0 - w1                  # (w1, 0, 1-w1)  bottom-right end
        y_r = 0.0
        t_lf = np.linspace(0.0, 1.0, 30)
        x_lf = np.linspace(x_t, x_r, 30)
        y_lf_arr = np.linspace(y_t, y_r, 30)
        w2_lf = (1.0 - t_lf) * (1.0 - w1)   # 1-w1 → 0
        w3_lf = t_lf * (1.0 - w1)            # 0 → 1-w1
        c_lf = weights_to_rgb(np.full(30, w1), w2_lf, w3_lf)
        pts_lf = np.column_stack([x_lf, y_lf_arr])
        segs_lf = np.stack([pts_lf[:-1], pts_lf[1:]], axis=1)
        c_lf_mids = 0.5 * (c_lf[:-1] + c_lf[1:])
        ax.add_collection(LineCollection(segs_lf, colors=c_lf_mids,
                                        lw=1.0, alpha=0.7, zorder=1))

    # Scatter grid points colored by (w1, w2, w3)
    ax.scatter(px, py, c=rgb_all, s=20, zorder=3, edgecolors="none")

    # Vertex markers and labels
    ax.plot(0.0, 0.0, "o", color=(1.0, 0.0, 0.0), markersize=8, zorder=5)
    ax.plot(0.5, h3, "o", color=(0.0, 1.0, 0.0), markersize=8, zorder=5)
    ax.plot(1.0, 0.0, "o", color=(0.0, 0.0, 1.0), markersize=8, zorder=5)

    offset = 0.05
    ax.text(0.0 - offset, 0.0, r"$(1,0,0)$" "\n" r"$\lambda=0$",
            ha="right", va="center", fontsize=8.5, color=(0.8, 0.0, 0.0))
    ax.text(0.5, h3 + offset, r"$(0,1,0)$" "\n" r"$(\lambda\!\to\!\infty,\ \alpha\!=\!1)$",
            ha="center", va="bottom", fontsize=8.5, color=(0.0, 0.65, 0.0))
    ax.text(1.0 + offset, 0.0, r"$(0,0,1)$" "\n" r"$(\lambda\!\to\!\infty,\ \alpha\!=\!0)$",
            ha="left", va="center", fontsize=8.5, color=(0.0, 0.0, 0.9))
    ax.text(0.82, h3 * 0.45, "base edge\n(null model)", fontsize=7.5, ha="center",
            color="dimgray", rotation=-60)


    # ===========================================================================
    # Right panel – quotient (leaf/eye) space, rotated 90° counterclockwise
    #
    # The 2-simplex has its base edge (connecting (0,1,0) and (0,0,1)) collapsed
    # to a single point P*.  The resulting CW complex has:
    #   - vertex A  = (1,0,0)  at the LEFT        (lambda = 0)
    #   - vertex P* = collapsed base edge  at the RIGHT  (lambda -> inf, null model)
    #   - two 1-cells (edges) from A to P*, shown as curves
    #   - one 2-cell (face) enclosed by the two edges
    # The overall shape resembles a leaf or eye (horizontal orientation).
    # Points are colored by (w1, w2, w3) = (R, G, B).
    # At P*, a large green dot (behind) and a smaller blue dot (in front) are
    # overlaid to show that (0,1,0) and (0,0,1) are both mapped here.
    # ===========================================================================
    ax = ax_leaf

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Quotient space\n(base edge identified to null model $P^*$)", fontsize=11, pad=28)

    R = 0.55  # half-width of the widest cross-section

    # Rotation helper: 90° counterclockwise  (x, y) → (-y, x)
    def rot90ccw(x, y):
        return -y, x

    # Boundary curves with color gradient
    # t parameterizes from A (t=0) to P* (t=1)
    t_bnd = np.linspace(0.0, 1.0, 300)
    # Before rotation in the (x, y)-plane:
    #   left boundary (α=0):  x = -R * sin(π t),  y = cos(π t)
    #   right boundary (α=1): x =  R * sin(π t),  y = cos(π t)
    # After 90° CCW rotation (x, y) → (-y, x):
    #   α=0 boundary → new_x = -cos(π t), new_y = -R * sin(π t)  (bottom, y ≤ 0)
    #   α=1 boundary → new_x = -cos(π t), new_y =  R * sin(π t)  (top,    y ≥ 0)
    x_alpha0_bnd, y_alpha0_bnd = rot90ccw(-R * np.sin(np.pi * t_bnd), np.cos(np.pi * t_bnd))
    x_alpha1_bnd, y_alpha1_bnd = rot90ccw( R * np.sin(np.pi * t_bnd), np.cos(np.pi * t_bnd))

    # Bottom boundary (α=0): color transitions red→blue  (w1=1-t, w2=0, w3=t)
    pts_bb = np.column_stack([x_alpha0_bnd, y_alpha0_bnd])
    segs_bb = np.stack([pts_bb[:-1], pts_bb[1:]], axis=1)
    t_bnd_mids = 0.5 * (t_bnd[:-1] + t_bnd[1:])
    c_bb = np.stack([1.0 - t_bnd_mids, np.zeros_like(t_bnd_mids), t_bnd_mids], axis=-1)
    ax.add_collection(LineCollection(segs_bb, colors=c_bb, lw=2.0, zorder=3))

    # Top boundary (α=1): color transitions red→green  (w1=1-t, w2=t, w3=0)
    pts_tb = np.column_stack([x_alpha1_bnd, y_alpha1_bnd])
    segs_tb = np.stack([pts_tb[:-1], pts_tb[1:]], axis=1)
    c_tb = np.stack([1.0 - t_bnd_mids, t_bnd_mids, np.zeros_like(t_bnd_mids)], axis=-1)
    ax.add_collection(LineCollection(segs_tb, colors=c_tb, lw=2.0, zorder=3))

    # Leaves: constant-λ (constant w1) segments — now vertical after rotation
    # Skip w1=0: it maps to P* (zero-width leaf) and is shown by the vertex marker
    for w1 in unique_w1_finite:
        t = 1.0 - w1
        y_lf = np.cos(np.pi * t)    # original y of the leaf line (before rotation)
        width = R * np.sin(np.pi * t)
        # Before rotation: horizontal segment from (-width, y_lf) to (width, y_lf)
        # After 90° CCW: x = -y_lf (constant), y from width to -width
        ax.plot([-y_lf, -y_lf], [-width, width], color="gray", lw=0.7, alpha=0.45)

    # Constant-α curves from A to P* (now flow top-to-bottom after rotation)
    alpha_lines = np.linspace(0.0, 1.0, N_ALPHAS)
    t_line = np.linspace(0.0, 1.0, 200)
    for alpha in alpha_lines:
        x_orig = (2.0 * alpha - 1.0) * R * np.sin(np.pi * t_line)
        y_orig = np.cos(np.pi * t_line)
        x_rot, y_rot = rot90ccw(x_orig, y_orig)
        ax.plot(x_rot, y_rot, color="gray", lw=0.5, alpha=0.35)

    # Grid scatter colored by (w1, w2, w3)
    grid_main = grid[:n_main]
    w1_m = grid_main[:, 0]
    w2_m = grid_main[:, 1]
    w3_m = grid_main[:, 2]
    rgb_m = weights_to_rgb(w1_m, w2_m, w3_m)
    w23_sum = w2_m + w3_m
    # On the base edge (w1=0, w2+w3=1) we define alpha = w2 / (w2 + w3).
    # Use a fallback of 0.5 when w2 + w3 is (numerically) zero to avoid division
    # by zero in degenerate cases; this does not occur for the current grid_main
    # but is kept as a defensive default.
    mask = w23_sum > 1e-10
    alpha_m = np.full_like(w23_sum, 0.5, dtype=float)
    np.divide(w2_m, w23_sum, out=alpha_m, where=mask)
    x_eye, y_eye = to_eye_coords(w1_m, alpha_m, r=R)
    x_eye_rot, y_eye_rot = rot90ccw(x_eye, y_eye)
    ax.scatter(x_eye_rot, y_eye_rot, c=rgb_m, s=20, zorder=3, edgecolors="none")

    # Vertex A: w=(1,0,0) = red — now at LEFT (-1, 0)
    ax.plot(-1, 0, "o", color=(1.0, 0.0, 0.0), markersize=8, zorder=5)

    # Collapsed point P*: large green dot (behind) + smaller blue dot (in front) — now at RIGHT (1, 0)
    # This visualizes that both (0,1,0) and (0,0,1) map to P*.
    ax.plot(1, 0, "o", color=(0.0, 1.0, 0.0), markersize=14, zorder=4)  # green, larger
    ax.plot(1, 0, "o", color=(0.0, 0.0, 1.0), markersize=8,  zorder=5)  # blue, smaller

    # Labels
    ax.text(-1.10, 0, r"$A$: $(1, 0, 0)$, $\lambda=0$",
            ha="right", va="center", fontsize=8.5, color=(0.8, 0.0, 0.0))
    ax.text(1.12, 0,
            r"$P^*$: null model" "\n" r"$(0,1,0)$ and $(0,0,1)$ identified",
            ha="left", va="center", fontsize=8.0, color="dimgray")

    # Edge labels at the widest point (now top and bottom of the shape)
    ax.text(0, R + 0.06, r"$\alpha=1$" "\n(pure L1)",
            ha="center", va="bottom", fontsize=8)
    ax.text(0, -(R + 0.06), r"$\alpha=0$" "\n(pure L2)",
            ha="center", va="top", fontsize=8)


    # ===========================================================================
    # Save
    # ===========================================================================

    _repo_root = Path(__file__).resolve().parent.parent
    _out_path = _repo_root / "docs" / "_static" / "elastic_net_leaf_space.png"
    _out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {_out_path}")


if __name__ == "__main__":
    main()
