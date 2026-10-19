"""Generate the figure of Bézier simploids and a simploidal Bézier space.

The figure has three panels drawn with :mod:`torch_bsf.plotting`:

1. A Bézier simploid of type ``(1, 1)`` (a saddle-shaped square patch) with its control net,
   drawn by :func:`~torch_bsf.plotting.plot_bezier_simploid`.
2. A Bézier simploid of type ``(2, 1)`` (a prism), drawn through its
   2-dimensional faces (two triangles and three squares).
3. A simploidal Bézier space made of a triangle and a square glued along an
   edge, fitted to samples of a surface and drawn by
   :func:`~torch_bsf.plotting.plot_simploidal_bezier_space`; the two pieces
   meet along the shared edge.

Usage::

    python examples/generate_simploidal_plot.py

The script writes ``docs/_static/simploidal_bezier_space.png``.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from torch_bsf.plotting import plot_bezier_simploid, plot_simploidal_bezier_space  # noqa: E402
from torch_bsf.simploidal import BezierSimploid, SimploidalSet, fit_simploid, fit_simploidal_set, sample_simploid  # noqa: E402


def height(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """The surface fitted in the third panel."""
    return 0.4 * torch.sin(3 * x) * torch.cos(2 * y)


def square_patch() -> BezierSimploid:
    """A biquadratic saddle patch."""
    t = sample_simploid((1, 1), 200, seed=0)
    x, y = t[:, 1], t[:, 3]
    values = torch.stack([x, y, 0.5 * ((2 * x - 1) ** 2 - (2 * y - 1) ** 2)], dim=1)
    return fit_simploid(t, values, simploid_type=(1, 1), degree=2)


def prism() -> BezierSimploid:
    """A twisted triangular prism."""
    t = sample_simploid((2, 1), 300, seed=1)
    u1, u2, h = t[:, 1], t[:, 2], t[:, 4]
    px, py = u1 + 0.5 * u2, 0.866 * u2
    angle = 0.8 * h
    values = torch.stack(
        [px * torch.cos(angle) - py * torch.sin(angle), px * torch.sin(angle) + py * torch.cos(angle), h],
        dim=1,
    )
    return fit_simploid(t, values, simploid_type=(2, 1), degree=2)


def triangle_and_square():
    """A triangle and a square sharing an edge, fitted to the graph of :func:`height`."""
    S = SimploidalSet()
    tri = S.add_simplex(2)
    sq = S.add_simploid((1, 1))
    # The edge (u0, u1) of the triangle is the edge v = (1, 0) of the square.
    S.identify(tri.face(0, 2), sq.face(1, 1))

    t_tri = sample_simploid((2,), 200, seed=2)
    x_tri = t_tri[:, 1] + 0.5 * t_tri[:, 2]
    y_tri = -0.866 * t_tri[:, 2]
    t_sq = sample_simploid((1, 1), 200, seed=3)
    x_sq, y_sq = t_sq[:, 1], t_sq[:, 3]

    def values(x, y):
        return torch.stack([x, y, height(x, y)], dim=1)

    return fit_simploidal_set(S, [(tri, t_tri, values(x_tri, y_tri)), (sq, t_sq, values(x_sq, y_sq))], degree=3)


def main() -> None:
    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    plot_bezier_simploid(square_patch(), num=20, ax=ax)
    ax.set_title("Bézier simploid of type (1, 1)")

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    plot_bezier_simploid(prism(), num=12, ax=ax, color="C2")
    ax.set_title("Bézier simploid of type (2, 1)")

    ax = fig.add_subplot(1, 3, 3, projection="3d")
    plot_simploidal_bezier_space(triangle_and_square(), num=20, ax=ax)
    ax.set_title("Triangle and square glued along an edge")

    for ax in fig.axes:
        ax.view_init(elev=25, azim=-60)

    out_path = Path(__file__).resolve().parent.parent / "docs" / "_static" / "simploidal_bezier_space.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
