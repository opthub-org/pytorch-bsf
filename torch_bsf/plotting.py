import itertools
import math

import numpy as np
import torch
from torch_bsf.bezier_simplex import BezierSimplex
from torch_bsf.simploidal import BezierSimploid, SimploidalBezierSpace, simploid_indices

# Maximum number of sample points for the pairwise scatter plot.
# When the combinatorial meshgrid would exceed this count, uniformly-random
# simplex samples are drawn instead to bound memory usage and plot time.
_MAX_PAIRWISE_POINTS = 2000

# Default maximum number of control points to render in the pairwise plot.
# Control point count grows combinatorially with degree and n_params, so a cap
# prevents worst-case slowdowns when show_control_points=True.
_MAX_CONTROL_POINTS = 500


def plot_bezier_simplex(
    model: BezierSimplex,
    num: int = 100,
    ax=None,
    show_control_points: bool = True,
    *,
    max_control_points: int = _MAX_CONTROL_POINTS,
    max_pairwise_points: int = _MAX_PAIRWISE_POINTS,
    **kwargs,
):
    """Plots the Bézier simplex.

    Parameters
    ----------
    model : BezierSimplex
        The Bézier simplex model to plot.
    num : int
        The number of grid points for each edge.  For ``model.n_params >= 4``
        this value is used only to decide whether to use a full meshgrid or
        random sampling: if the combinatorial meshgrid size
        ``comb(num + n_params - 1, n_params - 1)`` exceeds
        ``max_pairwise_points``, exactly ``max_pairwise_points`` uniformly
        random simplex samples are drawn instead and ``num`` no longer
        controls the sample count.
    ax : matplotlib.axes.Axes or None
        The matplotlib axes to plot on. If None, a new figure is created.
        Ignored when ``model.n_params >= 4``; for pairwise plots a new
        figure is created only when ``model.n_values > 0`` (when
        ``model.n_values == 0`` an empty ``(0, 0)`` ndarray is returned
        without creating a figure).
    show_control_points : bool
        Whether to show control points.
    max_control_points : int
        Maximum number of control points to render in pairwise plots
        (``model.n_params >= 4``).  When the model has more control points
        than this limit, a random subset of ``max_control_points`` is drawn
        instead to avoid combinatorial slowdowns.  Defaults to 500.
        Ignored when ``show_control_points`` is ``False`` or
        ``model.n_params < 4``.  Must be a non-negative integer.
    max_pairwise_points : int
        Maximum number of sample points for the pairwise scatter plot
        (``model.n_params >= 4``).  When the combinatorial meshgrid size
        would exceed this limit, ``max_pairwise_points`` uniformly random
        simplex samples are drawn instead to bound memory usage and plot
        time.  Defaults to 2000.  Ignored when ``model.n_params < 4``.
        Must be a non-negative integer.
    **kwargs
        Additional keyword arguments forwarded to the plot call.
        For ``model.n_params == 2``, forwarded to ``ax.plot`` (curve).
        For ``model.n_params == 3`` and ``model.n_values >= 3``, forwarded
        to ``ax.plot_trisurf`` (3D surface).
        For ``model.n_params == 3`` and ``model.n_values == 2``, ignored.
        For ``model.n_params >= 4``, forwarded to ``ax.scatter`` (pairwise).

    Returns
    -------
    matplotlib.axes.Axes or mpl_toolkits.mplot3d.axes3d.Axes3D or numpy.ndarray
        The axes containing the plot.  For ``model.n_params <= 3`` a single
        ``Axes`` (or ``Axes3D``) is returned.  For ``model.n_params >= 4`` a
        2-D ``numpy.ndarray`` of ``Axes`` with shape
        ``(n_values, n_values)`` is returned (pairwise scatter plot).

    Raises
    ------
    ImportError
        If matplotlib is not installed. This dependency is required for all
        plotting backends used by this function.
    ImportError
        If SciPy is not installed and ``model.n_params == 3``. SciPy is
        required for the triangulation-based plotting used in the
        Bézier triangle case.
    ValueError
        If ``model.n_params < 2``. This function only supports Bézier simplex
        models with at least two parameters.
    """
    if model.n_params == 2:
        return _plot_bezier_curve(model, num, ax, show_control_points, **kwargs)
    if model.n_params == 3:
        return _plot_bezier_triangle(model, num, ax, show_control_points, **kwargs)
    if model.n_params >= 4:
        return _plot_bezier_simplex_pairwise(
            model, num, show_control_points, max_control_points, max_pairwise_points, **kwargs
        )
    raise ValueError(
        f"plot_bezier_simplex only supports models with n_params >= 2; got n_params = {model.n_params}"
    )


def _plot_bezier_curve(model, num, ax, show_control_points, **kwargs):
    """Plots a Bézier curve (n_params == 2).

    For ``model.n_values == 2``, this function produces a 2D plot using
    the first two output components (``xs[:, 0]`` and ``xs[:, 1]``).
    For ``model.n_values >= 3``, this function produces a 3D plot using
    only the first three output components (``xs[:, 0]``, ``xs[:, 1]``,
    and ``xs[:, 2]``); any additional output dimensions are ignored for
    visualization.
    For ``model.n_values < 2``, no data is plotted and the axes are
    returned empty.

    Parameters
    ----------
    model : BezierSimplex
        The Bézier simplex model to plot.
    num : int
        The number of grid points along the curve.
    ax : matplotlib.axes.Axes or None
        The matplotlib axes to plot on. If None, a new figure is created.
    show_control_points : bool
        Whether to overlay the control points.
    **kwargs
        Additional keyword arguments forwarded to the plot call.

    Returns
    -------
    matplotlib.axes.Axes or mpl_toolkits.mplot3d.axes3d.Axes3D
        The axes containing the plot. Returns a 2D ``Axes`` when
        ``model.n_values == 2``, or a 3D ``Axes3D`` when
        ``model.n_values >= 3``.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        ) from e

    ts, xs = model.meshgrid(num=num)
    xs = xs.detach().cpu().numpy()

    if ax is None:
        fig = plt.figure()
        if model.n_values >= 3:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

    if model.n_values == 2:
        ax.plot(xs[:, 0], xs[:, 1], **kwargs)
        if show_control_points:
            cp = model.control_points.matrix.detach().cpu().numpy()
            ax.scatter(cp[:, 0], cp[:, 1], c="r", marker="o", label="Control Points")
            # Connect control points with a dashed line
            ax.plot(cp[:, 0], cp[:, 1], "r--", alpha=0.3)
    elif model.n_values >= 3:
        ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], **kwargs)
        if show_control_points:
            cp = model.control_points.matrix.detach().cpu().numpy()
            ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2], c="r", marker="o", label="Control Points")
            ax.plot(cp[:, 0], cp[:, 1], cp[:, 2], "r--", alpha=0.3)

    return ax


def _plot_bezier_triangle(model, num, ax, show_control_points, **kwargs):
    """Plots a Bézier triangle using triangulation (n_params == 3).

    Depending on ``model.n_values``, this produces either:

    * a 2D triangulated plot plus scattered points when ``model.n_values == 2``, or
    * a 3D triangulated surface plot when ``model.n_values >= 3``.

    Parameters
    ----------
    model : BezierSimplex
        The Bézier simplex model to plot.
    num : int
        The number of grid points along each edge of the triangle.
    ax : matplotlib.axes.Axes or None
        The matplotlib axes to plot on. If None, a new figure is created.
    show_control_points : bool
        Whether to overlay the control points.
    **kwargs
        Additional keyword arguments forwarded to the 3D surface plot
        (``ax.plot_trisurf``) when ``model.n_values >= 3``.

    Returns
    -------
    matplotlib.axes.Axes or mpl_toolkits.mplot3d.axes3d.Axes3D
        The axes containing the plot. Returns a 2D ``Axes`` when
        ``model.n_values == 2``, or a 3D ``Axes3D`` when
        ``model.n_values >= 3``.

    Raises
    ------
    ImportError
        If matplotlib or scipy is not installed.
    """
    # This requires a bit more complex triangulation for plotting a surface
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        ) from e
    try:
        from scipy.spatial import Delaunay
    except ImportError as e:
        raise ImportError(
            "scipy is required for triangle plotting. "
            "Install it with: pip install scipy"
        ) from e

    ts, xs = model.meshgrid(num=num)
    xs = xs.detach().cpu().numpy()

    # Project 3D simplex parameters to 2D for triangulation
    # (Using the same projection as in examples)
    t_np = ts.detach().cpu().numpy()
    px = t_np[:, 1] + 0.5 * t_np[:, 2]
    py = np.sqrt(3) / 2 * t_np[:, 2]

    tri = Delaunay(np.stack([px, py], axis=1))

    if ax is None:
        fig = plt.figure()
        if model.n_values >= 3:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)

    if model.n_values == 2:
        # Use the triangulation connectivity in value space for a consistent 2D plot
        ax.triplot(xs[:, 0], xs[:, 1], tri.simplices, alpha=0.3)
        # We can't easily plot a manifold surface in 2D values unless we pick 2 dimensions
        ax.scatter(xs[:, 0], xs[:, 1], alpha=0.1, s=1)
        if show_control_points:
            cp = model.control_points.matrix.detach().cpu().numpy()
            ax.scatter(cp[:, 0], cp[:, 1], c="r", marker="o")
    elif model.n_values >= 3:
        ax.plot_trisurf(
            xs[:, 0], xs[:, 1], xs[:, 2], triangles=tri.simplices, alpha=0.5, **kwargs
        )
        if show_control_points:
            cp = model.control_points.matrix.detach().cpu().numpy()
            ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2], c="r", marker="o")

    return ax


def _plot_bezier_simplex_pairwise(model, num, show_control_points, max_control_points, max_pairwise_points, **kwargs):
    """Plots a high-dimensional Bézier simplex as a pairwise scatter plot.

    For ``model.n_params >= 4``, this function generates sample points from
    the simplex and creates a pairwise scatter plot (pair plot) of the output
    values.  Diagonal panels show histograms of individual output dimensions;
    off-diagonal panels show scatter plots of pairs of output dimensions.

    Parameters
    ----------
    model : BezierSimplex
        The Bézier simplex model to plot.
    num : int
        The number of grid points along each edge of the simplex.  Used to
        estimate the full meshgrid size via
        ``comb(num + n_params - 1, n_params - 1)``.  If that size exceeds
        ``max_pairwise_points``, ``num`` is ignored and exactly
        ``max_pairwise_points`` uniformly random simplex samples are drawn
        instead; in this case ``num`` no longer controls the sample count.
    show_control_points : bool
        Whether to overlay control points on the off-diagonal scatter panels.
    max_control_points : int
        Maximum number of control points to render.  When the model has more
        control points than this limit, a random subset of ``max_control_points``
        is drawn instead to prevent combinatorial slowdowns for high-degree
        models.  Defaults to 500.  Ignored when
        ``show_control_points`` is ``False``.  Must be a non-negative integer.
    max_pairwise_points : int
        Maximum number of sample points for the pairwise scatter plot.  When
        the combinatorial meshgrid size exceeds this limit,
        ``max_pairwise_points`` uniformly random simplex samples are drawn
        instead to bound memory usage and plot time.  Defaults to 2000.
        Must be a non-negative integer.
    **kwargs
        Additional keyword arguments forwarded to ``ax.scatter``.

    Returns
    -------
    numpy.ndarray of matplotlib.axes.Axes
        A 2-D array of axes with shape ``(n_values, n_values)``.

    Raises
    ------
    ValueError
        If ``max_control_points`` or ``max_pairwise_points`` is negative or
        not an integer.
    ImportError
        If matplotlib is not installed.
    """
    if not isinstance(max_control_points, int) or max_control_points < 0:
        raise ValueError(
            f"max_control_points must be a non-negative integer; got {max_control_points!r}"
        )
    if not isinstance(max_pairwise_points, int) or max_pairwise_points < 0:
        raise ValueError(
            f"max_pairwise_points must be a non-negative integer; got {max_pairwise_points!r}"
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib"
        ) from e

    # Early return before any sampling when there are no output values.
    n_v = model.n_values
    if n_v == 0:
        return np.empty((0, 0), dtype=object)

    n_p = model.n_params
    meshgrid_size = math.comb(num + n_p - 1, n_p - 1) if n_p > 0 else 1
    if meshgrid_size > max_pairwise_points:
        # The full combinatorial grid would be too large; use random simplex
        # samples instead to bound memory usage and plot time.
        from torch_bsf.sampling import simplex_random

        cp_matrix = model.control_points.matrix
        ts = simplex_random(n_params=n_p, n_samples=max_pairwise_points).to(
            device=cp_matrix.device, dtype=cp_matrix.dtype
        )
        with torch.no_grad():
            xs = model(ts).detach().cpu().numpy()
    else:
        with torch.no_grad():
            _ts, xs_t = model.meshgrid(num=num)
        xs = xs_t.detach().cpu().numpy()

    panel_size = max(1, min(3, 12 // max(n_v, 1)))
    figsize_dim = min(12, panel_size * n_v)
    fig, axes = plt.subplots(
        n_v, n_v, squeeze=False, figsize=(figsize_dim, figsize_dim)
    )
    if show_control_points:
        # Work on the torch tensor first to avoid transferring an enormous
        # control-point matrix to CPU when max_control_points is small.
        cp_t = model.control_points.matrix
        cp_len = cp_t.shape[0]
        if cp_len > max_control_points:
            idx = torch.randperm(cp_len, device=cp_t.device)[:max_control_points]
            cp_t = cp_t[idx]
        cp = cp_t.detach().cpu().numpy()
    else:
        cp = None

    # Allow caller to override scatter defaults via kwargs
    scatter_s = kwargs.pop("s", 1)
    scatter_alpha = kwargs.pop("alpha", 0.3)
    # Compute a reasonable bin count (Sturges' rule, minimum 10)
    n_samples = len(xs)
    bins = max(10, int(np.ceil(np.log2(n_samples))) + 1) if n_samples > 1 else 10

    for i in range(n_v):
        for j in range(n_v):
            a = axes[i, j]
            if i == j:
                a.hist(xs[:, i], bins=bins)
                if cp is not None:
                    x_vals = cp[:, i]
                    ymin, ymax = a.get_ylim()
                    a.vlines(x_vals, ymin, ymax, colors="r", alpha=0.5, linewidth=1)
            else:
                a.scatter(xs[:, j], xs[:, i], s=scatter_s, alpha=scatter_alpha, **kwargs)
                if cp is not None:
                    a.scatter(cp[:, j], cp[:, i], c="r", s=20, marker="o", zorder=5)

    return axes


# ---------------------------------------------------------------------------
# Bézier simploids and simploidal Bézier spaces
# ---------------------------------------------------------------------------

# The vertices of each factor of a simploid spanned by one of its cells: one tuple per factor.
_Cell = tuple[tuple[int, ...], ...]


def _check_simploid_plot_args(n_values: int, num: int, ax, name: str) -> None:
    """Validates the arguments shared by the simploid plotting functions."""
    if n_values not in (2, 3):
        raise ValueError(f"{name} only supports n_values of 2 or 3; got n_values = {n_values}")
    if isinstance(num, bool) or not isinstance(num, int) or num < 1:
        raise ValueError(f"num must be a positive integer; got {num!r}")
    if ax is not None:
        is_3d = getattr(ax, "name", None) == "3d"
        if n_values == 3 and not is_3d:
            raise ValueError("n_values == 3 requires a 3D axes; create it with projection='3d'")
        if n_values == 2 and is_3d:
            raise ValueError("n_values == 2 requires a 2D axes, but a 3D axes is given")


def _require_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib") from e
    return plt


def _cells_to_draw(simploid_type: tuple[int, ...]) -> list[_Cell]:
    """The cells through which a simploid is drawn.

    A simploid of dimension at most 2 is drawn as a whole.  A simploid of
    dimension 3 or more is drawn through its 2-dimensional faces: a triangle
    of one factor, or an edge of each of two factors, every other factor
    being fixed to one of its vertices.
    """
    if sum(simploid_type) <= 2:
        return [tuple(tuple(range(a + 1)) for a in simploid_type)]
    n = len(simploid_type)
    cells: list[_Cell] = []
    for active in [(i,) for i in range(n)] + list(itertools.combinations(range(n), 2)):
        size = 3 if len(active) == 1 else 2
        per_factor = [
            list(itertools.combinations(range(a + 1), size)) if i in active else [(v,) for v in range(a + 1)]
            for i, a in enumerate(simploid_type)
        ]
        cells.extend(tuple(c) for c in itertools.product(*per_factor))
    return cells


def _cell_grid(simploid_type: tuple[int, ...], cell: _Cell, num: int) -> tuple[torch.Tensor, np.ndarray]:
    """A grid on a cell, as simploid parameters and as integer lattice coordinates.

    Returns
    -------
    params
        The grid points as simploid parameters, of shape ``(N, sum(a_i + 1))``.
    coords
        The integer coordinates of the grid points on the cell, of shape
        ``(N, dim)``: one column per spanned edge, or two columns for a
        spanned triangle.
    """
    from torch_bsf.sampling import simplex_grid

    params = torch.zeros(1, 0)
    coords = np.zeros((1, 0), dtype=int)
    for a, verts in zip(simploid_type, cell):
        g = simplex_grid(len(verts), num) if len(verts) > 1 else torch.ones(1, 1)
        t = torch.zeros(g.shape[0], a + 1)
        t[:, list(verts)] = g
        c = np.rint(g[:, 1:].numpy() * num).astype(int)
        m, k = params.shape[0], t.shape[0]
        params = torch.cat([params.repeat_interleave(k, dim=0), t.repeat(m, 1)], dim=1)
        coords = np.concatenate([np.repeat(coords, k, axis=0), np.tile(c, (m, 1))], axis=1)
    return params, coords


def _cell_triangles(coords: np.ndarray, num: int, is_triangle: bool) -> np.ndarray:
    """Triangulates the lattice grid of a triangle or a square cell."""
    row = {(int(i), int(j)): r for r, (i, j) in enumerate(coords)}
    tris = []
    for i in range(num):
        for j in range(num - i if is_triangle else num):
            tris.append([(i, j), (i + 1, j), (i, j + 1)])
            if not is_triangle or i + j + 1 < num:
                tris.append([(i + 1, j), (i + 1, j + 1), (i, j + 1)])
    return np.array([[row[p] for p in tri] for tri in tris], dtype=int)


def _boundary_edges(triangles: np.ndarray) -> np.ndarray:
    """The edges on the boundary of a triangulated cell, as pairs of rows of shape ``(n, 2)``."""
    count: dict[tuple[int, int], int] = {}
    for a, b, c in triangles.tolist():
        for e in ((a, b), (b, c), (c, a)):
            key = (min(e), max(e))
            count[key] = count.get(key, 0) + 1
    return np.array([e for e, n in count.items() if n == 1], dtype=int).reshape(-1, 2)


def _add_segments(ax, points: np.ndarray, edges: np.ndarray, **kwargs) -> None:
    """Draws the segments ``points[edges]`` on a 2D or 3D axes."""
    if len(edges) == 0:
        return
    segments = list(points[edges])
    if points.shape[1] == 3:
        from mpl_toolkits.mplot3d.art3d import Line3DCollection  # type: ignore

        ax.add_collection3d(Line3DCollection(segments, **kwargs))
    else:
        from matplotlib.collections import LineCollection

        ax.add_collection(LineCollection(segments, **kwargs))


def _with_defaults(kwargs: dict, defaults: dict[tuple[str, ...], object]) -> dict:
    """``kwargs`` plus the defaults whose keyword is given under none of its aliases.

    Each key of ``defaults`` is a group of Matplotlib aliases, the first one
    being the name used for the default; e.g. ``("color", "c")``.
    """
    result = dict(kwargs)
    for aliases, value in defaults.items():
        if not any(a in kwargs for a in aliases):
            result[aliases[0]] = value
    return result


def _draw_cell(ax, xs: np.ndarray, coords: np.ndarray, num: int, is_triangle: bool, color, kwargs: dict) -> None:
    """Draws the image ``xs`` of a cell grid: a point, a curve or a surface."""
    dim = coords.shape[1]
    if dim == 0:
        ax.scatter(*xs.T, color=color)
        return
    if dim == 1:
        order = np.argsort(coords[:, 0])
        ax.plot(*xs[order].T, **_with_defaults(kwargs, {("color", "c"): color}))
        return
    triangles = _cell_triangles(coords, num, is_triangle)
    surface_kwargs = {"alpha": 0.4, **kwargs}
    if xs.shape[1] == 3:
        # plot_trisurf passes its ``color`` to Poly3DCollection as ``facecolors``,
        # so a single face color given under any alias becomes ``color`` instead;
        # per-triangle face colors are set on the returned collection.
        from matplotlib.colors import is_color_like

        surface_color, face_colors = color, None
        for alias in ("facecolor", "facecolors", "fc"):
            if alias in surface_kwargs:
                face_colors = surface_kwargs.pop(alias)
        if face_colors is not None and is_color_like(face_colors):
            surface_color, face_colors = face_colors, None
        surface = ax.plot_trisurf(xs[:, 0], xs[:, 1], xs[:, 2], triangles=triangles, color=surface_color, **surface_kwargs)
        if face_colors is not None:
            surface.set_facecolor(face_colors)
    else:
        from matplotlib.collections import PolyCollection

        poly_kwargs = _with_defaults(
            surface_kwargs,
            {("facecolor", "facecolors", "fc"): color, ("edgecolor", "edgecolors", "ec"): "none"},
        )
        ax.add_collection(PolyCollection(xs[triangles], **poly_kwargs))
    _add_segments(ax, xs, _boundary_edges(triangles), colors=[color], linewidths=1.0)


def _control_net_edges(indices: list) -> np.ndarray:
    """The edges of a control net, as pairs of rows of shape ``(n, 2)``.

    Two control points are joined when their indices differ by
    :math:`e_k - e_l` in the multi-index of a single factor.
    """
    pos = {alpha: p for p, alpha in enumerate(indices)}
    pairs = []
    for p, alpha in enumerate(indices):
        for i, m in enumerate(alpha):
            for k, l in itertools.permutations(range(len(m)), 2):  # noqa: E741
                if m[l] == 0:
                    continue
                m2 = list(m)
                m2[k] += 1
                m2[l] -= 1
                neighbor = list(alpha)
                neighbor[i] = tuple(m2)
                q = pos[tuple(neighbor)]
                if p < q:
                    pairs.append((p, q))
    return np.array(pairs, dtype=int).reshape(-1, 2)


def _draw_simploid(ax, simploid_type, evaluate, indices, cp, num: int, show_control_points: bool, color, kwargs) -> None:
    """Draws one Bézier simploid, given by its evaluation function and its control net."""
    for cell in _cells_to_draw(simploid_type):
        params, coords = _cell_grid(simploid_type, cell, num)
        with torch.no_grad():
            xs = evaluate(params).detach().cpu().numpy()
        is_triangle = any(len(verts) == 3 for verts in cell)
        _draw_cell(ax, xs, coords, num, is_triangle, color, kwargs)
    if show_control_points:
        cp = cp.detach().cpu().numpy()
        _add_segments(ax, cp, _control_net_edges(indices), colors="r", linestyles="--", alpha=0.3)
        ax.scatter(*cp.T, c="r", marker="o")


def _prepare_axes(ax, n_values: int):
    plt = _require_pyplot()
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d") if n_values == 3 else fig.add_subplot(111)
    return plt, ax


def plot_bezier_simploid(
    model: BezierSimploid,
    num: int = 20,
    ax=None,
    show_control_points: bool = True,
    **kwargs,
):
    r"""Plots a Bézier simploid whose values are 2- or 3-dimensional.

    The image of a grid on the parameter simploid is drawn: a curve for a
    simploid of dimension 1 (type ``(1,)``) and a triangulated surface, with
    its boundary curves, for a simploid of dimension 2 (a triangle ``(2,)``
    or a square ``(1, 1)``).  A simploid of dimension 3 or more (e.g. a prism
    ``(2, 1)`` or a cube ``(1, 1, 1)``) is drawn through all its
    2-dimensional faces, i.e. as the boundary surface of the solid.  A vertex
    (type ``()``) is drawn as a point.

    Parameters
    ----------
    model : torch_bsf.simploidal.BezierSimploid
        The Bézier simploid to plot.  ``model.n_values`` must be 2 or 3.
    num : int
        The number of grid intervals along each edge of the parameter simploid.
    ax : matplotlib.axes.Axes or None
        The axes to plot on: a 2D axes when ``model.n_values == 2`` and a 3D
        axes (``projection="3d"``) when ``model.n_values == 3``.  If None, a
        new figure is created.
    show_control_points : bool
        Whether to show the control net: the control points, and dashed
        segments joining the control points whose indices differ by
        :math:`e_k - e_l` in a single factor.
    **kwargs
        Additional keyword arguments forwarded to the drawing call of the
        curves (``ax.plot``) and surfaces (``ax.plot_trisurf`` in 3D,
        :class:`matplotlib.collections.PolyCollection` in 2D).  ``color``
        sets the color of the model (default ``"C0"``).

    Returns
    -------
    matplotlib.axes.Axes or mpl_toolkits.mplot3d.axes3d.Axes3D
        The axes containing the plot.

    Raises
    ------
    ValueError
        If ``model.n_values`` is neither 2 nor 3, ``num`` is not a positive
        integer, or ``ax`` is a 2D axes for 3 values or a 3D axes for 2 values.
    ImportError
        If matplotlib is not installed.

    Examples
    --------
    The plotting lines are skipped by doctest because matplotlib is optional.

    >>> from torch_bsf.simploidal import BezierSimploid
    >>> from torch_bsf.plotting import plot_bezier_simploid
    >>> patch = BezierSimploid.randn((1, 1), degree=2, n_values=3)
    >>> ax = plot_bezier_simploid(patch, num=10)  # doctest: +SKIP
    >>> ax.name  # doctest: +SKIP
    '3d'
    """
    _check_simploid_plot_args(model.n_values, num, ax, "plot_bezier_simploid")
    _, ax = _prepare_axes(ax, model.n_values)
    kwargs = dict(kwargs)
    color = kwargs.pop("color", "C0")
    _draw_simploid(
        ax, model.simploid_type, model, model.indices(), model.control_points, num, show_control_points, color, kwargs
    )
    if model.n_values == 2:
        ax.autoscale_view()
    return ax


def plot_simploidal_bezier_space(
    space: SimploidalBezierSpace,
    num: int = 20,
    ax=None,
    show_control_points: bool = True,
    **kwargs,
):
    r"""Plots a simploidal Bézier space whose values are 2- or 3-dimensional.

    Every main simploid of the simploidal set (see
    :meth:`~torch_bsf.simploidal.SimploidalSet.main_simploids`) is drawn on
    one axes as :func:`plot_bezier_simploid` draws a Bézier simploid, each in
    its own color from the Matplotlib color cycle.  The boundary curves of the
    pieces are drawn as solid lines, so that pieces sharing a face can be seen
    to meet along it.

    Parameters
    ----------
    space : torch_bsf.simploidal.SimploidalBezierSpace
        The simploidal Bézier space to plot.  ``space.n_values`` must be 2 or 3.
    num : int
        The number of grid intervals along each edge of every piece.
    ax : matplotlib.axes.Axes or None
        The axes to plot on: a 2D axes when ``space.n_values == 2`` and a 3D
        axes (``projection="3d"``) when ``space.n_values == 3``.  If None, a
        new figure is created.
    show_control_points : bool
        Whether to show the control net of every piece.  The control points
        of a shared face are drawn once per piece containing it.
    **kwargs
        Additional keyword arguments forwarded to the drawing call of every
        piece, as in :func:`plot_bezier_simploid`.  ``color`` sets one color
        for all pieces instead of the color cycle.

    Returns
    -------
    matplotlib.axes.Axes or mpl_toolkits.mplot3d.axes3d.Axes3D
        The axes containing the plot.

    Raises
    ------
    ValueError
        If ``space.n_values`` is neither 2 nor 3, ``num`` is not a positive
        integer, or ``ax`` is a 2D axes for 3 values or a 3D axes for 2 values.
    ImportError
        If matplotlib is not installed.

    Examples
    --------
    The plotting lines are skipped by doctest because matplotlib is optional.

    >>> import torch
    >>> from torch_bsf.simploidal import SimploidalSet, SimploidalBezierSpace
    >>> from torch_bsf.plotting import plot_simploidal_bezier_space
    >>> S = SimploidalSet()
    >>> tri, sq = S.add_simplex(2), S.add_simploid((1, 1))
    >>> _ = S.identify(tri.face(0, 2), sq.face(1, 1))
    >>> space = SimploidalBezierSpace(S, degree=2, n_values=2)
    >>> _ = torch.nn.init.normal_(space.proper_control_points)
    >>> ax = plot_simploidal_bezier_space(space, num=10)  # doctest: +SKIP
    >>> ax.name  # doctest: +SKIP
    'rectilinear'
    """
    _check_simploid_plot_args(space.n_values, num, ax, "plot_simploidal_bezier_space")
    plt, ax = _prepare_axes(ax, space.n_values)
    kwargs = dict(kwargs)
    fixed_color = kwargs.pop("color", None)
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    for k, s in enumerate(space.simploidal_set.main_simploids()):
        color = fixed_color if fixed_color is not None else cycle[k % len(cycle)]
        _draw_simploid(
            ax,
            s.simploid_type,
            lambda t, s=s: space.evaluate(s, t),
            simploid_indices(s.simploid_type, space.simploid_degree(s)),
            space.control_points(s),
            num,
            show_control_points,
            color,
            kwargs,
        )
    if space.n_values == 2:
        ax.autoscale_view()
    return ax
