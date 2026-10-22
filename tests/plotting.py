import pytest
import torch

from torch_bsf.plotting import plot_bezier_simplex, plot_bezier_simploid, plot_simploidal_bezier_space
from torch_bsf.simploidal import BezierSimploid, SimploidalBezierSpace, SimploidalSet, simploid_indices


def _simplex(n_params: int, degree: int, n_values: int) -> BezierSimploid:
    """A random Bézier simplex with ``n_params`` vertices (a vertex, of type ``()``, if ``n_params == 1``)."""
    return BezierSimploid.randn((n_params - 1,) if n_params > 1 else (), degree, n_values)


@pytest.fixture
def require_matplotlib():
    """Skip the test if matplotlib is not installed and configure the Agg backend."""
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    return mpl


@pytest.fixture
def bezier_curve_2d():
    return _simplex(2, 2, 2)


@pytest.fixture
def bezier_curve_3d():
    return _simplex(2, 2, 3)


@pytest.fixture
def bezier_triangle_2d():
    pytest.importorskip("scipy.spatial")
    return _simplex(3, 2, 2)


@pytest.fixture
def bezier_triangle_3d():
    pytest.importorskip("scipy.spatial")
    return _simplex(3, 2, 3)


def test_plot_bezier_curve_2d_returns_axes(require_matplotlib, bezier_curve_2d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_curve_2d, num=10)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_bezier_curve_3d_returns_axes(require_matplotlib, bezier_curve_3d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_curve_3d, num=10)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_bezier_triangle_2d_returns_axes(require_matplotlib, bezier_triangle_2d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_triangle_2d, num=5)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_bezier_triangle_3d_returns_axes(require_matplotlib, bezier_triangle_3d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_triangle_3d, num=5)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_bezier_simplex_high_dim_returns_axes_array(require_matplotlib):
    import matplotlib.pyplot as plt

    model = _simplex(4, 1, 2)
    result = plot_bezier_simplex(model, num=3)
    assert result is not None
    assert hasattr(result, "shape")
    assert result.shape == (2, 2)
    plt.close(result[0, 0].figure)


def test_plot_bezier_simplex_high_dim_3_values(require_matplotlib):
    import matplotlib.pyplot as plt

    model = _simplex(4, 1, 3)
    result = plot_bezier_simplex(model, num=3)
    assert result is not None
    assert result.shape == (3, 3)
    plt.close(result[0, 0].figure)


def test_plot_bezier_simplex_high_dim_no_control_points(require_matplotlib):
    import matplotlib.pyplot as plt

    model = _simplex(5, 1, 2)
    result = plot_bezier_simplex(model, num=3, show_control_points=False)
    assert result is not None
    assert result.shape == (2, 2)
    plt.close(result[0, 0].figure)


def test_plot_bezier_simplex_pairwise_zero_n_values(require_matplotlib):
    """n_values == 0 must return an empty (0, 0) ndarray, not (1, 1)."""
    import numpy as np

    model = _simplex(4, 1, 0)
    result = plot_bezier_simplex(model, num=3)
    assert isinstance(result, np.ndarray)
    assert result.shape == (0, 0)


def test_plot_bezier_simplex_pairwise_large_n_values_bounded_figsize(require_matplotlib):
    """Figure size must be capped at 12 inches even for large n_values."""
    import matplotlib.pyplot as plt

    model = _simplex(4, 1, 15)
    result = plot_bezier_simplex(model, num=3)
    fig = result[0, 0].figure
    w, h = fig.get_size_inches()
    assert w <= 12.0, f"Figure width {w} exceeds 12-inch cap"
    assert h <= 12.0, f"Figure height {h} exceeds 12-inch cap"
    plt.close(fig)


def test_plot_bezier_simplex_raises_for_low_n_params():
    """plot_bezier_simplex must raise ValueError for anything but a single factor (n_params < 2 or a product)."""
    with pytest.raises(ValueError, match="plot_bezier_simploid"):
        plot_bezier_simplex(BezierSimploid.randn((1, 1), 1, 2), num=3)
    model = _simplex(1, 1, 2)
    with pytest.raises(ValueError, match="plot_bezier_simploid"):
        plot_bezier_simplex(model, num=3)


def test_plot_bezier_simplex_pairwise_sampling_cap(require_matplotlib):
    """When meshgrid would exceed _MAX_PAIRWISE_POINTS, random sampling is used
    and the result still has the expected (n_values, n_values) shape."""
    import math

    import matplotlib.pyplot as plt

    import torch_bsf.plotting as plotting_module

    # n_params=4, num=30 => binom(33, 3) = 5456, which exceeds _MAX_PAIRWISE_POINTS
    n_params, num = 4, 30
    model = _simplex(n_params, 1, 2)
    assert math.comb(num + n_params - 1, n_params - 1) > plotting_module._MAX_PAIRWISE_POINTS, (
        "precondition: meshgrid size must exceed the cap so random sampling is triggered"
    )
    result = plot_bezier_simplex(model, num=num)
    assert result.shape == (2, 2)
    plt.close(result[0, 0].figure)


def test_plot_bezier_simplex_pairwise_max_control_points_default(require_matplotlib):
    """Control points exceeding _MAX_CONTROL_POINTS are subsampled by default."""
    import matplotlib.pyplot as plt

    import torch_bsf.plotting as plotting_module

    # degree=20, n_params=4 => comb(23,3) = 1771 control points > _MAX_CONTROL_POINTS (500)
    degree = 20
    model = _simplex(4, degree, 2)
    import math

    n_cp = math.comb(model.n_params + degree - 1, degree)
    assert n_cp > plotting_module._MAX_CONTROL_POINTS, (
        f"precondition: model must have more than {plotting_module._MAX_CONTROL_POINTS} control points; got {n_cp}"
    )
    result = plot_bezier_simplex(model, num=3)
    assert result.shape == (2, 2)
    plt.close(result[0, 0].figure)


def test_plot_bezier_simplex_pairwise_max_control_points_custom(require_matplotlib):
    """Custom max_control_points value is respected."""
    import matplotlib.pyplot as plt

    model = _simplex(4, 5, 2)
    # Set a very small cap so subsampling is always triggered
    result = plot_bezier_simplex(model, num=3, max_control_points=2)
    assert result.shape == (2, 2)
    plt.close(result[0, 0].figure)


def test_plot_bezier_simplex_pairwise_max_control_points_invalid():
    """plot_bezier_simplex must raise ValueError for negative max_control_points."""
    model = _simplex(4, 1, 2)
    with pytest.raises(ValueError, match="max_control_points"):
        plot_bezier_simplex(model, num=3, max_control_points=-1)


def test_plot_bezier_simplex_pairwise_max_pairwise_points_custom(require_matplotlib):
    """Custom max_pairwise_points value forces random sampling and returns correct shape."""
    import matplotlib.pyplot as plt

    model = _simplex(4, 1, 2)
    # With max_pairwise_points=10, even a small meshgrid will exceed the cap
    result = plot_bezier_simplex(model, num=100, max_pairwise_points=10)
    assert result.shape == (2, 2)
    plt.close(result[0, 0].figure)


def test_plot_bezier_simplex_pairwise_max_pairwise_points_invalid():
    """plot_bezier_simplex must raise ValueError for negative max_pairwise_points."""
    model = _simplex(4, 1, 2)
    with pytest.raises(ValueError, match="max_pairwise_points"):
        plot_bezier_simplex(model, num=3, max_pairwise_points=-1)


def test_plot_bezier_simplex_pairwise_max_pairwise_points_default(require_matplotlib):
    """Default max_pairwise_points matches _MAX_PAIRWISE_POINTS and returns correct shape."""
    import matplotlib.pyplot as plt

    import torch_bsf.plotting as plotting_module

    model = _simplex(4, 1, 2)
    result = plot_bezier_simplex(model, num=3, max_pairwise_points=plotting_module._MAX_PAIRWISE_POINTS)
    assert result.shape == (2, 2)
    plt.close(result[0, 0].figure)


def test_plot_curve_with_existing_axes(require_matplotlib, bezier_curve_2d):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result = plot_bezier_simplex(bezier_curve_2d, num=10, ax=ax)
    assert result is ax
    plt.close(fig)


def test_plot_triangle_with_existing_axes(require_matplotlib, bezier_triangle_2d):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result = plot_bezier_simplex(bezier_triangle_2d, num=5, ax=ax)
    assert result is ax
    plt.close(fig)


def test_plot_curve_without_control_points(require_matplotlib, bezier_curve_2d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_curve_2d, num=10, show_control_points=False)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_triangle_without_control_points(require_matplotlib, bezier_triangle_2d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_triangle_2d, num=5, show_control_points=False)
    assert ax is not None
    plt.close(ax.figure)


# ---------------------------------------------------------------------------
# ImportError handling tests (matplotlib / scipy not available)
# ---------------------------------------------------------------------------


def test_plot_bezier_curve_no_matplotlib(monkeypatch):
    """plot_bezier_simplex should raise ImportError when matplotlib is unavailable (curve)."""
    import builtins

    from torch_bsf.plotting import _plot_bezier_curve

    model = _simplex(2, 1, 2)
    real_import = builtins.__import__

    def _import_no_matplotlib(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_no_matplotlib)
    with pytest.raises(ImportError):
        _plot_bezier_curve(model, 10, None, True)


def test_plot_bezier_triangle_no_matplotlib(monkeypatch):
    """plot_bezier_simplex should raise ImportError when matplotlib is unavailable (triangle)."""
    import builtins

    from torch_bsf.plotting import _plot_bezier_triangle

    model = _simplex(3, 1, 3)
    real_import = builtins.__import__

    def _import_no_matplotlib(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_no_matplotlib)
    with pytest.raises(ImportError):
        _plot_bezier_triangle(model, 5, None, True)


def test_plot_bezier_triangle_no_scipy(monkeypatch, require_matplotlib):
    """_plot_bezier_triangle should raise ImportError when scipy is unavailable."""
    import builtins

    from torch_bsf.plotting import _plot_bezier_triangle

    model = _simplex(3, 1, 3)
    real_import = builtins.__import__

    def _import_no_scipy(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_no_scipy)
    with pytest.raises(ImportError, match="scipy"):
        _plot_bezier_triangle(model, 5, None, True)


def test_plot_pairwise_no_matplotlib(monkeypatch):
    """_plot_bezier_simplex_pairwise should raise ImportError when matplotlib is unavailable."""
    import builtins

    from torch_bsf.plotting import _plot_bezier_simplex_pairwise

    model = _simplex(4, 1, 2)
    real_import = builtins.__import__

    def _import_no_matplotlib(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _import_no_matplotlib)
    with pytest.raises(ImportError):
        _plot_bezier_simplex_pairwise(model, 5, True, 500, 2000)


# ---------------------------------------------------------------------------
# plot_bezier_simploid / plot_simploidal_bezier_space
# ---------------------------------------------------------------------------


def _triangle_and_square_space(n_values):
    """A random simploidal Bézier space on a triangle and a square glued along an edge."""
    S = SimploidalSet()
    tri = S.add_simplex(2)
    sq = S.add_simploid((1, 1))
    S.identify(tri.face(0, 2), sq.face(1, 1))
    space = SimploidalBezierSpace(S, degree=2, n_values=n_values)
    torch.nn.init.normal_(space.proper_control_points)
    return space


def _collections(ax, name):
    return [c for c in ax.collections if type(c).__name__ == name]


@pytest.mark.parametrize("n_values", [2, 3])
def test_plot_simploidal_bezier_space_mixed_degrees(require_matplotlib, n_values):
    """A space whose pieces and factors have different degrees is drawn with the right control nets."""
    import matplotlib.pyplot as plt

    S = SimploidalSet()
    tri = S.add_simplex(2)
    sq = S.add_simploid((1, 1))
    S.identify(tri.face(0, 2), sq.face(1, 1))
    # Degree 3 along the shared edge (hence for the triangle), 2 across it.
    space = SimploidalBezierSpace(S, {sq: (3, 2)}, n_values=n_values)
    assert space.simploid_degree(tri) == (3,)
    torch.nn.init.normal_(space.proper_control_points)
    ax = plot_simploidal_bezier_space(space, num=4)
    assert _n_control_points_drawn(ax) == 10 + 4 * 3
    plt.close(ax.figure)


def _n_control_points_drawn(ax):
    """The total number of red markers scattered on the axes."""
    from matplotlib.colors import to_rgba

    red = to_rgba("r")
    return sum(
        len(c.get_offsets())
        for c in _collections(ax, "PathCollection") + _collections(ax, "Path3DCollection")
        if len(c.get_facecolor()) > 0 and tuple(c.get_facecolor()[0]) == red
    )


def _n_dashed_collections(ax):
    """The number of dashed line collections (control nets) on the axes."""
    return sum(
        1
        for c in _collections(ax, "LineCollection") + _collections(ax, "Line3DCollection")
        if c.get_linestyle()[0][1] is not None
    )


@pytest.mark.parametrize("simploid_type", [(1,), (2,), (1, 1)])
@pytest.mark.parametrize("n_values", [2, 3])
def test_plot_bezier_simploid(require_matplotlib, simploid_type, n_values):
    import matplotlib.pyplot as plt

    model = BezierSimploid.randn(simploid_type, 2, n_values)
    ax = plot_bezier_simploid(model, num=5)
    assert ax.name == ("3d" if n_values == 3 else "rectilinear")
    assert _n_control_points_drawn(ax) == len(model.indices())
    assert _n_dashed_collections(ax) == 1
    if sum(simploid_type) == 1:
        assert len(ax.lines) == 1
        assert len(ax.lines[0].get_xdata()) == 6  # num + 1 grid points
    else:
        surface = "Poly3DCollection" if n_values == 3 else "PolyCollection"
        assert len(_collections(ax, surface)) == 1
    plt.close(ax.figure)


@pytest.mark.parametrize("simploid_type, n_surfaces", [((2, 1), 5), ((1, 1, 1), 6), ((3,), 4), ((), 0)])
def test_plot_bezier_simploid_other_dimensions(require_matplotlib, simploid_type, n_surfaces):
    """Simploids of dimension >= 3 are drawn through their 2-faces, and a vertex as a point."""
    import matplotlib.pyplot as plt

    model = BezierSimploid.randn(simploid_type, 1, 3)
    ax = plot_bezier_simploid(model, num=3, show_control_points=False)
    assert len(_collections(ax, "Poly3DCollection")) == n_surfaces
    if not simploid_type:
        assert len(_collections(ax, "Path3DCollection")) == 1
    plt.close(ax.figure)


def test_cells_to_draw():
    from torch_bsf.plotting import _cells_to_draw

    assert _cells_to_draw(()) == [()]
    assert _cells_to_draw((1,)) == [((0, 1),)]
    assert _cells_to_draw((2,)) == [((0, 1, 2),)]
    assert _cells_to_draw((1, 1)) == [((0, 1), (0, 1))]
    assert len(_cells_to_draw((2, 1))) == 2 + 3  # 2 triangles + 3 squares
    assert len(_cells_to_draw((1, 1, 1))) == 6
    assert len(_cells_to_draw((4,))) == 10
    assert len(_cells_to_draw((2, 2))) == 2 * 3 + 3 * 3  # triangle x vertex (twice) + edge x edge


def test_cell_triangles():
    from torch_bsf.plotting import _cell_grid, _cell_triangles

    _, coords = _cell_grid((2,), ((0, 1, 2),), 5)
    assert len(coords) == 21
    assert _cell_triangles(coords, 5, True).shape == (25, 3)
    _, coords = _cell_grid((1, 1, 1), ((0, 1), (1,), (0, 1)), 5)
    assert len(coords) == 36
    assert _cell_triangles(coords, 5, False).shape == (50, 3)


def test_plot_bezier_simploid_surface_matches_model(require_matplotlib):
    """The triangles drawn in 2D have their vertices on the model."""
    import matplotlib.pyplot as plt
    import numpy as np

    from torch_bsf.sampling import simplex_grid

    model = BezierSimploid.randn((1, 1), 2, 2)
    ax = plot_bezier_simploid(model, num=4, show_control_points=False)
    (poly,) = _collections(ax, "PolyCollection")
    assert len(poly.get_paths()) == 2 * 4 * 4
    vertices = np.concatenate([p.vertices for p in poly.get_paths()])
    g = simplex_grid(2, 4)
    params = torch.cat([g.repeat_interleave(len(g), dim=0), g.repeat(len(g), 1)], dim=1)
    with torch.no_grad():
        xs = model(params).numpy()
    dist = np.linalg.norm(vertices[:, None, :] - xs[None, :, :], axis=2).min(axis=1)
    assert dist.max() < 1e-5
    plt.close(ax.figure)


def test_control_net_edges():
    from torch_bsf.plotting import _control_net_edges

    # Bilinear patch: the 4 sides of the square, no diagonal.
    edges = _control_net_edges(simploid_indices((1, 1), 1))
    assert sorted(map(tuple, edges.tolist())) == [(0, 1), (0, 2), (1, 3), (2, 3)]
    # Quadratic triangle: the 9 edges of the triangular net.
    assert len(_control_net_edges(simploid_indices((2,), 2))) == 9
    # Biquadratic patch: a 3 x 3 grid has 12 edges.
    assert len(_control_net_edges(simploid_indices((1, 1), 2))) == 12
    # Vertex: no edge.
    assert _control_net_edges(simploid_indices((), 2)).shape == (0, 2)


@pytest.mark.parametrize("n_values", [2, 3])
def test_plot_simploidal_bezier_space(require_matplotlib, n_values):
    """A triangle and a square glued along an edge are drawn on one given axes."""
    import matplotlib.pyplot as plt

    space = _triangle_and_square_space(n_values)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d" if n_values == 3 else None)
    assert plot_simploidal_bezier_space(space, num=5, ax=ax) is ax
    surfaces = _collections(ax, "Poly3DCollection" if n_values == 3 else "PolyCollection")
    assert len(surfaces) == 2
    # Each piece has its own color.
    assert tuple(surfaces[0].get_facecolor()[0][:3]) != tuple(surfaces[1].get_facecolor()[0][:3])
    # The control nets of both pieces (6 + 9 points) are drawn.
    assert _n_control_points_drawn(ax) == 15
    assert _n_dashed_collections(ax) == 2
    plt.close(fig)


def test_plot_simploidal_bezier_space_shared_edge(require_matplotlib):
    """The boundary curves of both pieces pass through the image of the shared edge."""
    import matplotlib.pyplot as plt
    import numpy as np

    space = _triangle_and_square_space(2)
    ax = plot_simploidal_bezier_space(space, num=4, show_control_points=False)
    boundaries = _collections(ax, "LineCollection")
    assert len(boundaries) == 2
    tri = space.simploidal_set.main_simploids()[0]
    assert tri.simploid_type == (2,)
    w = torch.linspace(0, 1, 5)[:, None]
    with torch.no_grad():
        edge = space.evaluate(tri, torch.cat([1 - w, w, torch.zeros(5, 1)], dim=1)).numpy()
    for lc in boundaries:
        points = np.concatenate(lc.get_segments())
        dist = np.linalg.norm(edge[:, None, :] - points[None, :, :], axis=2).min(axis=1)
        assert dist.max() < 1e-5
    plt.close(ax.figure)


@pytest.mark.parametrize("n_values", [2, 3])
def test_plot_simploid_without_control_points(require_matplotlib, n_values):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simploid(BezierSimploid.randn((2,), 2, n_values), num=4, show_control_points=False)
    assert _n_control_points_drawn(ax) == 0
    assert _n_dashed_collections(ax) == 0
    plt.close(ax.figure)
    ax = plot_simploidal_bezier_space(_triangle_and_square_space(n_values), num=4, show_control_points=False)
    assert _n_control_points_drawn(ax) == 0
    assert _n_dashed_collections(ax) == 0
    plt.close(ax.figure)


def test_plot_simploid_kwargs(require_matplotlib):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    ax = plot_bezier_simploid(BezierSimploid.randn((1,), 2, 2), num=4, color="g", linewidth=3)
    assert ax.lines[0].get_color() == "g"
    assert ax.lines[0].get_linewidth() == 3
    plt.close(ax.figure)
    ax = plot_simploidal_bezier_space(_triangle_and_square_space(2), num=4, color="g", show_control_points=False)
    polys = _collections(ax, "PolyCollection")
    assert len(polys) == 2
    assert all(tuple(p.get_facecolor()[0][:3]) == to_rgba("g")[:3] for p in polys)
    plt.close(ax.figure)


@pytest.mark.parametrize(
    "kwargs", [{"facecolors": "y", "edgecolors": "k"}, {"fc": "y", "ec": "k"}, {"facecolor": "y", "edgecolor": "k"}]
)
def test_plot_simploid_2d_surface_color_aliases(require_matplotlib, kwargs):
    """Every PolyCollection alias of facecolor / edgecolor overrides the defaults without TypeError."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    ax = plot_bezier_simploid(BezierSimploid.randn((2,), 2, 2), num=4, show_control_points=False, **kwargs)
    (poly,) = _collections(ax, "PolyCollection")
    assert tuple(poly.get_facecolor()[0][:3]) == to_rgba("y")[:3]
    assert tuple(poly.get_edgecolor()[0][:3]) == to_rgba("k")[:3]
    plt.close(ax.figure)


@pytest.mark.parametrize("alias", ["facecolor", "facecolors", "fc"])
def test_plot_simploid_3d_surface_face_color_aliases(require_matplotlib, alias):
    """Face colors given under any alias are honored by 3D surfaces without TypeError."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    ax = plot_bezier_simploid(BezierSimploid.randn((1, 1), 2, 3), num=4, show_control_points=False, **{alias: "y"})
    (surface,) = _collections(ax, "Poly3DCollection")
    face_rgb = surface.get_facecolor()[:, :3]
    # Shading changes the brightness, not the hue: no blue channel for yellow.
    assert (face_rgb[:, 2] < 0.05).all() and (face_rgb[:, 0] > 0).all()
    assert to_rgba("y")[2] == 0
    plt.close(ax.figure)


@pytest.mark.parametrize("alias", ["facecolor", "facecolors", "fc"])
def test_plot_simploid_3d_surface_per_triangle_face_colors(require_matplotlib, alias):
    """A sequence of per-triangle face colors is applied to 3D surfaces without ValueError."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    num = 4
    n_triangles = 2 * num * num  # a square cell grid split into triangles
    colors = ["r", "b"] * (n_triangles // 2)
    ax = plot_bezier_simploid(BezierSimploid.randn((1, 1), 2, 3), num=num, show_control_points=False, **{alias: colors})
    (surface,) = _collections(ax, "Poly3DCollection")
    face = surface.get_facecolor()
    assert len(face) == n_triangles
    # The faces are depth-sorted when drawn, so compare the multiset of colors.
    reds = sum(1 for rgba in face if tuple(rgba[:3]) == to_rgba("r")[:3])
    blues = sum(1 for rgba in face if tuple(rgba[:3]) == to_rgba("b")[:3])
    assert reds == blues == n_triangles // 2
    plt.close(ax.figure)


def test_plot_simploid_curve_c_alias(require_matplotlib):
    """``c`` (the Line2D alias of ``color``) sets the curve color without TypeError."""
    import matplotlib.pyplot as plt

    ax = plot_bezier_simploid(BezierSimploid.randn((1,), 2, 2), num=4, c="m", show_control_points=False)
    assert ax.lines[0].get_color() == "m"
    plt.close(ax.figure)
    ax = plot_simploidal_bezier_space(_edge_space(), num=4, c="m", show_control_points=False)
    assert [line.get_color() for line in ax.lines] == ["m"]
    plt.close(ax.figure)


def _edge_space():
    S = SimploidalSet()
    S.add_simplex(1)
    space = SimploidalBezierSpace(S, degree=2, n_values=2)
    torch.nn.init.normal_(space.proper_control_points)
    return space


def test_plot_simploid_2d_surface_kwargs_override_defaults(require_matplotlib):
    """PolyCollection options such as facecolor / edgecolor can be passed through **kwargs."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    model = BezierSimploid.randn((1, 1), 2, 2)
    ax = plot_bezier_simploid(model, num=4, facecolor="y", edgecolor="k", show_control_points=False)
    (poly,) = _collections(ax, "PolyCollection")
    assert tuple(poly.get_facecolor()[0][:3]) == to_rgba("y")[:3]
    assert tuple(poly.get_edgecolor()[0][:3]) == to_rgba("k")[:3]
    plt.close(ax.figure)


@pytest.mark.parametrize("n_values", [0, 1, 4])
def test_plot_simploid_invalid_n_values(n_values):
    with pytest.raises(ValueError, match="n_values of 2 or 3"):
        plot_bezier_simploid(BezierSimploid.zeros((1, 1), 1, n_values))
    with pytest.raises(ValueError, match="n_values of 2 or 3"):
        plot_simploidal_bezier_space(_triangle_and_square_space(n_values))


@pytest.mark.parametrize("num", [0, -1, 2.5, True])
def test_plot_simploid_invalid_num(num):
    with pytest.raises(ValueError, match="num"):
        plot_bezier_simploid(BezierSimploid.zeros((1, 1), 1, 2), num=num)
    with pytest.raises(ValueError, match="num"):
        plot_simploidal_bezier_space(_triangle_and_square_space(3), num=num)


def test_plot_simploid_axes_mismatch(require_matplotlib):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax2d = fig.add_subplot(121)
    ax3d = fig.add_subplot(122, projection="3d")
    with pytest.raises(ValueError, match="3D axes"):
        plot_bezier_simploid(BezierSimploid.zeros((2,), 1, 3), ax=ax2d)
    with pytest.raises(ValueError, match="3D axes"):
        plot_simploidal_bezier_space(_triangle_and_square_space(2), ax=ax3d)
    plt.close(fig)


def test_plot_simploid_no_matplotlib(monkeypatch):
    """The simploid plotting functions raise ImportError without matplotlib."""
    import builtins

    real_import = builtins.__import__

    def _import_no_matplotlib(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, globals, locals, fromlist, level)

    space = _triangle_and_square_space(2)
    monkeypatch.setattr(builtins, "__import__", _import_no_matplotlib)
    with pytest.raises(ImportError, match="matplotlib"):
        plot_bezier_simploid(BezierSimploid.zeros((1,), 1, 2))
    with pytest.raises(ImportError, match="matplotlib"):
        plot_simploidal_bezier_space(space)
