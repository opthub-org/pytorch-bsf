"""Hypothesis strategies shared by the property-based tests.

The strategies generate the inputs on which numerical code tends to break, not only typical ones:

* :func:`log_uniform` and the strategies built on it spread magnitudes over many orders, so that
  translations (:func:`translations`, up to ``1e12``) and scalings (:func:`scalings`, ``1e-6`` to ``1e6``)
  are exercised at every scale rather than mostly near the largest value;
* :func:`extreme_scalings` multiplies data of order one by powers of two up to the float64 limits, which
  is exact, so that a property can compare with the unscaled result without a rounding allowance, and
  :func:`axis_scalings` scales each axis independently (anisotropic scaling);
* :func:`split_fractions` includes split ratios ``s`` within ``1e-12`` of ``0`` and ``1``;
* :func:`simplex_points` includes vertices, points on faces and coordinates that are tiny but not zero.

They are written for reuse: the meshes of :func:`kuhn_meshes` and the points of :func:`points_in_cells`
take only ``(vertices, simplices)``, so that properties of other mesh constructions can use them too.
"""

import itertools
import math
from collections.abc import Sequence

import torch
from hypothesis import strategies as st

#: Unit roundoff of float64, :math:`u = 2^{-53}`. A correctly rounded operation has a relative error at most ``U``.
U = 2.0**-53

#: The largest finite float64.
FLOAT64_MAX = torch.finfo(torch.float64).max

#: The smallest positive normal float64.
FLOAT64_TINY = torch.finfo(torch.float64).tiny


def seeds() -> st.SearchStrategy[int]:
    """Seeds accepted by both NumPy and PyTorch local generators, including zero."""
    return st.integers(0, 2**32 - 1)


@st.composite
def feature_values(draw: st.DrawFn, *, min_rows: int = 1) -> torch.Tensor:
    """Small float32/float64 tables, including constant columns and differing feature scales.

    Dyadic entries and powers of two keep the inputs exact. Exponents are bounded by 30
    (float32) or 100 (float64) so variance computations remain representable.
    """
    dtype = draw(st.sampled_from([torch.float32, torch.float64]))
    max_exponent = 30 if dtype == torch.float32 else 100
    n_rows, n_columns = draw(st.integers(min_rows, 12)), draw(st.integers(1, 4))
    columns = []
    for _ in range(n_columns):
        constant = draw(st.booleans())
        entries = draw(st.lists(st.integers(-32, 32), min_size=1 if constant else n_rows, max_size=1 if constant else n_rows))
        column = torch.tensor(entries, dtype=dtype).expand(n_rows)
        columns.append(column * draw(extreme_scalings(max_exponent)))
    return torch.stack(columns, dim=1)


def log_uniform(min_value: float, max_value: float) -> st.SearchStrategy[float]:
    """Positive floats in ``[min_value, max_value]`` spread evenly over binary orders of magnitude.

    The bounds themselves are always among the candidates, and Hypothesis tries them early.
    """
    if not 0 < min_value <= max_value <= FLOAT64_MAX:
        raise ValueError("bounds must satisfy 0 < min_value <= max_value <= FLOAT64_MAX.")
    lo, hi = math.frexp(min_value)[1], math.frexp(max_value)[1]
    spread = st.builds(math.ldexp, st.floats(0.5, 1.0, exclude_max=True), st.integers(lo, hi))
    return st.sampled_from([min_value, max_value]) | spread.filter(lambda v: min_value <= v <= max_value)


def signed(magnitudes: st.SearchStrategy[float]) -> st.SearchStrategy[float]:
    """Floats whose absolute values are drawn from ``magnitudes``, of either sign."""
    return st.tuples(magnitudes, st.booleans()).map(lambda p: -p[0] if p[1] else p[0])


def translations(dimension: int, *, max_magnitude: float = 1e12) -> st.SearchStrategy[torch.Tensor]:
    """Float64 translation vectors whose components are ``0`` or of magnitude ``1e-6`` to ``max_magnitude``."""
    component = st.just(0.0) | signed(log_uniform(1e-6, max_magnitude))
    return st.lists(component, min_size=dimension, max_size=dimension).map(lambda v: torch.tensor(v, dtype=torch.float64))


def scalings(min_value: float = 1e-6, max_value: float = 1e6) -> st.SearchStrategy[float]:
    """Positive scale factors between ``min_value`` and ``max_value``, not necessarily powers of two."""
    return log_uniform(min_value, max_value)


def extreme_scalings(max_exponent: int = 1000) -> st.SearchStrategy[float]:
    r"""Powers of two :math:`2^k`, :math:`|k| \le` ``max_exponent``, whose product with data of order one is exact.

    The default reaches about ``1e301`` and ``1e-301``, near the float64 limits (``1.8e308`` and the
    smallest normal ``2.2e-308``) but with room for the sizes of a few simplices or coefficients.
    """
    edges = st.sampled_from([-max_exponent, max_exponent])
    return (edges | st.integers(-max_exponent, max_exponent)).map(lambda k: math.ldexp(1.0, k))


def axis_scalings(dimension: int, scale: st.SearchStrategy[float] | None = None) -> st.SearchStrategy[torch.Tensor]:
    """Float64 vectors of one scale factor per axis, drawn independently from ``scale`` (default :func:`extreme_scalings`).

    Axes then differ in scale by up to ``2**2000`` by default, as when coordinates mix units, e.g. ``1e200`` and
    ``1e-200``. With the default, the scaled coordinates of data of order one are exact.
    """
    factor = extreme_scalings() if scale is None else scale
    return st.lists(factor, min_size=dimension, max_size=dimension).map(lambda v: torch.tensor(v, dtype=torch.float64))


def split_fractions(min_value: float = 1e-12) -> st.SearchStrategy[float]:
    """Split ratios ``s`` in the open interval ``(0, 1)``, including values within ``min_value`` of ``0`` and ``1``."""
    near_zero = log_uniform(min_value, 0.5)
    fractions = st.one_of(
        st.sampled_from([0.5, min_value, 1.0 - min_value]),
        st.floats(0.0, 1.0, exclude_min=True, exclude_max=True).filter(lambda s: min_value <= s <= 1 - min_value),
        near_zero,
        near_zero.map(lambda s: 1.0 - s),
    )
    # 1 - s rounds to 1 for s below 2**-54, e.g. when min_value is the smallest normal float64.
    return fractions.filter(lambda s: 0.0 < s < 1.0)


def simploid_types(
    *, min_factors: int = 1, max_factors: int = 3, max_dimension: int = 3, max_width: int = 7
) -> st.SearchStrategy[tuple[int, ...]]:
    """Simploid types ``(a_1, ..., a_n)``, with at most ``max_width`` barycentric coordinates in total."""
    factors = st.lists(st.integers(1, max_dimension), min_size=min_factors, max_size=max_factors)
    return factors.map(tuple).filter(lambda t: sum(a + 1 for a in t) <= max_width)


def degrees(simploid_type: Sequence[int], *, min_degree: int = 0, max_degree: int = 3) -> st.SearchStrategy[tuple[int, ...]]:
    """One degree per factor of ``simploid_type``."""
    return st.tuples(*(st.integers(min_degree, max_degree) for _ in simploid_type))


@st.composite
def factor_edges(draw: st.DrawFn, simploid_type: Sequence[int]) -> tuple[int, int, int]:
    """An edge ``(factor, i, j)`` of a factor of ``simploid_type``, with ``i != j`` in either order."""
    factor = draw(st.integers(0, len(simploid_type) - 1))
    i, j = draw(st.permutations(range(simploid_type[factor] + 1)))[:2]
    return factor, i, j


def _coordinate() -> st.SearchStrategy[float]:
    """Unnormalized nonnegative weights: zero, tiny but positive, or of order one."""
    return st.one_of(st.just(0.0), log_uniform(FLOAT64_TINY, 1e-12), st.floats(0.0, 1.0))


@st.composite
def simplex_points(draw: st.DrawFn, n_vertices: int, n_points: int) -> torch.Tensor:
    """Float64 barycentric coordinates of shape ``(n_points, n_vertices)``.

    Vertices, faces (zero coordinates) and points just off a face (coordinates down to the smallest normal
    float64) are all generated. Each row is normalized in float64, so it sums to one up to rounding.
    """
    row = st.lists(_coordinate(), min_size=n_vertices, max_size=n_vertices)
    rows = draw(st.lists(row, min_size=n_points, max_size=n_points))
    t = torch.tensor(rows, dtype=torch.float64).reshape(n_points, n_vertices)
    empty = t.sum(dim=1) == 0
    t[empty, 0] = 1.0
    return t / t.sum(dim=1, keepdim=True)


@st.composite
def simploid_points(draw: st.DrawFn, simploid_type: Sequence[int], n_points: int) -> torch.Tensor:
    """Float64 parameters of shape ``(n_points, sum(a_i + 1))``: the barycentric coordinates of every factor."""
    blocks = [draw(simplex_points(a + 1, n_points)) for a in simploid_type]
    return torch.cat([torch.empty(n_points, 0, dtype=torch.float64), *blocks], dim=1)


@st.composite
def matrices(draw: st.DrawFn, n_rows: int, n_columns: int, *, scale: st.SearchStrategy[float] | None = None) -> torch.Tensor:
    """Float64 matrices whose entries lie in ``[-c, c]`` for a scale ``c`` drawn from ``scale`` (default ``1``)."""
    entries = draw(st.lists(st.floats(-1.0, 1.0), min_size=n_rows * n_columns, max_size=n_rows * n_columns))
    c = 1.0 if scale is None else draw(scale)
    return torch.tensor(entries, dtype=torch.float64).reshape(n_rows, n_columns) * c


def _kuhn_simplices(dimension: int, n: int) -> list[list[int]]:
    """The Kuhn triangulation of an ``n``-per-axis grid: ``dimension!`` simplices per grid cube."""
    number = {p: k for k, p in enumerate(itertools.product(range(n + 1), repeat=dimension))}
    simplices = []
    for corner in itertools.product(range(n), repeat=dimension):
        for perm in itertools.permutations(range(dimension)):
            p = list(corner)
            simplex = [number[tuple(p)]]
            for axis in perm:
                p[axis] += 1
                simplex.append(number[tuple(p)])
            simplices.append(simplex)
    return simplices


@st.composite
def kuhn_meshes(
    draw: st.DrawFn, *, dimensions: Sequence[int] = (1, 2, 3), max_n: int = 2, quantum: float | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Conforming meshes ``(vertices, simplices)`` of the unit cube with jittered interior vertices.

    The grid has ``n <= max_n`` cells per axis. Each interior vertex moves by at most ``0.15 / n`` per axis,
    which keeps every simplex non-degenerate with a relative volume of order one. If ``quantum`` is given,
    the jitter is a multiple of it, so that vertex coordinates are exact dyadic numbers when ``quantum`` and
    ``1 / n`` are powers of two. The vertex order of each simplex is shuffled.
    """
    dimension = draw(st.sampled_from(list(dimensions)))
    n = draw(st.integers(1, max_n))
    grid = list(itertools.product(range(n + 1), repeat=dimension))
    x = torch.tensor(grid, dtype=torch.float64).reshape(len(grid), dimension) / n
    interior = ((x > 0) & (x < 1)).all(dim=1)
    n_interior = int(interior.sum())
    jitter = draw(st.lists(st.floats(-0.15, 0.15), min_size=n_interior * dimension, max_size=n_interior * dimension))
    shift = torch.tensor(jitter, dtype=torch.float64).reshape(n_interior, dimension) / n
    if quantum is not None:
        shift = torch.round(shift / quantum) * quantum
    x[interior] += shift
    simplices = [draw(st.permutations(s)) for s in _kuhn_simplices(dimension, n)]
    return x, torch.tensor(simplices, dtype=torch.long)


@st.composite
def points_in_cells(
    draw: st.DrawFn, vertices: torch.Tensor, simplices: torch.Tensor, n_points: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Points in the simplices of a mesh: ``(simplex numbers, barycentric coordinates, positions)``.

    The barycentric coordinates refer to the vertices of each simplex in ascending vertex order, as in
    :attr:`torch_bsf.interpolation.SimplicialMesh.simplices`.
    """
    numbers = draw(st.lists(st.integers(0, len(simplices) - 1), min_size=n_points, max_size=n_points))
    rows = torch.tensor(numbers, dtype=torch.long)
    lam = draw(simplex_points(simplices.shape[1], n_points))
    corners = vertices[torch.sort(simplices, dim=1).values[rows]]
    return rows, lam, torch.einsum("nk,nkd->nd", lam, corners)
