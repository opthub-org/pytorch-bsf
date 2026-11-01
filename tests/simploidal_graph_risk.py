"""Exact checks of the closed risk formulas for arbitrary one-dimensional domains."""

from fractions import Fraction
from math import comb, factorial

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.sampling import _exact_solve
from tests.simploidal_risk import _assembled_grams, _metric, _population_system, _traces
from torch_bsf import BezierSpace, SimploidalSet


def _graph_space(n_vertices, edges, degrees):
    domain = SimploidalSet()
    vertices = [domain.add_simplex(0) for _ in range(n_vertices)]
    handles = []
    for a, b in edges:
        edge = domain.add_simplex(1)
        domain.identify(edge.face(0, 0), vertices[a])
        domain.identify(edge.face(0, 1), vertices[b])
        handles.append(edge)
    assignments = dict(zip(handles, degrees, strict=True))
    return BezierSpace.on_set(domain, degree=assignments, default_degree=1)


def _graph_traces(space, weights):
    """The claimed scalar formula, indexed by the compiled cell IDs."""
    result = {}
    vertices = {}
    for cell_id, cell in enumerate(space.cells):
        if not cell.dimensions:
            vertices[cell.controls[0]] = cell_id
            result[cell_id] = weights[cell_id]
    for cell_id, cell in enumerate(space.cells):
        if cell.dimensions:
            degree = cell.degree[0]
            a, b = (vertices[cell.controls[j]] for j in (0, -1))
            if a == b:
                result[a] += weights[cell_id] * Fraction(2, (degree + 1) * (degree + 1 + (-1) ** degree))
            else:
                for vertex in (a, b):
                    result[vertex] += weights[cell_id] * Fraction(1, degree * (degree + 2))
            if degree > 1:
                result[cell_id] = weights[cell_id] * (degree - 1)
    return result


@pytest.mark.parametrize("degree", range(1, 13))
def test_graph_endpoint_inner_products_from_exact_bernstein_gram(degree):
    # Eliminate every interior mode in the Bernstein Gram matrix. This checks the
    # cross term independently of the Legendre proof, including both parities.
    gram = [
        [
            Fraction(
                comb(degree, i) * comb(degree, j) * factorial(i + j) * factorial(2 * degree - i - j),
                factorial(2 * degree + 1),
            )
            for j in range(degree + 1)
        ]
        for i in range(degree + 1)
    ]
    endpoints = (0, degree)
    residual = [[gram[i][j] for j in endpoints] for i in endpoints]
    if degree > 1:
        interior = range(1, degree)
        projection = _exact_solve(
            [[gram[i][j] for j in interior] for i in interior],
            [[gram[i][j] for j in endpoints] for i in interior],
        )
        for i, endpoint in enumerate(endpoints):
            for j in range(2):
                residual[i][j] -= sum(gram[endpoint][k] * projection[k - 1][j] for k in interior)
    norm = Fraction(1, degree * (degree + 2))
    cross = Fraction((-1) ** (degree - 1), degree * (degree + 1) * (degree + 2))
    assert residual == [[norm, cross], [cross, norm]]
    loop = sum(map(sum, residual))
    assert loop == Fraction(2, (degree + 1) * (degree + 1 + (-1) ** degree))
    # The distinct reciprocal values 1, 6, 15, 28, ... are the hexagonal numbers.
    k = degree // 2 + 1
    assert loop == Fraction(1, k * (2 * k - 1))


def _endpoint_polynomial(degree):
    # Rodrigues' power coefficients for shifted Legendre polynomials, followed
    # by differentiation. Integrating the expanded product below is exact.
    result = [Fraction(0)] * (degree + 1)
    for n, denominator in ((degree, 2 * degree * (degree + 1)), (degree + 1, 2 * (degree + 1) * (degree + 2))):
        for k in range(1, n + 1):
            result[k - 1] += Fraction((-1) ** (n - k) * comb(n, k) * comb(n + k, k) * k, denominator)
    return result


@pytest.mark.parametrize("degree", (32, 33))
def test_graph_endpoint_inner_products_by_high_degree_exact_polynomial_integration(degree):
    right = _endpoint_polynomial(degree)
    left = [sum(right[k] * comb(k, j) * (-1) ** j for k in range(j, degree + 1)) for j in range(degree + 1)]
    norm = sum(a * b / (i + j + 1) for i, a in enumerate(right) for j, b in enumerate(right))
    cross = sum(a * b / (i + j + 1) for i, a in enumerate(left) for j, b in enumerate(right))
    assert norm == Fraction(1, degree * (degree + 2))
    assert cross == Fraction((-1) ** (degree - 1), degree * (degree + 1) * (degree + 2))


_GRAPHS = [
    (1, (), ()),
    (4, ((0, 1), (1, 2)), (2, 4)),
    (3, ((0, 1), (0, 1), (1, 1), (1, 2)), (1, 3, 4, 2)),
    (3, ((0, 0), (0, 1), (1, 2), (2, 2), (0, 2)), (3, 1, 4, 2, 3)),
] + [(1, ((0, 0),), (degree,)) for degree in range(1, 9)]


@pytest.mark.parametrize(("n_vertices", "edges", "degrees"), _GRAPHS)
@pytest.mark.parametrize("evaluation", ("all", "maximal", "vertices"))
def test_graph_closed_form_matches_assembled_exact_propagation(n_vertices, edges, degrees, evaluation):
    space = _graph_space(n_vertices, edges, degrees)
    numerators = [
        (i + 1)
        * (
            evaluation == "all"
            or (evaluation == "maximal" and i in space.main_cells)
            or (evaluation == "vertices" and not cell.dimensions)
        )
        for i, cell in enumerate(space.cells)
    ]
    weights = [Fraction(n, sum(numerators)) for n in numerators]
    grams, stages = _assembled_grams(space)
    propagation, blocks = _population_system(space, grams, stages)
    actual = dict(zip((i for i, _ in stages), _traces(_metric(grams, weights), propagation, stages, blocks), strict=True))
    assert actual == _graph_traces(space, weights)
    # Linear edges have no innovation block; their endpoints are still fitted.
    assert len(stages) == n_vertices + sum(degree > 1 for degree in degrees)
    assert space.n_controls == n_vertices + sum(degree - 1 for degree in degrees)


@pytest.mark.parametrize("degree", range(2, 9))
@pytest.mark.parametrize("edges", (((0, 1),), ((0, 1), (1, 2)), ((0, 1), (1, 2), (2, 0))))
def test_graph_grouped_coefficients_and_all_at_once_threshold(degree, edges):
    n_vertices = max(max(edge) for edge in edges) + 1
    n_edges = len(edges)
    space = _graph_space(n_vertices, edges, (degree,) * n_edges)
    # Nonuniform edge evaluation cancels from the dimension-grouped sums.
    edge_ids = [i for i, cell in enumerate(space.cells) if cell.dimensions]
    weights = [
        Fraction(edge_ids.index(i) + 1, n_edges * (n_edges + 1) // 2) if i in edge_ids else Fraction(0)
        for i in range(space.n_cells)
    ]
    grams, stages = _assembled_grams(space)
    propagation, blocks = _population_system(space, grams, stages)
    traces = _traces(_metric(grams, weights), propagation, stages, blocks)
    a = [
        sum(c for (i, _), c in zip(stages, traces, strict=True) if bool(space.cells[i].dimensions) == bool(m)) * count
        for m, count in enumerate((n_vertices, n_edges))
    ]
    capacity = degree * (degree + 2)
    assert a == [Fraction(2 * n_vertices, capacity), Fraction(n_edges * (degree - 1))]
    gap = space.n_controls - sum(a)
    # (sqrt(a0)+sqrt(a1))^2 < p iff 4*a0*a1 < (p-a0-a1)^2.
    assert gap > 0
    threshold = Fraction((capacity - 2) ** 2, 8 * capacity * (degree - 1))
    assert (4 * a[0] * a[1] < gap**2) == (Fraction(n_edges, n_vertices) < threshold)


def test_quadratic_chain_individual_optimum_exceeds_all_at_once():
    space = _graph_space(3, ((0, 1), (1, 2)), (2, 2))
    weights = [Fraction(int(i in space.main_cells), 2) for i in range(space.n_cells)]
    grams, stages = _assembled_grams(space)
    propagation, blocks = _population_system(space, grams, stages)
    traces = _traces(_metric(grams, weights), propagation, stages, blocks)
    assert sorted(traces) == [Fraction(1, 16), Fraction(1, 16), Fraction(1, 8), Fraction(1, 2), Fraction(1, 2)]
    # The individual optimum is (1/2 + 5/(2*sqrt(2)))^2 = 27/8 + 5/(2*sqrt(2)) > p=5.
    assert space.n_controls == 5
    assert Fraction(25, 8) > (space.n_controls - Fraction(27, 8)) ** 2


@pytest.mark.parametrize("degree", range(1, 15))
def test_single_loop_all_at_once_comparison_is_fully_classified(degree):
    space = _graph_space(1, ((0, 0),), (degree,))
    weights = [Fraction(int(i in space.main_cells)) for i in range(space.n_cells)]
    grams, stages = _assembled_grams(space)
    propagation, blocks = _population_system(space, grams, stages)
    traces = _traces(_metric(grams, weights), propagation, stages, blocks)
    vertex = next(c for (i, _), c in zip(stages, traces, strict=True) if not space.cells[i].dimensions)
    edge = sum(c for (i, _), c in zip(stages, traces, strict=True) if space.cells[i].dimensions)
    assert edge == degree - 1
    assert space.n_controls == degree
    gap = degree - vertex - edge
    assert gap >= 0
    discriminant = gap**2 - 4 * vertex * edge
    assert (discriminant == 0) == (degree == 1)
    assert (discriminant < 0) == (degree in (3, 5))
    assert (discriminant > 0) == (degree == 2 or degree == 4 or degree >= 6)


@pytest.mark.parametrize(("n_vertices", "edges", "degrees"), _GRAPHS)
@pytest.mark.parametrize("evaluation", ("all", "maximal", "vertices"))
def test_public_graph_coefficients_and_auto_allocation_use_closed_form(n_vertices, edges, degrees, evaluation):
    from unittest.mock import patch

    from torch_bsf.sampling import optimal_skeleton_ratio, skeleton_risk_coefficients

    space = _graph_space(n_vertices, edges, degrees)
    raw = [
        (i + 1)
        * (
            evaluation == "all"
            or (evaluation == "maximal" and i in space.main_cells)
            or (evaluation == "vertices" and not c.dimensions)
        )
        for i, c in enumerate(space.cells)
    ]
    weights = [Fraction(n, sum(raw)) for n in raw]
    evaluation_weights = {i: float(w) for i, w in enumerate(weights)}
    reference = _graph_traces(space, weights)
    # Exact integer denominators, one division/product per contribution, then positive summation.
    tolerance = 16 * (space.n_cells + 2) * 2**-53
    with patch("torch_bsf.sampling._assembled_risk_system", side_effect=AssertionError("no dense graph system")):
        actual = skeleton_risk_coefficients(space, evaluation_weights=evaluation_weights, allocation="cell")
        assert actual == pytest.approx({cid: float(c) for cid, c in reference.items()}, rel=tolerance, abs=0)
        groups: dict[int, list[Fraction]] = {}
        for cid, c in reference.items():
            groups.setdefault(sum(space.cells[cid].dimensions), []).append(c)
        if any(not sum(values) for values in groups.values()):
            with pytest.raises(ValueError, match="zero-risk active groups"):
                optimal_skeleton_ratio(space, evaluation_weights=evaluation_weights)
        else:
            plan = optimal_skeleton_ratio(space, evaluation_weights=evaluation_weights)
            assert plan.strategy == ("cell" if all(reference.values()) else "dimension")
            assert sum(plan.cell_weights.values()) == pytest.approx(1, rel=tolerance, abs=0)


def test_graph_closed_form_exceeds_dense_degree_and_global_control_limits():
    from torch_bsf.sampling import optimal_skeleton_ratio, sample_skeleton, skeleton_risk_coefficients

    space = _graph_space(260, tuple((i, i + 1) for i in range(259)), (2,) * 259)
    assert space.n_controls > 512
    assert skeleton_risk_coefficients(space) == pytest.approx((65, 259), rel=16 * space.n_cells * 2**-53, abs=0)
    plan = optimal_skeleton_ratio(space)
    assert plan.strategy == "cell" and len(plan.cell_weights) == space.n_controls
    assert len(sample_skeleton(space, 37, allocation=plan, seed=0)) == 37
    loop = _graph_space(1, ((0, 0),), (100,))
    coefficients = skeleton_risk_coefficients(loop)
    assert coefficients == pytest.approx((2 / (101 * 102), 99), rel=16 * 2**-53, abs=0)


@given(degrees=st.tuples(*(st.integers(1, 6) for _ in range(4))), data=st.data())
def test_closed_graph_risk_and_allocation_property(degrees, data):
    from torch_bsf.sampling import optimal_skeleton_ratio, skeleton_risk_coefficients

    space = _graph_space(3, ((0, 0), (0, 1), (1, 2), (2, 2)), degrees)
    raw = data.draw(st.lists(st.integers(0, 9), min_size=space.n_cells, max_size=space.n_cells))
    if not any(raw):
        raw[0] = 1
    weights = [Fraction(w, sum(raw)) for w in raw]
    expected = _graph_traces(space, weights)
    evaluation = {i: float(w) for i, w in enumerate(weights)}
    actual = skeleton_risk_coefficients(space, evaluation_weights=evaluation, allocation="cell")
    # Four positive arithmetic/normalization paths, with at most ncells terms.
    tolerance = 16 * (space.n_cells + 2) * 2**-53
    assert actual == pytest.approx({cid: float(c) for cid, c in expected.items()}, rel=tolerance, abs=0)
    groups = [[cid for cid in expected if sum(space.cells[cid].dimensions) == m] for m in range(2)]
    if any(group and not sum(expected[cid] for cid in group) for group in groups):
        with pytest.raises(ValueError, match="zero-risk active groups"):
            optimal_skeleton_ratio(space, evaluation_weights=evaluation)
    else:
        plan = optimal_skeleton_ratio(space, evaluation_weights=evaluation)
        assert plan.strategy == ("cell" if all(expected.values()) else "dimension")
        assert all(q > 0 for q in plan.cell_weights.values())
        if plan.strategy == "cell":
            risk = sum(float(c) / plan.cell_weights[cid] for cid, c in expected.items())
            assert risk == pytest.approx(sum(float(c) ** 0.5 for c in expected.values()) ** 2, rel=tolerance, abs=0)
        else:
            assert plan.fallback_reason == "zero_cell_coefficients"
            for group in groups:
                assert len({plan.cell_weights[cid] for cid in group}) <= 1
