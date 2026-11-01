"""Independent exact checks of the closed incidence and corner-collision formulas."""

import math
from fractions import Fraction
from itertools import product

import pytest

from tests.sampling import _scalar_risk_sum
from tests.simploidal_risk import _assembled_grams, _metric, _population_system, _space, _traces
from torch_bsf import BezierSpace, SimploidalSet
from torch_bsf._kernels.combinatorics import _indices


def _regular_space(kind, degree):
    if kind not in ("tetrahedra", "nonpure", "nonmanifold"):
        return _space(kind, degree)
    domain = SimploidalSet()
    if kind == "tetrahedra":
        left, right = domain.add_simplex(3), domain.add_simplex(3)
        domain.identify(left.face(0, 0), right.face(0, 1))
    elif kind == "nonpure":
        triangle, segment = domain.add_simplex(2), domain.add_simplex(1)
        domain.identify(triangle.face(0, 0).face(0, 0), segment.face(0, 0))
    else:
        triangles = [domain.add_simplex(2) for _ in range(3)]
        for triangle in triangles[1:]:
            domain.identify(triangles[0].face(0, 0), triangle.face(0, 0))
    return BezierSpace.on_set(domain, degree=degree)


def _weights(space, mode):
    raw = [i + 1 if mode == "all" or i in space.main_cells else 0 for i in range(space.n_cells)]
    return [Fraction(w, sum(raw)) for w in raw]


def _incidence_traces(space, stages, weights):
    """Use only face ownership, scalar simplex sums and product factorization."""
    answer = []
    for _, proper in stages:
        trace = Fraction(0)
        for chart, weight in zip(space.cells, weights, strict=True):
            # In a regular chart every local control has a distinct global ID.
            assert len(set(chart.controls)) == len(chart.controls)
            if not weight or proper[0] not in chart.controls:
                continue
            local = list(_indices(chart.dimensions, chart.degree))
            index = local[chart.controls.index(proper[0])]
            shapes = [sum(a > 0 for a in alpha) for alpha in index]
            # All proper source controls must belong to this same unique local face.
            for control in proper:
                support = local[chart.controls.index(control)]
                assert [tuple(i for i, a in enumerate(alpha) if a) for alpha in support] == [
                    tuple(i for i, a in enumerate(alpha) if a) for alpha in index
                ]
            trace += weight * math.prod(
                _scalar_risk_sum(dimension + 1, degree, size) / math.comb(dimension + 1, size) ** 2
                for dimension, degree, size in zip(chart.dimensions, chart.degree, shapes, strict=True)
            )
        answer.append(trace)
    return answer


@pytest.mark.parametrize(
    ("kind", "degree"),
    [
        ("chain", 4),
        ("disconnected", 3),
        ("triangles", 3),
        ("mixed", 3),
        ("tetrahedra", 3),
        ("nonpure", 2),
        ("nonmanifold", 3),
    ],
)
@pytest.mark.parametrize("mode", ["all", "maximal"])
def test_regular_incidence_matches_independent_assembled_population_solve(kind, degree, mode):
    space = _regular_space(kind, degree)
    grams, stages = _assembled_grams(space)
    propagation, blocks = _population_system(space, grams, stages)
    weights = _weights(space, mode)
    assert _incidence_traces(space, stages, weights) == _traces(_metric(grams, weights), propagation, stages, blocks)


@pytest.mark.parametrize(("kind", "degree"), [("triangles", 1), ("triangles", 2), ("nonmanifold", 3), ("tetrahedra", 2)])
def test_pure_simplicial_grouped_coefficients_depend_only_on_face_counts(kind, degree):
    space = _regular_space(kind, degree)
    dimension = sum(space.cells[space.main_cells[0]].dimensions)
    grams, stages = _assembled_grams(space)
    propagation, blocks = _population_system(space, grams, stages)
    # Unequal top-chart weights test the claimed independence from their distribution.
    traces = _traces(_metric(grams, _weights(space, "maximal")), propagation, stages, blocks)
    for m in range(dimension + 1):
        faces = [cell for cell in space.cells if sum(cell.dimensions) == m]
        actual = len(faces) * sum(
            c for (cell_id, _), c in zip(stages, traces, strict=True) if sum(space.cells[cell_id].dimensions) == m
        )
        expected = (
            len(faces) * _scalar_risk_sum(dimension + 1, degree, m + 1) / math.comb(dimension + 1, m + 1) if m < degree else 0
        )
        assert actual == expected
        if dimension == degree == 2:
            assert actual == (Fraction(len(faces), 8) if m == 0 else len(faces) if m == 1 else 0)


def _corner_collision_traces(space, stages, weights):
    answer = dict.fromkeys(range(space.n_controls), Fraction(0))
    for chart, weight in zip(space.cells, weights, strict=True):
        sizes = [dimension + 1 for dimension in chart.dimensions]
        corners = [tuple(alpha.index(1) for alpha in index) for index in _indices(chart.dimensions, chart.degree)]
        denominator = math.prod(size * (size + 1) for size in sizes)
        for i, j in product(range(len(corners)), repeat=2):
            if chart.controls[i] == chart.controls[j]:
                coincidences = sum(a == b for a, b in zip(corners[i], corners[j], strict=True))
                answer[chart.controls[i]] += weight * Fraction(2**coincidences, denominator)
    assert all(len(proper) == 1 for _, proper in stages)
    return [answer[proper[0]] for _, proper in stages]


@pytest.mark.parametrize("kind", ["point", "segment", "loop", "chain", "mixed", "triangles", "torus", "sphere"])
@pytest.mark.parametrize("mode", ["all", "maximal"])
def test_linear_corner_collisions_include_identifications_and_degenerate_faces(kind, mode):
    space = _space(kind, 1)
    grams, stages = _assembled_grams(space)
    propagation, blocks = _population_system(space, grams, stages)
    weights = _weights(space, mode)
    traces = _corner_collision_traces(space, stages, weights)
    assert traces == _traces(_metric(grams, weights), propagation, stages, blocks)
    assert len(traces) == space.n_controls
    grouped = len(traces) * sum(traces)
    assert grouped <= space.n_controls
    if any(weight and len(set(chart.controls)) >= 2 for chart, weight in zip(space.cells, weights, strict=True)):
        assert grouped < space.n_controls
    if space.n_controls == 1:
        assert traces == [Fraction(1)]
