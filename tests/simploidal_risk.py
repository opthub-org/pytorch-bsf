"""Exact and estimator-level checks of the assembled-domain risk theorem.

These are reference calculations for the theory, not a public skeleton API.
"""

import math
from fractions import Fraction
from itertools import product

import numpy as np
import pytest
import torch

from tests.sampling import _condition, _exact_solve, _rational_gram, _rational_skeleton_risk
from torch_bsf import BezierSpace, FixedControls, Points, Samples, SimploidalSet, fit
from torch_bsf._kernels.combinatorics import _indices


def _space(kind, degree):
    domain = SimploidalSet()
    if kind == "point":
        domain.add_simplex(0)
    elif kind in ("segment", "loop", "chain", "disconnected"):
        left = domain.add_simplex(1)
        if kind == "loop":
            domain.identify(left.face(0, 0), left.face(0, 1))
        elif kind in ("chain", "disconnected"):
            right = domain.add_simplex(1)
            if kind == "chain":
                domain.identify(left.face(0, 0), right.face(0, 1))
    elif kind == "mixed":
        triangle, square = domain.add_simplex(2), domain.add_simploid((1, 1))
        domain.identify(triangle.face(0, 0), square.face(0, 0))
        return BezierSpace.on_set(domain, degree={triangle: degree, square: (1, degree)})
    elif kind == "triangles":
        left, right = domain.add_simplex(2), domain.add_simplex(2)
        domain.identify(left.face(0, 0), right.face(0, 1))
    else:
        square = domain.add_simploid((1, 1))
        if kind == "sphere":
            for j in range(2):
                edge = square.face(1, j)
                domain.degenerate(edge, edge.face(0, 0), 0, -1)
        else:
            assert kind == "torus"
            domain.identify(square.face(1, 0), square.face(1, 1))
        domain.identify(square.face(0, 0), square.face(0, 1))
    return BezierSpace.on_set(domain, degree=degree)


def _assembled_grams(space):
    """Scatter exact local moments, summing every occurrence of an identified control."""
    grams, stages = [], []
    for cell_id, cell in enumerate(space.cells):
        alphas = [sum(index, ()) for index in _indices(cell.dimensions, cell.degree)]
        local = _rational_gram(
            cell.dimensions, cell.degree, alphas, tuple(range(cell.width)), range(len(alphas)), range(len(alphas))
        )
        gram = [[Fraction(0)] * space.n_controls for _ in range(space.n_controls)]
        for i, row in enumerate(cell.controls):
            for j, column in enumerate(cell.controls):
                gram[row][column] += local[i][j]
        grams.append(gram)
        proper = tuple(cell.controls[i] for i, alpha in enumerate(alphas) if all(a > 0 for a in alpha))
        if proper:
            stages.append((cell_id, proper))
    stages.sort(key=lambda stage: sum(space.cells[stage[0]].dimensions))
    assert sorted(row for _, proper in stages for row in proper) == list(range(space.n_controls))
    return grams, stages


def _population_system(space, grams, stages):
    """Assemble the theorem's unit block triangular normal equations, then solve once."""
    p = space.n_controls
    system = [[Fraction(0)] * p for _ in range(p)]
    blocks = []
    for cell_id, proper in stages:
        gram = grams[cell_id]
        h = [[gram[i][j] for j in proper] for i in proper]
        normalized = _exact_solve(h, [gram[i] for i in proper])
        for i, row in enumerate(proper):
            system[row] = normalized[i]
        blocks.append(h)
    identity = [[Fraction(i == j) for j in range(p)] for i in range(p)]
    return _exact_solve(system, identity), blocks


def _traces(metric, propagation, stages, blocks):
    p = len(metric)
    result = []
    for (_, proper), h in zip(stages, blocks, strict=True):
        inner = [
            [sum(propagation[r][i] * metric[r][c] * propagation[c][j] for r in range(p) for c in range(p)) for j in proper]
            for i in proper
        ]
        solved = _exact_solve(h, inner)
        result.append(sum(solved[i][i] for i in range(len(proper))))
    return result


def _metric(grams, weights):
    return [
        [sum(w * g[i][j] for w, g in zip(weights, grams, strict=True)) for j in range(len(grams[0]))]
        for i in range(len(grams[0]))
    ]


def _quadrature(cell):
    """Independent Gauss integration after stick breaking, exact for basis products."""
    factors = []
    for dimension, degree in zip(cell.dimensions, cell.degree, strict=True):
        nodes, weights = np.polynomial.legendre.leggauss(degree + dimension + 1)
        nodes, weights = (nodes + 1) / 2, weights / 2
        factor_rows = []
        for indices in product(range(len(nodes)), repeat=dimension):
            remaining, weight, coordinates = 1.0, float(math.factorial(dimension)), []
            for j, i in enumerate(indices):
                coordinates.append(remaining * nodes[i])
                remaining *= 1 - nodes[i]
                weight *= weights[i] * (1 - nodes[i]) ** (dimension - j - 1)
            factor_rows.append((coordinates + [remaining], weight))
        factors.append(factor_rows)
    rows = list(product(*factors))
    parameters = torch.tensor([sum((x[0] for x in row), []) for row in rows], dtype=torch.float64)
    quadrature_weights = torch.tensor([math.prod(x[1] for x in row) for row in rows], dtype=torch.float64)
    return parameters, quadrature_weights


def _integrated_risk(space, covariance, chart_weights):
    """Evaluate covariance directly in each chart, independently of the assembled metric."""
    risk = 0.0
    for cell_id, chart_weight in enumerate(chart_weights):
        if chart_weight:
            parameters, weights = _quadrature(space.cells[cell_id])
            design = space.basis(Points.from_cells(space, [(cell_id, parameters)])).to_dense()
            variances = ((design @ covariance) * design).sum(1)
            risk += float(chart_weight) * (weights @ variances).item()
    return risk


_U = torch.finfo(torch.float64).eps / 2


def _quadrature_error(cell, n_controls):
    # First-order reference-test budget, not an all-degree bound for leggauss. These rules have order <= 6.
    # D^2 budgets low-order node/weight generation; 2*width budgets basis operations; n_q and p budget
    # positive Gram reductions and repeated-control accumulation. Eight covers both evaluation paths
    # and higher-order terms. gamma(k) = k U / (1 - k U) bounds k floating-point operations.
    assert max((d + a + 1 for d, a in zip(cell.degree, cell.dimensions, strict=True)), default=0) <= 6
    n_rows = math.prod((d + a + 1) ** a for d, a in zip(cell.degree, cell.dimensions, strict=True))
    operations = n_rows + sum(cell.degree) ** 2 + 2 * cell.width + n_controls
    return 8 * operations * _U / (1 - operations * _U)


def _propagation_error(space, stages, propagation, blocks):
    # Each local normalization amplifies Gram errors by cond(H); the unit triangular solve by cond(P).
    # Summing over all stages is conservative (same-dimension stages do not recursively amplify).
    # Four covers the two Gram/solve paths, SVD backward error and higher-order terms.
    gram_error = max(_quadrature_error(cell, space.n_controls) for cell in space.cells)
    return 4 * len(stages) * _condition(propagation) * max(_condition(h) for h in blocks) * gram_error


_CASES = [
    ("point", 1),
    ("segment", 3),
    ("loop", 1),
    ("loop", 3),
    ("chain", 2),
    ("disconnected", 2),
    ("triangles", 3),
    ("mixed", 2),
    ("torus", 2),
    ("sphere", 2),
]


@pytest.mark.parametrize(("kind", "degree"), _CASES)
def test_assembled_risk_matches_sequential_fits_and_independent_integration(kind, degree):
    space = _space(kind, degree)
    grams, stages = _assembled_grams(space)
    propagation, blocks = _population_system(space, grams, stages)
    fitted = torch.zeros(space.n_controls, space.n_controls, dtype=torch.float64)
    # Every stage receives its own columns of deterministic innovations. Fitting through the public
    # constrained solver checks the signs, repeated boundary controls, and sharing between all later cells.
    for cell_id, proper in stages:
        parameters, weights = _quadrature(space.cells[cell_id])
        points = Points.from_cells(space, [(cell_id, parameters)])
        design = space.basis(points).to_dense()
        exact = torch.tensor(grams[cell_id], dtype=torch.float64)
        gram_error = _quadrature_error(space.cells[cell_id], space.n_controls)
        assert torch.allclose(design.T @ (weights[:, None] * design), exact, atol=gram_error, rtol=gram_error)
        values = torch.zeros(len(points), space.n_controls, dtype=torch.float64)
        values[:, list(proper)] = design[:, list(proper)]
        fixed = FixedControls({i: fitted[i] for i in range(space.n_controls) if i not in proper})
        result = fit(space, Samples(points, values, weights=weights), fixed=fixed).require_success()
        fitted[list(proper)] = result.model.coefficients.detach()[list(proper)]
    propagation_error = _propagation_error(space, stages, propagation, blocks)
    exact_propagation = torch.tensor(propagation, dtype=torch.float64)
    absolute_error = propagation_error * torch.linalg.matrix_norm(exact_propagation, ord=2).item()
    assert torch.allclose(fitted, exact_propagation, atol=absolute_error, rtol=propagation_error)
    # Nonuniform evaluation includes lower cells as well as maximal cells. It is independent of
    # the nonuniform sampling shares. Positive chart weights make the global metric definite.
    weights = [Fraction(i + 1, sum(range(1, space.n_cells + 1))) for i in range(space.n_cells)]
    metric = _metric(grams, weights)
    traces = _traces(metric, propagation, stages, blocks)
    assert all(c > 0 for c in traces)
    shares = [Fraction(i + 1, sum(range(1, len(stages) + 1))) for i in range(len(stages))]
    covariance = torch.zeros_like(fitted)
    for (_, proper), h, q in zip(stages, blocks, shares, strict=True):
        e = fitted[:, list(proper)]
        covariance += e @ torch.linalg.solve(torch.tensor(h, dtype=torch.float64), e.T) / float(q)
    observed = _integrated_risk(space, covariance, weights)
    expected = float(sum(c / q for c, q in zip(traces, shares, strict=True)))
    # Write risk as ||G_mu^(1/2) P B^(1/2)||_F^2 with B=blockdiag(H_s^-1/q_s). Perturbing P gives
    # a relative first-order term 2*cond(P)*sqrt(cond(G_mu))*eta_P. Perturbing the H solves contributes
    # cond(H)*eta_Q. Eight covers both evaluation paths, integration, reductions and higher orders.
    gram_error = max(_quadrature_error(cell, space.n_controls) for cell in space.cells)
    weighted_condition = max(1, _condition(propagation) * math.sqrt(_condition(metric)))
    trace_error = 8 * (weighted_condition * propagation_error + max(_condition(h) for h in blocks) * gram_error + gram_error)
    assert observed == pytest.approx(expected, rel=trace_error, abs=0)


@pytest.mark.parametrize(("kind", "degree"), [("chain", 2), ("loop", 2), ("loop", 1)])
def test_assembled_hand_computed_coefficients(kind, degree):
    space = _space(kind, degree)
    grams, stages = _assembled_grams(space)
    propagation, blocks = _population_system(space, grams, stages)
    weights = [Fraction(int(i in space.main_cells), len(space.main_cells)) for i in range(space.n_cells)]
    traces = _traces(_metric(grams, weights), propagation, stages, blocks)
    by_dimension = [
        [c for (cell_id, _), c in zip(stages, traces, strict=True) if sum(space.cells[cell_id].dimensions) == m]
        for m in range(2)
    ]
    coefficients = [len(cs) * sum(cs) for cs in by_dimension]
    if kind == "chain":
        assert sorted(by_dimension[0]) == [Fraction(1, 16), Fraction(1, 16), Fraction(1, 8)]
        assert coefficients == [Fraction(3, 4), Fraction(2)]
    elif degree == 2:
        # Both local endpoints are the same noise source: the cross term changes c_v from 1/4 to 1/6.
        assert coefficients == [Fraction(1, 6), Fraction(1)]
    else:
        # A linear loop has just one global constant control, despite two local endpoint occurrences.
        assert coefficients == [Fraction(1), Fraction(0)]


@pytest.mark.parametrize(
    ("dimensions", "degrees"),
    [((1,), (d,)) for d in range(1, 5)] + [((1, 1), (2, 3)), ((2, 1), (2, 1)), ((2,), (3,))],
)
def test_assembled_formula_reduces_to_the_single_product(dimensions, degrees):
    domain = SimploidalSet()
    cell = domain.add_simploid(dimensions)
    space = BezierSpace.on_set(domain, degree={cell: degrees})
    grams, stages = _assembled_grams(space)
    propagation, blocks = _population_system(space, grams, stages)
    traces = _traces(grams[space.main_cells[0]], propagation, stages, blocks)
    coefficients = []
    for dimension in range(sum(dimensions) + 1):
        cs = [c for (cell_id, _), c in zip(stages, traces, strict=True) if sum(space.cells[cell_id].dimensions) == dimension]
        coefficients.append(len(cs) * sum(cs))
    assert coefficients == _rational_skeleton_risk(dimensions, degrees)


def test_zero_evaluation_weight_leaves_active_controls_with_zero_trace():
    # Only the left endpoint is evaluated. Errors on the other vertex and the interior have zero
    # trace, but they still need data if the estimator is to fit every global control.
    space = _space("segment", 2)
    grams, stages = _assembled_grams(space)
    propagation, blocks = _population_system(space, grams, stages)
    vertex = next(i for i, cell in enumerate(space.cells) if not cell.dimensions)
    traces = _traces(grams[vertex], propagation, stages, blocks)
    assert sorted(traces) == [Fraction(0), Fraction(0), Fraction(1)]


@pytest.mark.parametrize(("kind", "degree"), _CASES)
def test_cell_and_dimension_optimal_allocations(kind, degree):
    space = _space(kind, degree)
    grams, stages = _assembled_grams(space)
    propagation, blocks = _population_system(space, grams, stages)
    traces = _traces(grams[space.main_cells[0]], propagation, stages, blocks)
    # With zero traces, the formula gives an infimum over strictly positive allocations.
    positive = [float(c) for c in traces if c]
    root_sum = sum(math.sqrt(c) for c in positive)
    optimum = sum(c / (math.sqrt(c) / root_sum) for c in positive)
    groups = []
    for dimension in range(max(sum(cell.dimensions) for cell in space.cells) + 1):
        cs = [c for (cell_id, _), c in zip(stages, traces, strict=True) if sum(space.cells[cell_id].dimensions) == dimension]
        if cs:
            groups.append(cs)
    grouped = [len(cs) * sum(cs) for cs in groups]
    fractions = [Fraction(i + 1, sum(range(1, len(groups) + 1))) for i in range(len(groups))]
    # Evaluate the original cell objective under equal shares within groups, independently of a_g.
    cell_risk = sum(c / (rho / len(cs)) for cs, rho in zip(groups, fractions, strict=True) for c in cs)
    group_risk = sum(a / rho for a, rho in zip(grouped, fractions, strict=True))
    assert cell_risk == group_risk
    group_root_sum = sum(math.sqrt(a) for a in grouped)
    optimal_fractions = [math.sqrt(a) / group_root_sum for a in grouped]
    cell_risk_at_group_optimum = sum(
        float(c) / (rho / len(cs)) for cs, rho in zip(groups, optimal_fractions, strict=True) for c in cs if c
    )
    grouped_optimum = sum(math.sqrt(a) for a in grouped) ** 2
    # Positive conversions/square roots/divisions/sums: fewer than eight operations per stage/group
    # in either path. Sixteen covers both paths and higher-order roundings; no cancellation occurs.
    rounding = 16 * (len(stages) + len(groups) + 2) * _U
    assert optimum == pytest.approx(root_sum**2, rel=rounding, abs=0)
    assert cell_risk_at_group_optimum == pytest.approx(grouped_optimum, rel=rounding, abs=0)
    assert optimum <= grouped_optimum * (1 + rounding)


@pytest.mark.parametrize(("kind", "degree"), _CASES)
def test_global_all_at_once_risk_with_matching_and_different_evaluation_weights(kind, degree):
    space = _space(kind, degree)
    grams, _ = _assembled_grams(space)
    chart_weights = [Fraction(int(i in space.main_cells), len(space.main_cells)) for i in range(space.n_cells)]
    design_metric = _metric(grams, chart_weights)
    parameters, weights = [], []
    for cell_id in space.main_cells:
        t, w = _quadrature(space.cells[cell_id])
        parameters.append((cell_id, t))
        weights.append(w * float(chart_weights[cell_id]))
    points = Points.from_cells(space, parameters)
    row_weights = torch.cat(weights)
    # Under these deterministic quadrature weights, responses with covariance diag(1 / weights)
    # reproduce the population OLS covariance. Identity innovations extract the whole influence map.
    innovations = torch.diag(row_weights.rsqrt())
    result = fit(space, Samples(points, innovations, weights=row_weights)).require_success()
    influence = result.model.coefficients.detach()
    covariance = influence @ influence.T
    observed = _integrated_risk(space, covariance, chart_weights)
    # Gram perturbation/weighted-SVD errors are amplified by cond(G_nu); using the Gram condition
    # rather than its square root is conservative for the design solve. Two influence factors,
    # integration and trace reduction fit within a factor of eight, including higher-order terms.
    gram_error = max(_quadrature_error(cell, space.n_controls) for cell in space.cells)
    trace_error = 8 * _condition(design_metric) * gram_error
    assert observed == pytest.approx(space.n_controls, rel=trace_error, abs=0)
    evaluation_weights = [Fraction(i + 1, sum(range(1, space.n_cells + 1))) for i in range(space.n_cells)]
    evaluation_metric = _metric(grams, evaluation_weights)
    solved = _exact_solve(design_metric, evaluation_metric)
    expected = float(sum(solved[i][i] for i in range(space.n_controls)))
    observed = _integrated_risk(space, covariance, evaluation_weights)
    # Different evaluation weights additionally amplify norm-to-relative-risk error by cond(G_mu).
    assert observed == pytest.approx(expected, rel=trace_error * _condition(evaluation_metric), abs=0)
