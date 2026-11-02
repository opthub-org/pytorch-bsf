"""Value-only fitting: geometric recovery, constraints and stopping evidence."""

import math
import warnings
from dataclasses import replace
from itertools import combinations
from typing import Any
from unittest.mock import patch

import pytest
import torch
from hypothesis import example, given
from hypothesis import strategies as st

from tests.strategies import U, matrices, simploid_points, simploid_types
from torch_bsf import (
    BezierModel,
    BezierSpace,
    ConvergenceWarning,
    FixedControls,
    Samples,
    SimploidalSet,
    fit,
    fit_point_cloud,
    fit_skeleton,
)
from torch_bsf._kernels.combinatorics import _indices
from torch_bsf._skeleton import _blocks, _faces_with_interior_controls
from torch_bsf.point_cloud import _nondominated, _parameters, _Polynomial, _project, _project_factors
from torch_bsf.sampling import grid, sample, sample_skeleton


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def curve():
    u = torch.linspace(0, 1, 21, dtype=torch.float64)
    return torch.stack((u.square(), (1 - u).square()), 1)


@pytest.mark.parametrize("fitter", [fit_point_cloud, fit_skeleton])
def test_recovers_schaffer_curve_without_coordinates(fitter):
    space = BezierSpace.simplex(1, degree=2)
    y = curve()
    original = y.clone()
    result = fitter(space, y).require_success()
    expected = torch.tensor([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(result.model.coefficients, expected, atol=1e-8, rtol=0)
    torch.testing.assert_close(y, original, rtol=0, atol=0)
    if fitter is fit_point_cloud:
        torch.testing.assert_close(result.model(result.parameters), y, atol=1e-8, rtol=0)
        assert all(b <= a + 1e-14 for a, b in zip(result.objective_history[:-1], result.objective_history[1:], strict=True))
        assert result.parameter_converged


def test_triangle_recovers_curved_surface_from_value_only_faces():
    space = BezierSpace.simplex(2, degree=2)
    # Squared distance to each vertex, a three-objective convex Pareto front.
    data: dict[tuple[int, ...], torch.Tensor] = {}
    for face in ((0,), (1,), (2,), (0, 1), (0, 2), (1, 2)):
        t = torch.zeros(9 if len(face) == 2 else 1, 3, dtype=torch.float64)
        if len(face) == 1:
            t[:, face[0]] = 1
        else:
            t[:, face[0]] = torch.linspace(0, 1, len(t))
            t[:, face[1]] = 1 - t[:, face[0]]
        data[face] = ((t[:, None] - torch.eye(3)).square()).sum(2)
    y = torch.cat(list(data.values()))
    result = fit_skeleton(space, y, face_values=data).require_success()
    t = torch.tensor([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]], dtype=torch.float64)
    expected = (t[:, None] - torch.eye(3)).square().sum(2)
    torch.testing.assert_close(result.model(t), expected, atol=1e-7, rtol=0)
    assert len(result.faces) == 6  # Degree two needs no triangle-interior fit.
    for fitted_face, local in result.faces.items():
        if len(fitted_face) == 2:
            for axis, vertex in enumerate(fitted_face):
                powers = [0, 0]
                powers[axis] = 2
                row = local.model.space.control_index(powers)
                torch.testing.assert_close(local.model.coefficients[row], data[(vertex,)][0], rtol=0, atol=0)


def test_auto_face_extraction_matches_explicit_faces():
    y = curve()
    space = BezierSpace.simplex(1, degree=2)
    explicit = {(0,): y[:1], (1,): y[-1:], (0, 1): y}
    auto = fit_skeleton(space, y)
    supplied = fit_skeleton(space, y, face_values=explicit)
    torch.testing.assert_close(auto.model.coefficients, supplied.model.coefficients, rtol=0, atol=0)


def test_dominance_keeps_ties_and_full_outputs():
    y = torch.tensor([[0.0, 3.0, 7.0], [0.0, 3.0, 7.0], [1.0, 2.0, 8.0], [2.0, 4.0, 9.0]])
    torch.testing.assert_close(_nondominated(y, (0,)), y[:2], rtol=0, atol=0)
    torch.testing.assert_close(_nondominated(y, (0, 1)), y[:3], rtol=0, atol=0)


def test_projection_enforces_simplex_and_boundary():
    t = _project(torch.tensor([[-2.0, 4.0, 1.0], [0.2, 0.3, 0.5]], dtype=torch.float64))
    torch.testing.assert_close(t, torch.tensor([[0.0, 1.0, 0.0], [0.2, 0.3, 0.5]], dtype=torch.float64), rtol=0, atol=0)


def test_newton_projects_outside_points_to_boundary():
    space = BezierSpace.simplex(2, degree=1)
    model = BezierModel(space, torch.eye(3, dtype=torch.float64))
    y = torch.tensor([[-1.0, 2.0, 0.0], [0.2, 0.3, 0.5]], dtype=torch.float64)
    t, converged = _parameters(model, y, torch.full((2, 3), 1 / 3, dtype=torch.float64), 100, 1e-9)
    assert converged
    torch.testing.assert_close(t, _project(y), atol=1e-8, rtol=0)


def _derivative_tolerance(poly: _Polynomial, order: int) -> float:
    """Bound on the difference of two float64 evaluations of a derivative of ``order`` of ``poly`` on its domain.

    Each of the ``N`` terms is a control times a scaled monomial of degree at most ``D`` in ``w`` coordinates, rounded at
    most ``D + 2w`` times, and summing the terms rounds ``N - 1`` times more. On a simplex or simploid, with ``D`` the
    total degree, the absolute values of the scaled monomials of a derivative of order ``k`` sum to at most ``D**k``, so
    one evaluation (analytic or autograd) errs by at most ``(N + D + 2w) u D**k`` times the largest control; twice that
    for the difference, doubled.
    """
    n, degree, width = len(poly.powers), float(sum(poly.degrees)), poly.powers.shape[1]
    return 4 * (n + degree + 2 * width) * U * degree**order * float(poly.coefficients.abs().max())


def _constant_fit_tolerance(y: list[list[float]]) -> float:
    """Bound on the error of the least-squares control of a constant (0-simplex) model fitted to the rows ``y``.

    The design matrix of the ``m`` rows is a column of ones, whose condition number is one, and Householder least
    squares is backward stable: the solution, the mean, errs by a small multiple (4 here) of ``m u`` times the largest
    value. Doubled.
    """
    return 2 * 4 * len(y) * U * max(abs(v) for row in y for v in row)


def test_analytic_derivatives_match_autograd_including_boundary():
    space = BezierSpace.simplex(2, degree=3)
    cp = torch.randn(space.n_controls, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(17))
    poly = _Polynomial(BezierModel(space, cp))
    for values in ([0.2, 0.3, 0.5], [1.0, 0.0, 0.0]):
        t = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        jac = torch.autograd.functional.jacobian(lambda v: poly.derivative(v[None])[0], t)
        for i in range(3):
            torch.testing.assert_close(
                poly.derivative(t[None], (i,))[0], jac[:, i], rtol=0, atol=_derivative_tolerance(poly, 1)
            )
            for j in range(3):
                hess = torch.autograd.functional.jacobian(lambda v, axis=i: poly.derivative(v[None], (axis,))[0], t)
                torch.testing.assert_close(
                    poly.derivative(t[None], (i, j))[0], hess[:, j], rtol=0, atol=_derivative_tolerance(poly, 2)
                )


def test_arbitrary_embedding_initial_and_fixed_controls_are_preserved():
    space = BezierSpace.simplex(1, degree=2)
    initial = BezierModel(space, torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    original = initial.coefficients.detach().clone()
    u = torch.linspace(0, 1, 17, dtype=torch.float64)
    y = torch.stack((u, u * (1 - u), u.square()), 1)
    fixed = FixedControls({0: y[0], 2: y[-1]})
    result = fit_point_cloud(space, y, initial=initial, fixed=fixed, tolerance=1e-7).require_success()
    torch.testing.assert_close(initial.coefficients, original, rtol=0, atol=0)
    torch.testing.assert_close(result.model.coefficients[[0, 2]], y[[0, -1]], atol=0, rtol=0)
    assert result.objective_history[-1] < 1e-9
    assert result.model.coefficients.dtype == torch.float64


def test_constant_simplex():
    y = [[1.0, 2.0], [3.0, 4.0]]
    result = fit_point_cloud(BezierSpace.simplex(0, degree=3), y).require_success()
    assert result.parameters.shape == (2, 0)
    expected = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
    torch.testing.assert_close(result.model.coefficients, expected, rtol=0, atol=_constant_fit_tolerance(y))


def test_iteration_limit_retains_finite_iterate():
    with pytest.warns(ConvergenceWarning):
        result = fit_point_cloud(BezierSpace.simplex(1, degree=2), curve(), max_iter=1)
    assert not result.success
    assert result.status == "iteration_limit"
    assert torch.isfinite(result.model.coefficients).all()
    with pytest.raises(RuntimeError):
        result.require_success()


def test_skeleton_stops_on_unconverged_face():
    y = curve()
    data = {(0,): torch.stack((y[0], y[1])), (1,): y[-1:], (0, 1): y}
    with pytest.warns(ConvergenceWarning):
        result = fit_skeleton(BezierSpace.simplex(1, degree=2), y, face_values=data, max_iter=1)
    assert not result.success
    assert list(result.faces) == [(0,)]
    with pytest.raises(RuntimeError):
        result.require_success()


@pytest.mark.parametrize("values", [[], [[float("nan"), 1.0]], [[complex(1, 2), 1.0]], [1.0, 2.0]])
def test_invalid_observations(values):
    with pytest.raises(ValueError):
        fit_point_cloud(BezierSpace.simplex(1, degree=2), values)


@pytest.mark.parametrize(
    "kwargs", [{"max_iter": 0}, {"parameter_max_iter": True}, {"tolerance": -1.0}, {"parameter_tolerance": float("nan")}]
)
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        fit_point_cloud(BezierSpace.simplex(1, degree=2), curve(), **kwargs)


def test_unsupported_spaces_and_embedding():
    domain = SimploidalSet()
    left, right = domain.add_simplex(1), domain.add_simplex(1)
    domain.identify(left.face(0, 0), right.face(0, 1))
    for fitter in (fit_point_cloud, fit_skeleton):
        with pytest.raises(ValueError, match="single simplex or simploid"):
            fitter(BezierSpace.on_set(domain, degree=2), curve())
        with pytest.raises(ValueError, match="degree >= 1"):
            fitter(BezierSpace.simploid((1, 1), degree=(2, 0)), curve())
        with pytest.raises(ValueError, match="requires a single simplex; supply initial"):
            fitter(BezierSpace.simploid((1, 1), degree=2), curve())
    with pytest.raises(ValueError, match="degree >= 1"):
        fit_point_cloud(BezierSpace.simplex(1, degree=0), curve())
    with pytest.raises(ValueError, match="initial"):
        fit_point_cloud(BezierSpace.simplex(2, degree=2), curve())
    with pytest.raises(ValueError, match="face_values"):
        fit_skeleton(BezierSpace.simplex(1, degree=2), curve(), face_values={(0,): curve()})


def test_fixed_shape_validation():
    with pytest.raises(ValueError, match="fixed"):
        fit_point_cloud(BezierSpace.simplex(1, degree=2), curve(), fixed=FixedControls({9: [0.0, 1.0]}))


@pytest.mark.parametrize("width, degree", [(3, 3), (5, 2)])
def test_higher_faces_and_support_cutoff(width, degree):
    space = BezierSpace.simplex(width - 1, degree=degree)
    data = {}
    for size in range(1, min(width, degree) + 1):
        for face in combinations(range(width), size):
            z = torch.rand(12, size, generator=torch.Generator().manual_seed(size), dtype=torch.float64)
            z /= z.sum(1, keepdim=True)
            t = torch.zeros(12, width, dtype=torch.float64)
            t[:, list(face)] = z
            data[face] = (t[:, None] - torch.eye(width)).square().sum(2)
    y = torch.cat(list(data.values()))
    result = fit_skeleton(space, y, face_values=data, max_iter=200).require_success()
    assert set(result.faces) == set(data)
    t = torch.full((1, width), 1 / width, dtype=torch.float64)
    expected = (t[:, None] - torch.eye(width)).square().sum(2)
    torch.testing.assert_close(result.model(t), expected, atol=1e-2, rtol=0)
    if degree == 3:
        assert result.faces[tuple(range(width))].objective_history[-1] < 1e-6


def test_overflow_returns_explicit_failure_with_finite_model():
    with pytest.warns(ConvergenceWarning, match="overflow"):
        result = fit_point_cloud(BezierSpace.simplex(1, degree=1), [[1e200, 0.0], [0.0, 1e200]])
    assert not result.success
    assert result.status == "numerical_failure"
    assert result.n_iterations == 0
    assert torch.isfinite(result.model.coefficients).all()


def test_skeleton_arbitrary_embedding_with_supplied_faces():
    space = BezierSpace.simplex(1, degree=1)
    y = torch.tensor([[0.0, 0.0, 0.0], [0.5, 1.0, 1.5], [1.0, 2.0, 3.0]], dtype=torch.float64)
    initial = BezierModel(space, y[[0, -1]])
    result = fit_skeleton(space, y, initial=initial, face_values={(0,): y[:1], (1,): y[-1:]}).require_success()
    torch.testing.assert_close(result.model.coefficients, y[[0, -1]], rtol=0, atol=0)


def test_invalid_space_initial_and_constraint_types():
    space = BezierSpace.simplex(1, degree=2)
    with pytest.raises(TypeError, match="BezierSpace"):
        fit_point_cloud(None, curve())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="FixedControls"):
        fit_point_cloud(space, curve(), fixed={})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="initial"):
        fit_point_cloud(space, curve(), initial=BezierModel(space, torch.zeros(3, 3)))


def test_skeleton_constant_and_embedding_validation():
    y = [[1.0, 2.0], [3.0, 4.0]]
    result = fit_skeleton(BezierSpace.simplex(0, degree=0), y).require_success()
    expected = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
    torch.testing.assert_close(result.model.coefficients, expected, rtol=0, atol=_constant_fit_tolerance(y))
    space = BezierSpace.simplex(1, degree=1)
    initial = BezierModel(space, torch.zeros(2, 3))
    with pytest.raises(ValueError, match="face_values"):
        fit_skeleton(space, torch.zeros(3, 3), initial=initial)
    with pytest.raises(ValueError, match="full observed output"):
        fit_skeleton(space, curve(), face_values={(0,): [[0.0]], (1,): [[1.0]]})


def test_parameter_iteration_exhaustion_preserves_descent():
    space = BezierSpace.simplex(1, degree=2)
    model = BezierModel(space, torch.tensor([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]], dtype=torch.float64))
    y = curve()
    start = torch.full((len(y), 2), 0.5, dtype=torch.float64)
    t, converged = _parameters(model, y, start, 1, 1e-12)
    assert not converged
    assert ((model(t) - y).square().sum(1) <= (model(start) - y).square().sum(1)).all()
    # _project rounds the cumulative sum of the w = 2 sorted coordinates w - 1 times and subtracts one, divides it once
    # for the threshold, and subtracts the threshold from each coordinate; the test sums the w results with w - 1 more
    # roundings. Each rounding errs by at most u times the magnitude M of the coordinates, which is at most 2 here (the
    # start 0.5 plus one step; 1.5 observed), so |sum - 1| <= 3 w u M; doubled.
    width, magnitude = 2, 2
    torch.testing.assert_close(t.sum(1), torch.ones(len(y), dtype=torch.float64), rtol=0, atol=2 * 3 * width * magnitude * U)


def test_derivative_overflow_preserves_initial_model():
    space = BezierSpace.simplex(1, degree=1)
    initial = BezierModel(space, torch.tensor([[1e200, 0.0], [-1e200, 0.0]], dtype=torch.float64))
    with pytest.warns(ConvergenceWarning, match="derivatives"):
        result = fit_point_cloud(space, [[0.0, 0.0]], initial=initial)
    assert not result.success
    assert result.status == "numerical_failure"
    assert result.control_result is None
    torch.testing.assert_close(result.model.coefficients, initial.coefficients, rtol=0, atol=0)


def test_control_solver_failure_retains_last_surface(monkeypatch):
    def failed_solve(*args, **kwargs):
        raise torch.linalg.LinAlgError("injected SVD failure")

    monkeypatch.setattr(torch.linalg, "lstsq", failed_solve)
    with pytest.warns(ConvergenceWarning, match="SVD failure"):
        result = fit_point_cloud(BezierSpace.simplex(1, degree=2), curve())
    assert not result.success
    assert result.status == "control_failure"
    assert result.control_result is not None and not result.control_result.success
    assert torch.isfinite(result.model.coefficients).all()
    assert result.n_iterations == 0


def test_overflow_after_control_update_rejects_candidate(monkeypatch):
    space = BezierSpace.simplex(1, degree=1)
    initial = BezierModel(space, torch.eye(2, dtype=torch.float64))
    # A finite candidate may still have a non-finite squared-distance metric.
    solved = fit(space, Samples(torch.eye(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64)))
    candidate = BezierModel(space, torch.full((2, 2), 1e200, dtype=torch.float64))
    monkeypatch.setattr("torch_bsf.point_cloud.fit", lambda *args, **kwargs: replace(solved, model=candidate))
    with pytest.warns(ConvergenceWarning, match="objective overflowed"):
        result = fit_point_cloud(space, torch.eye(2, dtype=torch.float64), initial=initial)
    assert result.status == "numerical_failure"
    assert not result.success
    torch.testing.assert_close(result.model.coefficients, initial.coefficients, rtol=0, atol=0)
    assert len(result.objective_history) == 1


# ---------------------------------------------------------------------------
# simploids
# ---------------------------------------------------------------------------


def _perturbed(model: BezierModel, scale: float, seed: int) -> BezierModel:
    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(model.coefficients.shape, generator=generator, dtype=torch.float64)
    return BezierModel(model.space, model.coefficients.detach() + scale * noise)


def _saddle(dimensions=(1, 1), degree=(2, 1)) -> BezierModel:
    """An injective map of a simploid into 4D: barycentric coordinates plus a curved output."""
    space = BezierSpace.simploid(dimensions, degree=degree)
    t = grid(space, max(space.cells[0].degree)).parameters
    # Fit the exact polynomial t -> (t_0, t_last, curved) on the control lattice.
    curved = t[:, 0] * t[:, -1] + 0.3 * t[:, 0] ** 2
    values = torch.stack((t[:, 0], t[:, 1], t[:, -1], curved), 1)
    return fit(space, Samples(t, values)).model


@example(dimensions=(2, 1), data=None).via("two rows outside and inside the prism")
@given(dimensions=simploid_types(), data=st.data())
def test_projection_onto_products_projects_each_factor(dimensions, data):
    width = sum(a + 1 for a in dimensions)
    if data is None:
        t = torch.tensor([[-2.0, 4.0, 1.0, 0.7, 0.9], [0.2, 0.3, 0.5, 0.25, 0.75]], dtype=torch.float64)
    else:
        n = data.draw(st.integers(1, 5))
        t = data.draw(st.one_of(matrices(n, width, scale=st.floats(1e-3, 1e3)), simploid_points(dimensions, n)))
    projected = _project_factors(t, dimensions)
    assert projected.shape == t.shape and (projected >= 0).all()
    # The same operations on each factor's columns, so the results agree exactly.
    for block in _blocks(dimensions):
        torch.testing.assert_close(
            projected[:, block.start : block.stop], _project(t[:, block.start : block.stop]), rtol=0, atol=0
        )


@pytest.mark.parametrize(("dimensions", "degree"), [((1, 1), (2, 3)), ((2, 1), (2, 2))])
def test_analytic_derivatives_of_simploids_match_autograd(dimensions, degree):
    space = BezierSpace.simploid(dimensions, degree=degree)
    cp = torch.randn(space.n_controls, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
    model = BezierModel(space, cp)
    poly = _Polynomial(model)
    t0 = torch.cat([sample(space, 1, seed=1).parameters, grid(space, 1).parameters[:1]])
    for row in t0:
        t = row.clone().requires_grad_()
        torch.testing.assert_close(
            poly.derivative(t[None])[0], model(t[None])[0].detach(), rtol=0, atol=_derivative_tolerance(poly, 0)
        )
        jac = torch.autograd.functional.jacobian(lambda v: poly.derivative(v[None])[0], t)
        for i in range(space.width):
            torch.testing.assert_close(
                poly.derivative(t[None], (i,))[0], jac[:, i], rtol=0, atol=_derivative_tolerance(poly, 1)
            )
            hess = torch.autograd.functional.jacobian(lambda v, axis=i: poly.derivative(v[None], (axis,))[0], t)
            for j in range(space.width):
                torch.testing.assert_close(
                    poly.derivative(t[None], (i, j))[0], hess[:, j], rtol=0, atol=_derivative_tolerance(poly, 2)
                )


def test_high_degree_derivatives_of_simploids_match_the_monomial_formula():
    # A degree of 16 switches to the recursive basis of each factor; at moderate degree both must agree.
    space = BezierSpace.simploid((1, 2), degree=(16, 2))
    cp = torch.randn(space.n_controls, 1, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
    high = _Polynomial(BezierModel(space, cp))
    assert high.high
    low = _Polynomial(BezierModel(space, cp))
    low.high = False
    low.factors = torch.tensor(
        [
            math.prod(
                math.factorial(d) / math.prod(math.factorial(v) for v in factor)
                for d, factor in zip((16, 2), index, strict=True)
            )
            for index in _indices((1, 2), (16, 2))
        ],
        dtype=torch.float64,
    )
    t = sample(space, 4, seed=0).parameters
    # On a product of simplices the scaled derivative basis of order k still sums to at most D**k, D = 18. The
    # recursive basis of degree d in w coordinates rounds about w + 1 times per level, sum(d * (w + 1)) = 56 times in
    # all, fewer than the N + D + 2w = 130 roundings that _derivative_tolerance allows each evaluation.
    for axes in [(), (0,), (2,), (0, 3), (1, 1), (2, 4), (0,) * 17, (2, 3, 4)]:
        expected = low.derivative(t, axes)
        torch.testing.assert_close(high.derivative(t, axes), expected, rtol=0, atol=_derivative_tolerance(low, len(axes)))
    torch.testing.assert_close(high.derivative(t, (0,) * 17), torch.zeros(4, 1, dtype=torch.float64), rtol=0, atol=0)


@pytest.mark.parametrize(("dimensions", "degree", "scale"), [((1, 1), (2, 1), 0.05), ((2, 1), (1, 2), 0.0)])
def test_point_cloud_recovers_simploids_from_values(dimensions, degree, scale):
    # Starting near (square) or at (prism) the true controls, the alternating fit drives every distance to zero.
    truth = _saddle(dimensions, degree)
    space = truth.space
    y = truth(sample(space, 30, seed=2).parameters).detach()
    result = fit_point_cloud(space, y, initial=_perturbed(truth, scale, 0), max_iter=300).require_success()
    assert result.objective_history[-1] < 1e-8
    # Every inferred row stays on each factor's simplex: projection clamps at zero and keeps each block's sum,
    # up to the rounding of a few block-sized sums.
    assert (result.parameters >= 0).all()
    for block in (slice(0, dimensions[0] + 1), slice(dimensions[0] + 1, None)):
        torch.testing.assert_close(
            result.parameters[:, block].sum(1), torch.ones(len(y), dtype=torch.float64), rtol=0, atol=8 * U
        )
    assert (result.model(result.parameters).detach() - y).square().sum(1).max() < 1e-6


def test_point_cloud_keeps_fixed_simploid_controls():
    truth = _saddle()
    space = truth.space
    y = truth(sample(space, 30, seed=4).parameters).detach()
    corner = space.control_index(((2, 0), (1, 0)))
    fixed = FixedControls({corner: truth.coefficients[corner].detach()})
    # The initial corner is off; it is replaced by the fixed value before the first parameter update.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)  # three iterations need not converge
        result = fit_point_cloud(space, y, initial=_perturbed(truth, 0.05, 1), fixed=fixed, max_iter=3)
    torch.testing.assert_close(result.model.coefficients[corner], truth.coefficients[corner].detach(), rtol=0, atol=0)
    history = result.objective_history
    assert len(history) == 4 and all(b <= a + 1e-14 for a, b in zip(history[:-1], history[1:], strict=True))


def test_skeleton_recovers_a_simploid_from_face_values():
    truth = _saddle((2, 1), (2, 2))
    space = truth.space
    cell = space.cells[0]
    points = sample_skeleton(space, 300, ratio=[1, 1, 1, 1], seed=0).parameters
    support = [tuple(row.nonzero().flatten().tolist()) for row in points > 0]
    required = _faces_with_interior_controls(cell.dimensions, cell.degree)
    face_values = {face: truth(points[[k for k, f in enumerate(support) if f == face]]).detach() for face in required}
    y = torch.cat(list(face_values.values()))
    # The vertex controls start 0.5 off and the other controls at the truth; every face is refitted from its values.
    initial = BezierModel(space, truth.coefficients.detach().clone())
    vertices = [
        space.control_index((tuple(2 if i == a else 0 for i in range(3)), tuple(2 if j == b else 0 for j in range(2))))
        for a in range(3)
        for b in range(2)
    ]
    with torch.no_grad():
        initial.coefficients[vertices] += 0.5
    with patch("torch_bsf.point_cloud.fit_point_cloud", wraps=fit_point_cloud) as face_fit:
        result = fit_skeleton(space, y, face_values=face_values, initial=initial).require_success()
    assert list(result.faces) == required
    # Each face fixes exactly its local controls on a proper subface: the support of a local multi-index, mapped to
    # the face's vertices, plus the single vertex of every factor at one vertex, must differ from the face.
    blocks = [range(0, 3), range(3, 5)]
    for face, call in zip(required, face_fit.call_args_list, strict=True):
        local_space = call.args[0]
        parts = [[i for i in face if i in block] for block in blocks]
        expected = set()
        for k, index in enumerate(_indices(local_space.cells[0].dimensions, local_space.cells[0].degree)):
            wide = iter(index)
            on: list[int] = []
            for part in parts:
                on += part if len(part) == 1 else [v for v, a in zip(part, next(wide), strict=True) if a]
            if tuple(sorted(on)) != face:
                expected.add(k)
        assert set(call.kwargs["fixed"].values) == expected, face
    # At a vertex, every basis function is exactly 0 or 1, so the truth's values there are its vertex controls exactly.
    # A vertex control is the float64 SVD least-squares fit of a constant to k identical rows: their mean in exact
    # arithmetic, computed backward stably with an error of a small multiple of (k + 1) U; 4 (k + 1) U bounds it.
    rows = max(len(face_values[face]) for face in required if len(face) == 2)
    atol = 4 * (rows + 1) * U * float(truth.coefficients.detach().abs().max())
    torch.testing.assert_close(result.model.coefficients[vertices], truth.coefficients[vertices].detach(), rtol=0, atol=atol)
    # Each face fit lies on the truth: its surface reproduces the face's values.
    for face, local in result.faces.items():
        assert local.objective_history[-1] < 1e-8, face


def test_skeleton_on_simploids_requires_face_values_and_initial():
    space = BezierSpace.simploid((1, 1), degree=1)
    initial = BezierModel(space, torch.zeros(4, 2, dtype=torch.float64))
    with pytest.raises(ValueError, match="supply face_values"):
        fit_skeleton(space, torch.zeros(3, 2), initial=initial)
    with pytest.raises(ValueError, match=r"\(0, 2\), \(0, 3\), \(1, 2\), \(1, 3\)"):
        fit_skeleton(space, torch.zeros(3, 2), initial=initial, face_values={(0,): torch.zeros(1, 2)})


# Reference implementations from before #630, kept to check that the faster code computes the same values.


def _reference_nondominated(y: torch.Tensor, face: tuple[int, ...]) -> torch.Tensor:
    selected = y[:, face]
    keep = torch.ones(len(y), dtype=torch.bool)
    for i, point in enumerate(selected):
        keep[i] = not bool(((selected <= point).all(1) & (selected < point).any(1)).any())
    return y[keep]


class _ReferencePolynomial:
    """``_Polynomial`` before #630: every derivative rebuilds its power table or basis lookups."""

    def __init__(self, poly: _Polynomial):
        self.coefficients, self.dimensions, self.degrees = poly.coefficients, poly.dimensions, poly.degrees
        self.powers, self.high, self.factors = poly.powers, poly.high, poly.factors

    def derivative(self, t: torch.Tensor, axes: tuple[int, ...] = ()) -> torch.Tensor:
        from torch_bsf._kernels.combinatorics import _recursive_basis, _simplex_indices

        powers = self.powers.clone()
        if self.high:
            for axis in axes:
                powers[:, axis] -= 1
            derivative = torch.ones((len(t), len(powers)), dtype=t.dtype)
            for block, full in zip(_blocks(self.dimensions), self.degrees, strict=True):
                degree = full - sum(axis in block for axis in axes)
                if degree < 0:
                    return torch.zeros((len(t), self.coefficients.shape[1]), dtype=t.dtype)
                positions = {alpha: i + 1 for i, alpha in enumerate(_simplex_indices(len(block), degree))}
                indices = torch.tensor([positions.get(tuple(row), 0) for row in powers[:, block.start : block.stop].tolist()])
                basis = _recursive_basis(t[:, block.start : block.stop], degree)
                padded = torch.cat((torch.zeros_like(basis[:, :1]), basis), 1)
                derivative = derivative * (padded[:, indices] * math.prod(range(degree + 1, full + 1)))
            return derivative @ self.coefficients
        factors = self.factors.clone()
        for axis in axes:
            factors *= powers[:, axis].clamp_min(0)
            powers[:, axis] -= 1
        basis = t[:, None, :].pow(powers.clamp_min(0)[None, :, :]).prod(2) * factors
        return basis @ self.coefficients

    def geometry(self, t: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = self.derivative(t) - y
        width = t.shape[1]
        jacobian = torch.stack([self.derivative(t, (i,)) for i in range(width)], dim=2)
        gradient = torch.einsum("no,noi->ni", residual, jacobian)
        hessian = torch.einsum("noi,noj->nij", jacobian, jacobian)
        for i in range(width):
            for j in range(i, width):
                term = (residual * self.derivative(t, (i, j))).sum(1)
                hessian[:, i, j] += term
                if i != j:
                    hessian[:, j, i] += term
        return residual.square().sum(1), gradient, hessian


def _reference_line_search(poly, t, direction, y):
    best = t.clone()
    error = (poly.derivative(t) - y).square().sum(1)
    pending = torch.ones(len(t), dtype=torch.bool)
    step = 1.0
    for _ in range(25):
        candidate = _project_factors(t + step * direction, poly.dimensions)
        candidate_error = (poly.derivative(candidate) - y).square().sum(1)
        improved = pending & torch.isfinite(candidate_error) & (candidate_error < error)
        best[improved], error[improved] = candidate[improved], candidate_error[improved]
        pending &= ~improved & ((candidate - t).abs().amax(1) > 1e-15)
        if not pending.any():
            break
        step *= 0.5
    return best


def _reference_parameters(model, y, t, max_iter, tolerance):
    poly = _ReferencePolynomial(_Polynomial(model))
    tangent = torch.block_diag(*(torch.cat((torch.eye(a), -torch.ones(1, a))).to(torch.float64) for a in poly.dimensions))
    for _ in range(max_iter):
        _, gradient, hessian = poly.geometry(t, y)
        if (_project_factors(t - gradient, poly.dimensions) - t).abs().max().item() <= tolerance:
            return t, True
        reduced = tangent.T @ hessian @ tangent
        rhs = gradient @ tangent
        delta = -(torch.linalg.pinv(reduced, hermitian=True) @ rhs[:, :, None]).squeeze(2) @ tangent.T
        newton = _reference_line_search(poly, t, delta, y)
        scale = hessian.abs().sum(2).amax(1).clamp_min(1e-12)
        descent = _reference_line_search(poly, t, -gradient / scale[:, None], y)
        use_newton = (poly.derivative(newton) - y).square().sum(1) < (poly.derivative(descent) - y).square().sum(1)
        t = torch.where(use_newton[:, None], newton, descent)
    _, gradient, _ = poly.geometry(t, y)
    return t, bool((_project_factors(t - gradient, poly.dimensions) - t).abs().max() <= tolerance)


@given(
    st.integers(0, 60),
    st.integers(1, 4),
    st.integers(2, 6),
    st.integers(0, 2**32 - 1),
)
@example(2500, 1, 6, 0)  # several blocks of rows: 2**22 // (2500 * 1) < 2500
@example(1500, 3, 4, 1)  # several blocks with three objectives, many ties
@example(0, 2, 3, 0)  # no points
def test_blocked_nondominated_matches_the_pointwise_reference(n, m, levels, seed):
    """The blocked comparison of #630 keeps exactly the rows that the pointwise loop kept, ties included."""
    y = torch.randint(0, levels, (n, m + 1), generator=torch.Generator().manual_seed(seed)).double()
    for k in range(1, m + 1):
        for face in combinations(range(m), k):
            torch.testing.assert_close(_nondominated(y, face), _reference_nondominated(y, face), rtol=0, atol=0)


def _random_polynomial(dimensions, degree, seed):
    space = BezierSpace.simploid(dimensions, degree=degree)
    generator = torch.Generator().manual_seed(seed)
    model = BezierModel(space, torch.randn(space.n_controls, 2, dtype=torch.float64, generator=generator))
    return model, sample(space, 9, seed=seed).parameters


POLYNOMIAL_CASES = [((1,), 3), ((2,), 4), ((1, 1), (2, 3)), ((2, 1), (2, 2)), ((1,), 17), ((1, 2), (16, 2))]


@pytest.mark.parametrize(("dimensions", "degree"), POLYNOMIAL_CASES)
def test_cached_derivatives_match_the_reference(dimensions, degree):
    """Shared power tables and cached lookups (#630) give the same derivatives, bit for bit, as recomputing them."""
    model, t = _random_polynomial(dimensions, degree, 7)
    t = torch.cat([t, grid(model.space, 1).parameters])  # vertices: zero coordinates
    poly = _Polynomial(model)
    reference = _ReferencePolynomial(poly)
    width = t.shape[1]
    memo: dict[Any, torch.Tensor] = {}
    for axes in [(), *((i,) for i in range(width)), *combinations(range(width), 2), (0, 0), (0,) * 3, (0,) * 18]:
        expected = reference.derivative(t, axes)
        torch.testing.assert_close(poly.derivative(t, axes), expected, rtol=0, atol=0)
        torch.testing.assert_close(poly.derivative(t, axes, memo), expected, rtol=0, atol=0)  # tables shared
        torch.testing.assert_close(poly.derivative(t, axes, memo), expected, rtol=0, atol=0)  # lookups cached
    y = torch.randn(len(t), 2, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    for actual, expected in zip(poly.geometry(t, y), reference.geometry(t, y), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(("dimensions", "degree"), POLYNOMIAL_CASES[:5])
def test_newton_parameters_match_the_reference(dimensions, degree):
    """Reusing the squared errors of the line searches (#630) leaves every Newton iterate unchanged.

    The reference recomputes each error from the same rows with the same operations, so the iterates agree exactly.
    """
    model, _ = _random_polynomial(dimensions, degree, 3)
    y = torch.randn(25, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(4))
    start = sample(model.space, 25, seed=5).parameters
    for max_iter in (1, 4, 10):
        t, converged = _parameters(model, y, start, max_iter, 1e-12)
        expected, expected_converged = _reference_parameters(model, y, start, max_iter, 1e-12)
        torch.testing.assert_close(t, expected, rtol=0, atol=0)
        assert converged == expected_converged
