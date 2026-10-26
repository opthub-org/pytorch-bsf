"""Value-only fitting: geometric recovery, constraints and stopping evidence."""

from dataclasses import replace
from itertools import combinations

import pytest
import torch

from torch_bsf import (
    BezierModel,
    BezierSpace,
    ConvergenceWarning,
    FixedControls,
    Samples,
    fit,
    fit_point_cloud,
    fit_skeleton,
)
from torch_bsf.point_cloud import _nondominated, _parameters, _Polynomial, _project


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
    torch.testing.assert_close(y, original)
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
    torch.testing.assert_close(auto.model.coefficients, supplied.model.coefficients)


def test_dominance_keeps_ties_and_full_outputs():
    y = torch.tensor([[0.0, 3.0, 7.0], [0.0, 3.0, 7.0], [1.0, 2.0, 8.0], [2.0, 4.0, 9.0]])
    torch.testing.assert_close(_nondominated(y, (0,)), y[:2])
    torch.testing.assert_close(_nondominated(y, (0, 1)), y[:3])


def test_projection_enforces_simplex_and_boundary():
    t = _project(torch.tensor([[-2.0, 4.0, 1.0], [0.2, 0.3, 0.5]], dtype=torch.float64))
    torch.testing.assert_close(t, torch.tensor([[0.0, 1.0, 0.0], [0.2, 0.3, 0.5]], dtype=torch.float64))


def test_newton_projects_outside_points_to_boundary():
    space = BezierSpace.simplex(2, degree=1)
    model = BezierModel(space, torch.eye(3, dtype=torch.float64))
    y = torch.tensor([[-1.0, 2.0, 0.0], [0.2, 0.3, 0.5]], dtype=torch.float64)
    t, converged = _parameters(model, y, torch.full((2, 3), 1 / 3, dtype=torch.float64), 100, 1e-9)
    assert converged
    torch.testing.assert_close(t, _project(y), atol=1e-8, rtol=0)


def test_analytic_derivatives_match_autograd_including_boundary():
    space = BezierSpace.simplex(2, degree=3)
    cp = torch.randn(space.n_controls, 2, dtype=torch.float64, generator=torch.Generator().manual_seed(17))
    poly = _Polynomial(BezierModel(space, cp))
    for values in ([0.2, 0.3, 0.5], [1.0, 0.0, 0.0]):
        t = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        jac = torch.autograd.functional.jacobian(lambda v: poly.derivative(v[None])[0], t)
        for i in range(3):
            torch.testing.assert_close(poly.derivative(t[None], (i,))[0], jac[:, i])
            for j in range(3):
                hess = torch.autograd.functional.jacobian(lambda v, axis=i: poly.derivative(v[None], (axis,))[0], t)
                torch.testing.assert_close(poly.derivative(t[None], (i, j))[0], hess[:, j])


def test_arbitrary_embedding_initial_and_fixed_controls_are_preserved():
    space = BezierSpace.simplex(1, degree=2)
    initial = BezierModel(space, torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    original = initial.coefficients.detach().clone()
    u = torch.linspace(0, 1, 17, dtype=torch.float64)
    y = torch.stack((u, u * (1 - u), u.square()), 1)
    fixed = FixedControls({0: y[0], 2: y[-1]})
    result = fit_point_cloud(space, y, initial=initial, fixed=fixed, tolerance=1e-7).require_success()
    torch.testing.assert_close(initial.coefficients, original)
    torch.testing.assert_close(result.model.coefficients[[0, 2]], y[[0, -1]], atol=0, rtol=0)
    assert result.objective_history[-1] < 1e-9
    assert result.model.coefficients.dtype == torch.float64


def test_constant_simplex():
    result = fit_point_cloud(BezierSpace.simplex(0, degree=3), [[1.0, 2.0], [3.0, 4.0]]).require_success()
    assert result.parameters.shape == (2, 0)
    torch.testing.assert_close(result.model.coefficients, torch.tensor([[2.0, 3.0]], dtype=torch.float64))


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
    for space in (BezierSpace.simploid((1, 1), degree=2), BezierSpace.simplex(1, degree=0)):
        with pytest.raises(ValueError):
            fit_point_cloud(space, curve())
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
    torch.testing.assert_close(result.model.coefficients, y[[0, -1]])


def test_invalid_space_initial_and_constraint_types():
    space = BezierSpace.simplex(1, degree=2)
    with pytest.raises(TypeError, match="BezierSpace"):
        fit_point_cloud(None, curve())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="FixedControls"):
        fit_point_cloud(space, curve(), fixed={})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="initial"):
        fit_point_cloud(space, curve(), initial=BezierModel(space, torch.zeros(3, 3)))


def test_skeleton_constant_and_embedding_validation():
    result = fit_skeleton(BezierSpace.simplex(0, degree=0), [[1.0, 2.0], [3.0, 4.0]]).require_success()
    torch.testing.assert_close(result.model.coefficients, torch.tensor([[2.0, 3.0]], dtype=torch.float64))
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
    torch.testing.assert_close(t.sum(1), torch.ones(len(y), dtype=torch.float64))


def test_derivative_overflow_preserves_initial_model():
    space = BezierSpace.simplex(1, degree=1)
    initial = BezierModel(space, torch.tensor([[1e200, 0.0], [-1e200, 0.0]], dtype=torch.float64))
    with pytest.warns(ConvergenceWarning, match="derivatives"):
        result = fit_point_cloud(space, [[0.0, 0.0]], initial=initial)
    assert not result.success
    assert result.status == "numerical_failure"
    assert result.control_result is None
    torch.testing.assert_close(result.model.coefficients, initial.coefficients)


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
    torch.testing.assert_close(result.model.coefficients, initial.coefficients)
    assert len(result.objective_history) == 1
