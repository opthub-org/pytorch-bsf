"""Numerical contracts independent of implementation details."""

import math
import sys
from unittest.mock import patch

import pytest
import torch

from torch_bsf import BezierSpace, FixedControls, Points, QuadraticPenalty, Samples, SimploidalSet, fit
from torch_bsf._problem import LeastSquaresProblem
from torch_bsf.core import BasisOperator
from torch_bsf.results import ConvergenceError, ConvergenceWarning
from torch_bsf.sampling import sample
from torch_bsf.solvers import DenseLeastSquares, SparseLeastSquares


def test_dense_driver_is_recorded():
    result = fit(BezierSpace.simplex(1, degree=1), Samples(torch.eye(2), [[1.0], [2.0]]))
    assert result.diagnostics.driver == "gelsd"
    assert result.diagnostics.output_success == (True,)


def test_dense_nonfinite_output_preserves_other_columns_and_fixed_values():
    from types import SimpleNamespace

    space = BezierSpace.simplex(1, degree=1)
    samples = Samples(torch.eye(2), [[1.0, 2.0], [3.0, 4.0]])
    answer = SimpleNamespace(
        solution=torch.tensor([[5.0, float("nan")]], dtype=torch.float64),
        singular_values=torch.ones(1, dtype=torch.float64),
        rank=torch.tensor(1),
    )
    with patch("torch.linalg.lstsq", return_value=answer), pytest.warns(ConvergenceWarning):
        result = fit(space, samples, fixed=FixedControls({0: [1.0, 2.0]}))
    assert not result.success and result.status == "numerical_failure"
    assert result.diagnostics.output_success == (True, False)
    assert result.diagnostics.output_statuses == ("solved", "numerical_failure")
    assert not result.diagnostics.solution_available
    assert result.diagnostics.system.method == "svd"
    assert torch.equal(result.model.coefficients, torch.tensor([[1.0, 2.0], [5.0, 0.0]], dtype=torch.float64))


def test_optional_data_diagnostic_failure_preserves_the_computed_fit():
    space = BezierSpace.simplex(1, degree=2)
    samples = Samples(torch.eye(2), [[1.0], [2.0]])
    penalty = QuadraticPenalty(ridge=0.1)
    reference = fit(space, samples, penalty=penalty)
    with (
        patch("torch.linalg.svdvals", side_effect=torch.linalg.LinAlgError("data SVD failed")),
        pytest.warns(ConvergenceWarning),
    ):
        result = fit(space, samples, penalty=penalty, solver=DenseLeastSquares(diagnostics="full"))
    assert not result.success and result.status == "diagnostic_failure"
    assert result.diagnostics.solution_available
    assert result.diagnostics.output_success == (True,)
    assert result.diagnostics.data.method == "not_computed"
    assert result.diagnostics.system.method == "svd"
    assert torch.equal(result.model.coefficients, reference.model.coefficients)
    assert result.metrics == reference.metrics


def test_disabled_penalties_do_not_overflow_on_exact_large_fixed_values():
    space = BezierSpace.simplex(1, degree=1)
    values = torch.tensor([[1e200], [-1e200]], dtype=torch.float64)
    result = fit(space, Samples(torch.eye(2), values), fixed=FixedControls({0: values[0], 1: values[1]}))
    assert result.success and result.status == "all_fixed"
    assert all(value == 0 for value in result.metrics.values())


def test_augmented_rhs_overflow_is_reported_before_executing_a_solver():
    space = BezierSpace.simplex(1, degree=1)
    with pytest.warns(ConvergenceWarning):
        result = fit(
            space,
            Samples(torch.eye(2), [[0.0], [0.0]]),
            fixed=FixedControls({0: torch.tensor([1e200], dtype=torch.float64)}),
            penalty=QuadraticPenalty(ridge=1e300),
        )
    assert not result.success and result.status == "numerical_failure"
    assert not result.diagnostics.solution_available
    assert result.diagnostics.driver is None


@pytest.mark.parametrize("solver", [DenseLeastSquares(), SparseLeastSquares()])
@pytest.mark.parametrize("bad_column", [0, 1])
def test_rhs_overflow_preserves_independently_solvable_columns(solver, bad_column):
    if isinstance(solver, SparseLeastSquares):
        pytest.importorskip("scipy")
    space = BezierSpace.simplex(1, degree=1)
    good_column = 1 - bad_column
    values = torch.zeros(2, 2, dtype=torch.float64)
    values[:, good_column] = torch.tensor([1.0, 2.0], dtype=torch.float64)
    fixed_values = torch.ones(2, dtype=torch.float64)
    fixed_values[bad_column] = 1e308
    samples = Samples(torch.eye(2), values)
    fixed = FixedControls({0: fixed_values})
    penalty = QuadraticPenalty(ridge=4.0)
    reference = fit(
        space, Samples(torch.eye(2), [[1.0], [2.0]]), fixed=FixedControls({0: [1.0]}), penalty=penalty, solver=solver
    )
    with pytest.warns(ConvergenceWarning):
        result = fit(space, samples, fixed=fixed, penalty=penalty, solver=solver)
    assert not result.success and not result.diagnostics.solution_available
    assert result.diagnostics.output_success[good_column]
    assert not result.diagnostics.output_success[bad_column]
    assert result.diagnostics.output_statuses[bad_column] == "rhs_overflow"
    assert torch.allclose(result.model.coefficients[:, good_column], reference.model.coefficients[:, 0], atol=1e-12)
    assert result.model.coefficients[0, bad_column] == 1e308
    assert result.model.coefficients[1, bad_column] == 0
    if isinstance(solver, SparseLeastSquares):
        assert result.diagnostics.stop_codes[bad_column] is None
        assert result.diagnostics.iterations[bad_column] is None
        assert result.diagnostics.residual_norms[bad_column] is None
        assert result.diagnostics.stop_codes[good_column] == reference.diagnostics.stop_codes[0]


def test_all_fixed_rhs_overflow_retains_the_complete_model():
    space = BezierSpace.simplex(0, degree=0)
    samples = Samples(torch.empty(1, 0), torch.tensor([[-1e308]], dtype=torch.float64))
    fixed = FixedControls({0: torch.tensor([1e308], dtype=torch.float64)})
    with patch("torch.linalg.lstsq", side_effect=AssertionError("unnecessary solve")), pytest.warns(ConvergenceWarning):
        result = fit(space, samples, fixed=fixed)
    assert not result.success  # Its objective exceeds float64 range.
    assert result.diagnostics.solution_available
    assert result.diagnostics.output_success == (True,)
    assert result.diagnostics.output_statuses == ("all_fixed",)
    assert result.diagnostics.driver is None
    assert result.model.coefficients.item() == 1e308


@pytest.mark.parametrize("solver", [DenseLeastSquares(), SparseLeastSquares()])
def test_zero_weight_observation_does_not_overflow_metrics(solver):
    if isinstance(solver, SparseLeastSquares):
        pytest.importorskip("scipy")
    space = BezierSpace.simplex(0, degree=0)
    values = torch.tensor([[2.0], [1e200]], dtype=torch.float64)
    result = fit(space, Samples(torch.empty(2, 0), values, weights=[1.0, 0.0]), solver=solver)
    assert result.success
    assert result.model.coefficients.item() == pytest.approx(2.0)
    assert result.metrics["objective"] == pytest.approx(0.0, abs=1e-24)


def test_small_positive_weight_is_applied_before_squaring_residuals():
    space = BezierSpace.simplex(0, degree=0)
    values = torch.tensor([[0.0], [1e200]], dtype=torch.float64)
    result = fit(space, Samples(torch.empty(2, 0), values, weights=[1.0, 1e-200]), fixed=FixedControls({0: [0.0]}))
    assert result.success
    assert result.metrics["data_loss"] == pytest.approx(1e200)


def test_extreme_weight_ratio_preserves_representable_weighted_error():
    space = BezierSpace.simplex(0, degree=0)
    values = torch.tensor([[0.0], [1e200]], dtype=torch.float64)
    result = fit(space, Samples(torch.empty(2, 0), values, weights=[1e308, 1e-100]), fixed=FixedControls({0: [0.0]}))
    assert result.success
    assert result.metrics["data_loss"] == pytest.approx(1e-8)


def test_small_weight_prevents_overflow_in_the_residual_difference():
    space = BezierSpace.simplex(0, degree=0)
    values = torch.tensor([[1e308], [-1e308]], dtype=torch.float64)
    result = fit(space, Samples(torch.empty(2, 0), values, weights=[1.0, 1e-320]), fixed=FixedControls({0: values[0]}))
    assert result.success
    assert result.metrics["data_loss"] == pytest.approx(4e296, rel=2e-5)


def test_small_positive_penalties_are_applied_before_squaring_controls():
    space = BezierSpace.simplex(1, degree=1)
    values = torch.tensor([[1e200], [-1e200]], dtype=torch.float64)
    result = fit(
        space,
        Samples(torch.eye(2), values),
        fixed=FixedControls({0: values[0], 1: values[1]}),
        penalty=QuadraticPenalty(ridge=1e-300, smoothness=1e-300),
    )
    assert result.success
    assert result.metrics["ridge_penalty"] == pytest.approx(2e100)
    assert result.metrics["smoothness_penalty"] == pytest.approx(4e100)


def test_large_equal_fixed_controls_have_zero_smoothness_penalty():
    space = BezierSpace.simplex(1, degree=1)
    values = torch.full((2, 1), 1e308, dtype=torch.float64)
    result = fit(
        space,
        Samples(torch.eye(2), values),
        fixed=FixedControls({0: values[0], 1: values[1]}),
        penalty=QuadraticPenalty(smoothness=4.0),
    )
    assert result.success and result.status == "all_fixed"
    assert result.diagnostics.solution_available
    assert all(value == 0 for value in result.metrics.values())


def test_weighted_metrics_preserve_adjacent_large_value_differences():
    point = torch.tensor(1e150, dtype=torch.float64)
    adjacent = torch.nextafter(point, torch.zeros_like(point))
    scale = 0.85003
    samples = Samples(torch.empty(2, 0), torch.stack([adjacent, point])[:, None], weights=[scale**2, 1 - scale**2])
    result = fit(BezierSpace.simplex(0, degree=0), samples, fixed=FixedControls({0: point.reshape(1)}))
    assert result.success
    expected = ((point - adjacent) * scale).square().item()
    assert result.metrics["data_loss"] == pytest.approx(expected)


def test_diminishing_smoothness_preserves_adjacent_controls_and_rhs():
    point = torch.tensor(1e308, dtype=torch.float64)
    adjacent = torch.nextafter(point, torch.zeros_like(point))
    values = torch.stack([point, adjacent])[:, None]
    space = BezierSpace.simplex(1, degree=1)
    samples = Samples(torch.eye(2), values)
    fixed = FixedControls({0: values[0], 1: values[1]})
    penalty = QuadraticPenalty(smoothness=5.026395887680219e-297)
    expected_row = (point - adjacent) * math.sqrt(penalty.smoothness)
    result = fit(space, samples, fixed=fixed, penalty=penalty)
    assert result.success
    assert result.metrics["smoothness_penalty"] == pytest.approx(expected_row.square().item())
    problem = LeastSquaresProblem(space, samples, penalty, fixed)
    assert problem.rhs[len(samples), 0].item() == pytest.approx(-expected_row.item())


def test_fixed_control_data_rhs_preserves_a_small_solvable_difference():
    point = torch.tensor(1e150, dtype=torch.float64)
    observed = torch.nextafter(point / 2, torch.zeros_like(point))
    scale = 0.85003
    samples = Samples([[0.5, 0.5], [1.0, 0.0]], torch.stack([observed, point])[:, None], weights=[scale**2, 1 - scale**2])
    result = fit(BezierSpace.simplex(1, degree=1), samples, fixed=FixedControls({0: point.reshape(1)}))
    assert result.success
    assert result.model.coefficients[1, 0].item() == pytest.approx(2 * (observed - point / 2).item())


def test_scaled_difference_fallback_keeps_finite_penalty_gradients():
    space = BezierSpace.simplex(1, degree=1)
    problem = LeastSquaresProblem(space, Samples(torch.eye(2), [[0.0], [0.0]]), QuadraticPenalty(smoothness=1e-320), None)
    coefficients = torch.tensor([[1e308], [-1e308]], dtype=torch.float64, requires_grad=True)
    problem.full_matmat(coefficients)[2:].square().sum().backward()
    assert coefficients.grad is not None
    assert torch.allclose(coefficients.grad[:, 0], torch.tensor([4e-12, -4e-12], dtype=torch.float64), atol=0, rtol=2e-5)


@pytest.mark.parametrize("solver", [DenseLeastSquares(), SparseLeastSquares()])
def test_large_fixed_component_does_not_overflow_a_free_component_rhs(solver):
    if isinstance(solver, SparseLeastSquares):
        pytest.importorskip("scipy")
    domain = SimploidalSet()
    a, b = domain.add_simplex(1), domain.add_simplex(1)
    space = BezierSpace.on_set(domain, degree=1)
    samples = Samples.from_cells(
        space, [(a, torch.eye(2), torch.full((2, 1), 1e308, dtype=torch.float64)), (b, torch.eye(2), [[1.0], [2.0]])]
    )
    large = torch.tensor([1e308], dtype=torch.float64)
    fixed = FixedControls.from_cells(space, [(a, (1, 0), large), (a, (0, 1), large)])
    result = fit(space, samples, fixed=fixed, penalty=QuadraticPenalty(smoothness=4.0), solver=solver)
    assert result.success
    assert torch.isfinite(result.model.coefficients).all()
    assert all(math.isfinite(value) for value in result.metrics.values())


@pytest.mark.parametrize("failure", ["decomposition", "nonfinite"])
def test_dense_numerical_failure_returns_a_failed_result(failure):
    space = BezierSpace.simplex(1, degree=1)
    observations = Samples(torch.eye(2), [[1.0], [2.0]])
    fixed = FixedControls({0: [3.0]})
    if failure == "decomposition":
        effect = torch.linalg.LinAlgError("SVD failed to converge")
        options = {"side_effect": effect}
    else:
        from types import SimpleNamespace

        options = {
            "return_value": SimpleNamespace(
                solution=torch.tensor([[float("nan")]], dtype=torch.float64),
                singular_values=torch.ones(1, dtype=torch.float64),
                rank=torch.tensor(1),
            )
        }
    with patch("torch.linalg.lstsq", **options), pytest.warns(ConvergenceWarning):
        result = fit(space, observations, fixed=fixed)
    assert not result.success and result.status == "numerical_failure"
    assert not result.diagnostics.solution_available
    assert result.diagnostics.output_success == (False,)
    assert result.model.coefficients[0, 0] == 3.0
    assert torch.isfinite(result.model.coefficients).all()
    with pytest.raises(ConvergenceError) as error:
        result.require_success()
    assert error.value.result is result


@pytest.mark.parametrize("penalty", [QuadraticPenalty(), QuadraticPenalty(0.03, 0.07)])
@pytest.mark.parametrize("fixed", [None, FixedControls({0: [2.0, -3.0]})])
def test_dense_sparse_same_objective(penalty, fixed):
    pytest.importorskip("scipy")
    space = BezierSpace.simploid((2, 1), degree=(3, 2))
    points = sample(space, 120, seed=3)
    values = torch.stack([points.parameters[:, 1] ** 2, points.parameters[:, 4]], 1)
    observations = Samples(points, values, weights=torch.arange(120))
    dense = fit(space, observations, penalty=penalty, fixed=fixed)
    with patch.object(BasisOperator, "to_dense", side_effect=AssertionError("dense allocation")):
        sparse = fit(
            space, observations, penalty=penalty, fixed=fixed, solver=SparseLeastSquares(atol=1e-12, btol=1e-12, max_iter=200)
        )
    assert sparse.success
    assert torch.allclose(dense.model.coefficients, sparse.model.coefficients, atol=1e-7, rtol=1e-7)
    assert dense.metrics["objective"] == pytest.approx(sparse.metrics["objective"], abs=1e-12)
    assert len(sparse.diagnostics.stop_codes) == 2


def test_rank_deficiency_is_distinct_from_regularized_solution():
    space = BezierSpace.simplex(1, degree=3)
    observations = Samples([[1.0, 0.0], [1.0, 0.0]], [[1.0], [2.0]])
    result = fit(space, observations, penalty=QuadraticPenalty(ridge=0.1), solver=DenseLeastSquares(diagnostics="full"))
    assert result.success
    assert result.diagnostics.data.rank == 1
    assert result.diagnostics.data.condition_number == float("inf")
    assert result.diagnostics.system.rank == 4


def test_sparse_failure_cannot_look_successful():
    pytest.importorskip("scipy")
    space = BezierSpace.simplex(2, degree=5)
    points = sample(space, 60, seed=3)
    with pytest.warns(ConvergenceWarning):
        result = fit(space, Samples(points, points.parameters[:, :1] ** 3), solver=SparseLeastSquares(max_iter=1))
    assert not result.success and 7 in result.diagnostics.stop_codes
    with pytest.raises(ConvergenceError) as error:
        result.require_success()
    assert error.value.result is result


@pytest.mark.parametrize("solver", [DenseLeastSquares(), SparseLeastSquares()])
def test_fixed_all_needs_no_factorization(solver):
    space = BezierSpace.simplex(1, degree=1)
    fixed = FixedControls({0: [1.0], 1: [2.0]})
    with (
        patch("torch.linalg.lstsq", side_effect=AssertionError),
        patch.object(LeastSquaresProblem, "sparse", side_effect=AssertionError),
    ):
        result = fit(space, Samples(torch.eye(2), [[0.0], [0.0]]), fixed=fixed, solver=solver)
    assert result.success and result.status == "all_fixed"
    assert result.diagnostics.n_free_controls == 0
    assert result.diagnostics.driver is None


def test_sparse_extra_error_is_lazy(monkeypatch):
    config = SparseLeastSquares()
    monkeypatch.setitem(sys.modules, "scipy", None)
    with pytest.raises(ImportError, match=r"pytorch-bsf\[sparse\]"):
        fit(BezierSpace.simplex(1, degree=1), Samples(torch.eye(2), [[1.0], [2.0]]), solver=config)


def test_basis_adjoint_and_order():
    space = BezierSpace.simploid((1, 1), degree=2)
    points = sample(space, 17, seed=3)
    operator = space.basis(points)
    x = torch.randn(space.n_controls, 3, dtype=torch.float64)
    y = torch.randn(17, 3, dtype=torch.float64)
    assert torch.allclose((operator.matmat(x) * y).sum(), (x * operator.rmatmat(y)).sum())
    assert torch.allclose(operator.matmat(x), operator.to_dense() @ x)


def test_fitting_autograd_inputs_is_explicitly_nondifferentiable():
    pytest.importorskip("scipy")
    t = torch.eye(2, dtype=torch.float64, requires_grad=True)
    result = fit(BezierSpace.simplex(1, degree=1), Samples(Points(t), t[:, :1]), solver=SparseLeastSquares())
    assert result.success


def _reference(space, samples, penalty, fixed=None):
    """Assemble the mathematical objective independently of LeastSquaresProblem."""
    scipy = pytest.importorskip("scipy.linalg")
    a = space.basis(samples.points.to()).to_dense()
    scale = (samples.weights / samples.weights.sum()).sqrt()[:, None]
    edges = space.adjacency()
    d = torch.zeros(len(edges), space.n_controls, dtype=torch.float64)
    for k, (i, j) in enumerate(edges):
        d[k, i], d[k, j] = 1, -1
    blocks = [a * scale]
    if penalty.ridge:
        blocks.append(math.sqrt(penalty.ridge) * torch.eye(space.n_controls, dtype=torch.float64))
    if penalty.smoothness:
        blocks.append(math.sqrt(penalty.smoothness) * d)
    augmented = torch.cat(blocks)
    rhs = torch.cat(
        [
            samples.values.double() * scale,
            torch.zeros(len(augmented) - len(samples), samples.values.shape[1], dtype=torch.float64),
        ]
    )
    controls = torch.zeros(space.n_controls, samples.values.shape[1], dtype=torch.float64)
    free = torch.ones(space.n_controls, dtype=torch.bool)
    for row, value in (fixed.values if fixed else {}).items():
        controls[row], free[row] = value, False
    if free.any():
        solved = scipy.lstsq(augmented[:, free].numpy(), (rhs - augmented @ controls).numpy(), lapack_driver="gelss")[0]
        controls[free] = torch.from_numpy(solved)
    return controls, augmented[:, free]


@pytest.mark.parametrize("case", ["full_rank", "deficient", "underdetermined", "ill_conditioned"])
@pytest.mark.parametrize("ridge", [0.0, 1e-18, 0.03])
def test_dense_float64_reference_and_diagnostic_definitions(case, ridge):
    space = BezierSpace.simplex(1, degree=3)
    x = {
        "full_rank": torch.linspace(0, 1, 15, dtype=torch.float64),
        "deficient": torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float64),
        "underdetermined": torch.tensor([0.2, 0.8], dtype=torch.float64),
        "ill_conditioned": torch.linspace(0, 0.001, 15, dtype=torch.float64),
    }[case]
    samples = Samples(torch.stack([1 - x, x], 1), torch.stack([torch.sin(x), 1 + x * x], 1))
    penalty = QuadraticPenalty(ridge=ridge)
    expected, augmented = _reference(space, samples, penalty)
    result = fit(space, samples, penalty=penalty, solver=DenseLeastSquares(diagnostics="full"))
    assert result.success
    assert torch.allclose(result.model.coefficients, expected, atol=2e-5, rtol=2e-5)
    assert torch.allclose(result.model(samples.points), space.basis(samples.points).matmat(expected), atol=1e-10)
    diagnostics = result.diagnostics
    singular = torch.linalg.svdvals(augmented)
    threshold = torch.finfo(torch.float64).eps * max(augmented.shape) * singular[0]
    assert diagnostics.system.rank_threshold == pytest.approx(threshold.item())
    assert diagnostics.system.rank == int((singular > threshold).sum())
    assert diagnostics.system.singular_values is not None
    assert torch.allclose(diagnostics.system.singular_values, singular, atol=1e-14)
    if case in ("deficient", "underdetermined"):
        assert diagnostics.data.condition_number == float("inf")
    else:
        assert diagnostics.data.condition_number is not None
        assert diagnostics.data.condition_number >= 1
    assert diagnostics.model_dtype == diagnostics.compute_dtype == "float64"
    assert diagnostics.compute_device == diagnostics.model_device == "cpu"


def _mixed_space():
    domain = SimploidalSet()
    prism, cube = domain.add_simploid((2, 1)), domain.add_simploid((1, 1, 1))
    domain.identify(prism.face(0, 0), cube.face(0, 0))
    space = BezierSpace.on_set(domain, degree={prism: (2, 3), cube: (4, 2, 3)})
    # Two references to the same vertex on the shared square.
    fixed = FixedControls.from_cells(
        space,
        [
            (prism, ((0, 2, 0), (3, 0)), [2.0, -1.0]),
            (cube, ((0, 4), (2, 0), (3, 0)), [2.0, -1.0]),
        ],
    )
    assert len(fixed.values) == 1
    return space, fixed


@pytest.mark.parametrize("backend", ["dense", "sparse"])
@pytest.mark.parametrize("all_fixed", [False, True])
def test_mixed_degrees_shared_fixed_weighted_penalties_reference(backend, all_fixed):
    space, fixed = _mixed_space()
    points = sample(space, 140, seed=42)
    values = torch.stack([points.parameters[:, 1] ** 2, points.parameters[:, 0] + 1], 1)
    samples = Samples(points, values, weights=torch.arange(140).remainder(9))
    if all_fixed:
        fixed = FixedControls({k: [k / 10, -k / 20] for k in range(space.n_controls)})
    penalty = QuadraticPenalty(0.01, 0.03)
    expected, _ = _reference(space, samples, penalty, fixed)
    solver = (
        DenseLeastSquares(diagnostics="full")
        if backend == "dense"
        else SparseLeastSquares(atol=1e-13, btol=1e-13, max_iter=500)
    )
    result = fit(space, samples, solver=solver, penalty=penalty, fixed=fixed)
    assert result.success
    assert torch.allclose(result.model.coefficients, expected, atol=1e-9, rtol=1e-9)
    p = result.model.coefficients.detach()
    residual = result.model(points) - values
    data = (residual.square().sum(1) * samples.weights).sum() / samples.weights.sum()
    edges = space.adjacency()
    ridge = penalty.ridge * p.square().sum()
    smooth = penalty.smoothness * (p[edges[:, 0]] - p[edges[:, 1]]).square().sum()
    assert result.metrics["data_mse"] == pytest.approx(data.item() / 2)
    assert result.metrics["objective"] == pytest.approx((data + ridge + smooth).item())
    assert result.diagnostics.weight_sum == samples.weights.sum().item()
    assert result.diagnostics.n_samples == 140
    if all_fixed:
        assert result.diagnostics.driver is None
    for row, value in fixed.values.items():
        assert torch.equal(p[row], value)


@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_weighted_replication_and_global_weight_scaling(backend):
    pytest.importorskip("scipy")
    space = BezierSpace.simplex(1, degree=3)
    points = sample(space, 20, seed=5)
    samples = Samples(points, torch.cat([points.parameters**2, points.parameters[:, :1]], 1), weights=torch.arange(20))
    penalty, fixed = QuadraticPenalty(0.02, 0.07), FixedControls({0: [1.0, 2.0, 3.0]})
    solver = DenseLeastSquares() if backend == "dense" else SparseLeastSquares(atol=1e-13, btol=1e-13, max_iter=100)
    original = fit(space, samples, solver=solver, penalty=penalty, fixed=fixed)
    for data in (
        Samples(points.parameters.repeat(3, 1), samples.values.repeat(3, 1), weights=samples.weights.repeat(3)),
        Samples(points, samples.values, weights=7 * samples.weights),
    ):
        result = fit(space, data, solver=solver, penalty=penalty, fixed=fixed)
        assert torch.allclose(result.model.coefficients, original.model.coefficients, atol=1e-11)
        assert result.metrics == pytest.approx(original.metrics, abs=1e-12)


@pytest.mark.parametrize("code", range(8))
def test_lsmr_stop_code_contract_for_each_output(code):
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    answer = (np.ones(2), code, 3, 0.1, 0.01, 1.0, 2.0, 1.0)
    with patch("scipy.sparse.linalg.lsmr", return_value=answer):
        if code in (3, 6, 7):
            with pytest.warns(ConvergenceWarning):
                result = fit(
                    BezierSpace.simplex(1, degree=1), Samples(torch.eye(2), torch.ones(2, 2)), solver=SparseLeastSquares()
                )
        else:
            result = fit(
                BezierSpace.simplex(1, degree=1), Samples(torch.eye(2), torch.ones(2, 2)), solver=SparseLeastSquares()
            )
    assert result.success == (code in (0, 1, 2, 4, 5))
    assert result.diagnostics.output_success == (result.success,) * 2
    assert result.diagnostics.stop_codes == (code, code)
    assert result.diagnostics.iterations == (3, 3)
    assert len(set(result.diagnostics.output_statuses)) == 1
    assert result.diagnostics.system.method == "estimate"
    assert result.diagnostics.system.rank is None
    assert result.diagnostics.system.singular_values is None


def test_real_lsmr_mixed_success_and_iteration_limit():
    pytest.importorskip("scipy")
    space = BezierSpace.simplex(2, degree=4)
    points = sample(space, 50, seed=9)
    values = torch.cat([torch.zeros(50, 1), points.parameters[:, :1] ** 2], 1)
    with pytest.warns(ConvergenceWarning):
        result = fit(space, Samples(points, values), solver=SparseLeastSquares(max_iter=1))
    assert result.diagnostics.stop_codes == (0, 7)
    assert result.diagnostics.output_success == (True, False)
    assert result.diagnostics.output_statuses == ("zero_solution", "iteration_limit")
    assert not result.success and result.diagnostics.solution_available


def test_real_lsmr_condition_limit_is_not_success():
    pytest.importorskip("scipy")
    space = BezierSpace.simplex(2, degree=4)
    points = sample(space, 50, seed=9)
    with pytest.warns(ConvergenceWarning):
        result = fit(
            space, Samples(points, points.parameters[:, :1] ** 2 + 0.1), solver=SparseLeastSquares(conlim=1.01, atol=0, btol=0)
        )
    assert result.diagnostics.stop_codes == (3,)
    assert result.diagnostics.output_statuses == ("condition_limit",)
    assert not result.success


@pytest.mark.parametrize("inconsistent", [False, True])
def test_real_lsmr_machine_precision_stopping(inconsistent):
    pytest.importorskip("scipy")
    space = BezierSpace.simplex(2, degree=4)
    points = sample(space, 50, seed=9)
    values = points.parameters[:, :1] ** 2 + 0.1
    if inconsistent:
        values = values + 0.01 * torch.arange(50, dtype=torch.float64)[:, None].sin()
    result = fit(space, Samples(points, values), solver=SparseLeastSquares(conlim=0, atol=0, btol=0, max_iter=500))
    assert result.success
    assert result.diagnostics.stop_codes[0] in (4, 5)


@pytest.mark.parametrize("nonfinite", ["coefficients", "residual", "condition"])
def test_lsmr_nonfinite_output_cannot_be_successful(nonfinite):
    np = pytest.importorskip("numpy")
    pytest.importorskip("scipy")
    good = (np.ones(2), 1, 2, 0.0, 0.0, 1.0, 1.0, 1.0)
    bad = list(good)
    index = {"coefficients": 0, "residual": 3, "condition": 6}[nonfinite]
    bad[index] = np.full(2, np.nan) if index == 0 else float("nan")
    with patch("scipy.sparse.linalg.lsmr", side_effect=[good, tuple(bad)]), pytest.warns(ConvergenceWarning):
        result = fit(BezierSpace.simplex(1, degree=1), Samples(torch.eye(2), torch.ones(2, 2)), solver=SparseLeastSquares())
    assert not result.success and result.status == "numerical_failure"
    assert result.diagnostics.output_success == (True, False)
    assert result.diagnostics.output_statuses[1] == "numerical_failure"
    assert result.diagnostics.solution_available == (index != 0)
    assert torch.isfinite(result.model.coefficients).all()


@pytest.mark.parametrize("observations", [[[1.0, 0.0], [1.0, 0.0]], [[0.2, 0.8], [0.7, 0.3]]])
def test_sparse_rank_deficient_and_underdetermined_minimum_norm(observations):
    pytest.importorskip("scipy")
    space = BezierSpace.simplex(1, degree=3)
    samples = Samples(observations, [[1.0, 2.0], [2.0, -1.0]])
    dense = fit(space, samples)
    sparse = fit(space, samples, solver=SparseLeastSquares(atol=1e-13, btol=1e-13, max_iter=100))
    assert sparse.success
    assert torch.allclose(sparse.model.coefficients, dense.model.coefficients, atol=1e-10)
    assert sparse.diagnostics.data.rank is None


def test_sparse_many_cells_sums_identified_columns_without_densifying():
    pytest.importorskip("scipy")
    domain = SimploidalSet()
    cells = [domain.add_simplex(1) for _ in range(80)]
    for left, right in zip(cells[:-1], cells[1:], strict=True):
        domain.identify(left.face(0, 0), right.face(0, 1))
    # A loop identifies two local endpoint columns of the same cell.
    domain.identify(cells[0].face(0, 0), cells[0].face(0, 1))
    space = BezierSpace.on_set(domain, degree=2)
    points = sample(space, 1200, seed=9)
    samples = Samples(points, torch.full((1200, 2), 2.0))
    reference = fit(space, samples, penalty=QuadraticPenalty(smoothness=0.01))
    with (
        patch.object(LeastSquaresProblem, "dense", side_effect=AssertionError("dense problem")),
        patch.object(BasisOperator, "to_dense", side_effect=AssertionError("dense basis")),
        patch("torch.eye", side_effect=AssertionError("dense identity")),
    ):
        result = fit(
            space,
            samples,
            penalty=QuadraticPenalty(smoothness=0.01),
            solver=SparseLeastSquares(atol=1e-12, btol=1e-12, max_iter=500),
        )
    assert result.success
    assert torch.allclose(result.model(points), reference.model(points), atol=1e-8)
    assert result.metrics["objective"] == pytest.approx(reference.metrics["objective"], abs=1e-14)


def test_basic_diagnostics_reuse_decomposition_and_full_is_opt_in():
    space = BezierSpace.simplex(1, degree=3)
    samples = Samples(torch.eye(2), [[1.0], [2.0]])
    penalty = QuadraticPenalty(ridge=0.1)
    with patch("torch.linalg.svdvals", side_effect=AssertionError("extra SVD")):
        basic = fit(space, samples, penalty=penalty)
    assert basic.diagnostics.data.rank is None
    assert basic.diagnostics.data.condition_number is None
    assert basic.diagnostics.system.rank == space.n_controls
    with patch("torch.linalg.svdvals", wraps=torch.linalg.svdvals) as svd:
        full = fit(space, samples, penalty=penalty, solver=DenseLeastSquares(diagnostics="full"))
    assert svd.call_count == 1
    assert full.diagnostics.data.rank == 2
    assert torch.equal(basic.model.coefficients, full.model.coefficients)


def test_rcond_controls_rank_cutoff_and_minimum_norm_solution():
    space = BezierSpace.simplex(1, degree=2)
    samples = Samples([[1.0, 0.0], [0.999, 0.001], [0.0, 1.0]], [[1.0], [2.0], [3.0]])
    result = fit(space, samples, solver=DenseLeastSquares(rcond=0.1))
    a = space.basis(samples.points.to()).to_dense() / math.sqrt(3)
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    inverse = torch.where(s > 0.1 * s[0], 1 / s, 0)
    reference = (vh.T * inverse) @ u.T @ (samples.values.double() / math.sqrt(3))
    assert torch.allclose(result.model.coefficients, reference, atol=1e-12)
    assert result.diagnostics.system.rank == 2
    assert result.diagnostics.system.condition_number == float("inf")


def test_objective_overflow_is_a_numerical_failure():
    with pytest.warns(ConvergenceWarning):
        result = fit(
            BezierSpace.simplex(0, degree=0),
            Samples(torch.empty(2, 0), torch.tensor([[1e308], [-1e308]], dtype=torch.float64)),
        )
    assert not result.success and result.status == "numerical_failure"
    with pytest.raises(ConvergenceError):
        result.require_success()


@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_constant_weighted_mean_reference(backend):
    pytest.importorskip("scipy")
    space = BezierSpace.simplex(0, degree=0)
    samples = Samples(torch.empty(3, 0), [[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]], weights=[1.0, 0.0, 3.0])
    solver = DenseLeastSquares() if backend == "dense" else SparseLeastSquares()
    result = fit(space, samples, solver=solver)
    assert result.success
    assert torch.allclose(result.model.coefficients, torch.tensor([[4.0, 8.0]], dtype=torch.float64))
