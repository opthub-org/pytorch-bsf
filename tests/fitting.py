"""Numerical contracts independent of implementation details."""

import sys
from unittest.mock import patch

import pytest
import torch

from torch_bsf import BezierSpace, FixedControls, Points, QuadraticPenalty, Samples, fit
from torch_bsf.core import BasisOperator
from torch_bsf.results import ConvergenceError, ConvergenceWarning
from torch_bsf.sampling import sample
from torch_bsf.solvers import DenseLeastSquares, SparseLeastSquares


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


def test_fixed_all_needs_no_factorization():
    space = BezierSpace.simplex(1, degree=1)
    fixed = FixedControls({0: [1.0], 1: [2.0]})
    with patch("torch.linalg.lstsq", side_effect=AssertionError):
        result = fit(space, Samples(torch.eye(2), [[0.0], [0.0]]), fixed=fixed)
    assert result.success and result.status == "all_fixed"
    assert result.diagnostics.n_free_controls == 0


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
