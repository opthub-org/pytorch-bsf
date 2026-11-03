"""Independent conditional PRESS, support conversion and retained actual BSF."""

import numpy as np
import pytest
import torch

pytest.importorskip("scipy")

from scipy.linalg import cho_factor, cho_solve  # noqa: E402

from torch_bsf._adaptive_curve.component_kernel import ComponentKernelModel, _covariance, _SmoothPosterior  # noqa: E402
from torch_bsf._adaptive_curve.export import _compile_result  # noqa: E402
from torch_bsf._adaptive_curve.kernel import _fixed_fit  # noqa: E402
from torch_bsf._adaptive_curve.polynomial import CurveGap, StructureLearningResult, _Spline  # noqa: E402
from torch_bsf._adaptive_curve.support_converter import approximate_operator, component_splines  # noqa: E402


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


def fitted_model(kind):
    points = np.linspace(-1, 1, 17)
    unit = (points + 1) / 2
    labels = np.array([0] * 9 + [1] * 8, dtype=np.int64)
    values = np.column_stack((np.sin(4 * points) + 0.5 * labels, np.cos(3 * points) - 0.2 * labels))
    covariance = _covariance(unit, unit, labels, labels, kind, 0.3, 0.4)
    diagonal = np.full(len(points), 0.003)
    trend = np.ones((len(points), 1))
    fit = _fixed_fit(covariance, values, trend, diagonal)
    assert fit is not None
    factor = cho_factor(covariance + np.diag(diagonal), lower=True)
    residual = values - trend @ fit.beta
    operator = _SmoothPosterior(points, unit, kind, 0.3, 0, factor, residual, fit.beta, np.zeros(2), 1.0)
    indicator = (labels[:, None] == np.arange(2)[None, :]).astype(float)
    offsets = cho_solve(factor, 0.4 * indicator).T @ residual
    return ComponentKernelModel(
        torch.from_numpy(points[:, None]),
        torch.from_numpy(values),
        [torch.arange(9), torch.arange(9, 17)],
        offsets,
        {},
        operator,
    )


@pytest.mark.parametrize("kind", ["matern32", "matern52"])
def test_known_support_intervals_preserve_native_accuracy_actual_bsf_mask_and_global_rng(kind):
    model = fitted_model(kind)
    state = torch.get_rng_state().clone()
    pieces = component_splines(model)
    assert torch.equal(state, torch.get_rng_state())
    gaps = (CurveGap(0, 0.125, 1, 0.01, 0),)
    actual = _compile_result(
        StructureLearningResult(model.inputs, model.values, gaps, "budget_exhausted", {}, pieces, (-1, 1))
    )
    query = torch.unique(torch.cat((torch.linspace(-1, 1, 2049, dtype=torch.float64), model.inputs[:, 0])))[:, None]
    native, compiled = model.predict(query), actual.predict(query)
    assert torch.equal(torch.isfinite(native), torch.isfinite(compiled))
    finite = torch.isfinite(native).all(1)
    error = float((native[finite] - compiled[finite]).norm(dim=1).max())
    assert error <= model.diagnostics["conversion_checked_tolerance"]
    assert model.diagnostics["initial_support_intervals"] == len(model.inputs) - 1
    assert model.diagnostics["conversion_oracle_rows"] == 0
    assert actual._compiled is not None and actual._compiled.domain.betti_numbers()[0] == 2


def test_support_partition_still_respects_global_piece_budget_and_rejects_invalid_support():
    model = fitted_model("matern32")
    model._smooth.support = np.linspace(-1, 1, 1026)
    with pytest.raises(ArithmeticError, match="piece limit"):
        approximate_operator(model.inputs, model.values, model._smooth, model._smooth_predict, {})
    for support in (np.array([-2.0, 0, 1]), np.array([np.nan]), np.array([[0.0]])):
        model._smooth.support = support
        with pytest.raises(ValueError, match="support breakpoints"):
            approximate_operator(model.inputs, model.values, model._smooth, model._smooth_predict, {})


def test_scalar_conversion_dispatch_retains_the_existing_path(monkeypatch):
    model = fitted_model("matern32")
    model.values = model.values[:, :1]
    sentinel: list[_Spline] = []
    monkeypatch.setattr(model, "as_splines", lambda: sentinel)
    assert component_splines(model) is sentinel
