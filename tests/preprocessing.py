"""Affine scaler contracts over generated feature tables."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from tests.strategies import U, feature_values, matrices, simplex_points
from torch_bsf.preprocessing import MinMaxScaler, NoneScaler, QuantileScaler, StdScaler, make_scaler


@pytest.mark.parametrize("scaler_class", [MinMaxScaler, StdScaler, QuantileScaler, NoneScaler])
@given(values=feature_values(), data=st.data())
def test_transform_applies_the_fitted_scale_to_new_values(scaler_class, values, data):
    unit_roundoff = torch.finfo(values.dtype).eps / 2
    scaler = scaler_class()
    original = values.clone()
    scaled = scaler.fit_transform(values)
    assert scaled.shape == values.shape and torch.isfinite(scaled).all()
    assert torch.equal(values, original)
    separately_fitted = scaler_class()
    separately_fitted.fit(values)
    assert torch.equal(scaled, separately_fitted.transform(values))
    assert torch.equal(scaled, scaler.transform(values))
    # Subtraction, division, multiplication and addition give at most four roundings.
    # Absolute error is measured against the feature magnitude, allowing cancellation near zero.
    magnitude = values.abs().amax(dim=0)
    assert ((scaler.inverse_transform(scaled) - values).abs() <= 8 * unit_roundoff * magnitude).all()

    new_values = data.draw(matrices(3, values.shape[1])).to(values.dtype) * magnitude
    transformed = scaler.transform(new_values)
    assert torch.isfinite(transformed).all()
    assert ((scaler.inverse_transform(transformed) - new_values).abs() <= 8 * unit_roundoff * magnitude).all()
    w = data.draw(simplex_points(3, 2)).to(values.dtype)
    left, right = scaler.transform(w @ new_values), w @ transformed
    # Each dot product has three terms; include normalization error in w and the
    # two affine evaluations, bounded in absolute units before cancellation.
    bound = (
        32 * unit_roundoff * (transformed.abs().amax(dim=0) + scaler.transform(torch.zeros_like(new_values)).abs().amax(dim=0))
    )
    assert ((left - right).abs() <= bound).all()


@given(values=feature_values())
def test_minmax_range(values):
    result = MinMaxScaler().fit_transform(values)
    varying = values.amax(dim=0) != values.amin(dim=0)
    assert torch.equal(result.amin(dim=0), torch.zeros(values.shape[1], dtype=values.dtype))
    assert torch.equal(result.amax(dim=0), varying.to(values.dtype))
    assert ((result >= 0) & (result <= 1)).all()


@given(values=feature_values(min_rows=2))
def test_standardized_mean_and_variance(values):
    result = StdScaler().fit_transform(values)
    varying = values.amax(dim=0) != values.amin(dim=0)
    # Mean/variance reductions take O(n) operations; account for centering and
    # division as well, with an absolute bound since the target mean is zero.
    bound = 16 * len(values) * torch.finfo(values.dtype).eps / 2
    assert (result.mean(dim=0).abs() <= bound).all()
    assert ((result.std(dim=0) - varying.to(values.dtype)).abs() <= bound).all()


@pytest.mark.parametrize("scaler_class", [MinMaxScaler, StdScaler, QuantileScaler])
@given(row=matrices(1, 4), n_rows=st.integers(1, 12))
def test_constant_columns_map_to_zero(scaler_class, row, n_rows):
    # Retains the explicit single-sample and zero-scale contracts of the old tests.
    values = row.expand(n_rows, -1)
    result = scaler_class().fit_transform(values)
    # StdScaler's mean reduction can round even for repeated identical values.
    assert (result.abs() <= 2 * n_rows * U * row.abs()).all()


@given(values=feature_values())
def test_quantiles_map_to_zero_and_one(values):
    scaler = QuantileScaler()
    scaler.fit(values)
    endpoints = torch.stack((values.quantile(scaler.q, dim=0), values.quantile(1 - scaler.q, dim=0)))
    result = scaler.transform(endpoints)
    varying = endpoints[0] != endpoints[1]
    assert torch.equal(result[0], torch.zeros_like(result[0]))
    assert torch.equal(result[1], varying.to(values.dtype))


@given(values=feature_values())
def test_none_scaler_is_identity(values):
    scaler = NoneScaler()
    scaler.fit(values)
    assert scaler.fit_transform(values) is values
    assert scaler.transform(values) is values
    assert scaler.inverse_transform(values) is values


@pytest.mark.parametrize(
    "normalize, scaler_class",
    [("max", MinMaxScaler), ("std", StdScaler), ("quantile", QuantileScaler), ("none", NoneScaler)],
)
def test_make_scaler(normalize, scaler_class):
    assert isinstance(make_scaler(normalize), scaler_class)


def test_make_scaler_rejects_unknown_names():
    with pytest.raises(ValueError, match="normalize"):
        make_scaler("minmax")  # type: ignore[arg-type]
