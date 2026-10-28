"""Affine scaler contracts over generated feature tables."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from tests.strategies import U, extreme_scalings, feature_values, matrices, simplex_points
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("scaler_class", [MinMaxScaler, StdScaler, QuantileScaler])
@pytest.mark.parametrize("kind", ["constant", "signed", "tiny", "offset", "single"])
def test_extreme_scaler_contract(scaler_class, dtype, kind):
    magnitude = torch.finfo(dtype).max / 2
    tiny = torch.finfo(dtype).tiny
    table = {
        "constant": [magnitude, magnitude],
        "signed": [-magnitude, magnitude],
        "tiny": [tiny, 2 * tiny],
        "offset": [magnitude, magnitude * (1 - 8 * torch.finfo(dtype).eps)],
        "single": [magnitude],
    }
    values = torch.tensor(table[kind], dtype=dtype)[:, None]
    scaler = scaler_class()
    normalized = scaler.fit_transform(values)
    assert torch.isfinite(normalized).all()
    restored = scaler.inverse_transform(normalized)
    # At most 16 rounded arithmetic operations on either affine evaluation;
    # the bound is relative to the input units, including cancellation.
    bound = 16 * torch.finfo(dtype).eps * values.abs().amax()
    assert ((restored - values).abs() <= bound).all()


@pytest.mark.parametrize("scaler_class", [MinMaxScaler, QuantileScaler])
def test_unrepresentable_single_fitted_range(scaler_class):
    scaler = scaler_class()
    with pytest.raises(ValueError, match="fitted range.*rescale"):
        scaler.fit_transform(torch.tensor([[-1e308], [1e308]], dtype=torch.float64))


def test_issue_452_std_extreme_signed_values():
    values = torch.tensor([[-1e308], [1e308]], dtype=torch.float64)
    scaler = StdScaler()
    z = scaler.fit_transform(values)
    assert torch.isfinite(z).all()
    # Two samples have sample standard deviation sqrt(2)*1e308.
    expected = torch.tensor([[-(2**-0.5)], [2**-0.5]], dtype=torch.float64)
    torch.testing.assert_close(z, expected, atol=0, rtol=8 * U)
    torch.testing.assert_close(scaler.inverse_transform(z), values, atol=0, rtol=8 * U)


@pytest.mark.parametrize("scaler_class", [MinMaxScaler, StdScaler, QuantileScaler])
def test_scaler_rejects_nonfinite_and_unrepresentable_results(scaler_class):
    scaler = scaler_class()
    with pytest.raises(ValueError, match="finite"):
        scaler.fit(torch.tensor([[float("nan")], [1.0]], dtype=torch.float64))
    scaler.fit(torch.tensor([[0.0], [2.0]], dtype=torch.float64))
    with pytest.raises(ValueError, match="inverse-transformed"):
        scaler.inverse_transform(torch.tensor([[torch.finfo(torch.float64).max]], dtype=torch.float64))


def test_affine_cancellation_avoids_intermediate_overflow():
    scaler = MinMaxScaler()
    scaler.fit(torch.tensor([[-1e308], [0.0]], dtype=torch.float64))
    values = torch.tensor([[1e308]], dtype=torch.float64)
    assert scaler.transform(values).item() == 2
    assert scaler.inverse_transform(torch.tensor([[2.0]], dtype=torch.float64)).item() == 1e308


@pytest.mark.parametrize("scaler_class", [MinMaxScaler, StdScaler, QuantileScaler])
@given(unit=extreme_scalings())
def test_scaler_power_of_two_equivariance(scaler_class, unit):
    values = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
    base = scaler_class().fit_transform(values)
    scaler = scaler_class()
    transformed = scaler.fit_transform(values * unit)
    # Exact power-of-two rescaling commutes with the affine map up to the
    # 16 rounded operations used by the scaled-statistics fallback.
    torch.testing.assert_close(transformed, base, atol=16 * U, rtol=16 * U)
    restored = scaler.inverse_transform(transformed) / unit
    torch.testing.assert_close(restored, values, atol=16 * U, rtol=16 * U)


def test_mixed_columns_retry_only_nonfinite_affine_results():
    scaler = MinMaxScaler()
    scaler.mins = torch.tensor([-1e308, 1e308], dtype=torch.float64)
    scaler.scales = torch.tensor([1e308, 1e-300], dtype=torch.float64)
    z = torch.tensor([[2.0, 1.0]], dtype=torch.float64)
    restored = scaler.inverse_transform(z)
    torch.testing.assert_close(restored, torch.tensor([[1e308, 1e308]], dtype=torch.float64), atol=0, rtol=0)


def test_std_centered_extreme_offset_preserves_spread():
    values = torch.tensor([[1e308], [1e308]], dtype=torch.float64)
    values[1] = torch.nextafter(values[0], torch.tensor(0.0, dtype=torch.float64))
    scaler = StdScaler()
    scaler.fit(values)
    # The two-point sample standard deviation is |x1-x0| / sqrt(2).
    expected = (values[0] - values[1]) / 2**0.5
    torch.testing.assert_close(scaler.stds, expected, atol=0, rtol=8 * U)


@pytest.mark.parametrize("scaler_class", [MinMaxScaler, StdScaler, QuantileScaler])
@pytest.mark.parametrize("shape", [(), (3,), (0, 1), (1, 0)])
def test_scalers_reject_non_tabular_or_empty_input(scaler_class, shape):
    values = torch.empty(shape, dtype=torch.float64)
    kind = "real numeric" if scaler_class is MinMaxScaler else "floating"
    with pytest.raises(ValueError, match=f"nonempty {kind} tensor.*n_samples, n_features"):
        scaler_class().fit(values)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_std_rejects_underflowed_deviation_of_varying_column(dtype):
    smallest = torch.nextafter(torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype))
    values = torch.zeros((5, 1), dtype=dtype)
    values[-1] = smallest
    # The nonzero sample standard deviation is smallest/sqrt(5), which rounds
    # to zero; treating this varying column as constant would skip scaling.
    with pytest.raises(ValueError, match="standard deviation.*too small.*rescale"):
        StdScaler().fit(values)
    constant = torch.full((5, 1), smallest.item(), dtype=dtype)
    scaler = StdScaler()
    normalized = scaler.fit_transform(constant)
    torch.testing.assert_close(normalized, torch.zeros_like(constant), atol=0, rtol=0)
    torch.testing.assert_close(scaler.inverse_transform(normalized), constant, atol=0, rtol=0)


@pytest.mark.parametrize("integer_dtype", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64])
@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
def test_minmax_preserves_integer_statistics_and_default_output_dtype(integer_dtype, default_dtype):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        maximum = torch.iinfo(integer_dtype).max
        values = torch.tensor([[maximum - 1, 7], [maximum, 7]], dtype=integer_dtype)
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(values)
        torch.testing.assert_close(normalized, torch.tensor([[0.0, 0.0], [1.0, 0.0]]), atol=0, rtol=0)
        assert normalized.dtype == default_dtype
        torch.testing.assert_close(scaler.mins, values[0], atol=0, rtol=0)
        torch.testing.assert_close(scaler.scales, torch.ones(2, dtype=integer_dtype), atol=0, rtol=0)
        # Inverse output retains the original floating result semantics; large
        # offsets can round together in that dtype, even though normalization
        # preserved their integer difference before division.
        torch.testing.assert_close(scaler.inverse_transform(normalized), values.to(default_dtype), atol=0, rtol=0)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("offset", [10**10, torch.iinfo(torch.int64).max - 2])
def test_minmax_subtracts_large_integer_offsets_before_dividing(offset):
    values = torch.tensor([[offset + 1], [offset + 2]], dtype=torch.int64)
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(values)
    expected = torch.tensor([[0.0], [1.0]], dtype=torch.get_default_dtype())
    torch.testing.assert_close(normalized, expected, atol=0, rtol=0)
    assert scaler.mins.dtype == scaler.scales.dtype == torch.int64


@pytest.mark.parametrize("dtype", [torch.bool, torch.complex64, torch.complex128])
def test_minmax_still_rejects_unsupported_nonreal_or_boolean_input(dtype):
    with pytest.raises(ValueError, match="nonempty.*tensor"):
        MinMaxScaler().fit(torch.ones((2, 1), dtype=dtype))


@pytest.mark.parametrize("scaler_class", [StdScaler, QuantileScaler])
def test_other_scalers_still_require_floating_input(scaler_class):
    with pytest.raises(ValueError, match="nonempty floating tensor"):
        scaler_class().fit(torch.tensor([[1], [3]], dtype=torch.int64))


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_minmax_rejects_unrepresentable_signed_integer_ranges(dtype):
    bounds = torch.iinfo(dtype)
    for endpoints in ((bounds.min, 0, bounds.max), (-1, 0, bounds.max), (bounds.min, -1, 0)):
        with pytest.raises(ValueError, match="fitted integer range.*not representable"):
            MinMaxScaler().fit(torch.tensor(endpoints, dtype=dtype).reshape(-1, 1))


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64])
def test_minmax_accepts_largest_representable_signed_integer_range(dtype):
    bounds = torch.iinfo(dtype)
    for endpoints in ((bounds.min, -1), (-1, bounds.max - 1), (0, bounds.max)):
        values = torch.tensor(endpoints, dtype=dtype).reshape(-1, 1)
        scaler = MinMaxScaler()
        torch.testing.assert_close(scaler.fit_transform(values), torch.tensor([[0.0], [1.0]]), atol=0, rtol=0)
        torch.testing.assert_close(scaler.scales, torch.tensor([bounds.max], dtype=dtype), atol=0, rtol=0)
