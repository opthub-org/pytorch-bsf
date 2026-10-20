import pytest
import torch

from torch_bsf.preprocessing import MinMaxScaler, NoneScaler, QuantileScaler, StdScaler, make_scaler


@pytest.fixture
def simple_values():
    return torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])


class TestMinMaxScaler:
    def test_fit_transform_range(self, simple_values):
        scaler = MinMaxScaler()
        result = scaler.fit_transform(simple_values)
        assert torch.allclose(result.min(dim=0).values, torch.zeros(2))
        assert torch.allclose(result.max(dim=0).values, torch.ones(2))

    def test_fit_transform_zero_scale(self):
        # All values in a column are identical => scale should be 1 (no division by zero)
        values = torch.tensor([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        scaler = MinMaxScaler()
        result = scaler.fit_transform(values)
        # Constant column should stay constant (divided by 1)
        assert torch.allclose(result[:, 0], torch.zeros(3))

    def test_inverse_transform_roundtrip(self, simple_values):
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(simple_values)
        recovered = scaler.inverse_transform(scaled)
        assert torch.allclose(recovered, simple_values, atol=1e-5)

    def test_fit_is_consistent_with_fit_transform(self, simple_values):
        # fit() must set up state that is equivalent to what fit_transform() uses
        scaler_a = MinMaxScaler()
        scaled_a = scaler_a.fit_transform(simple_values)

        scaler_b = MinMaxScaler()
        scaler_b.fit(simple_values)
        recovered = scaler_b.inverse_transform(scaled_a)
        assert torch.allclose(recovered, simple_values, atol=1e-6)

    def test_single_row(self):
        values = torch.tensor([[3.0, 7.0]])
        scaler = MinMaxScaler()
        result = scaler.fit_transform(values)
        # Single row: min == max => scale=1, output is 0
        assert torch.allclose(result, torch.zeros(1, 2))


class TestStdScaler:
    def test_fit_transform_zero_mean_unit_std(self, simple_values):
        scaler = StdScaler()
        result = scaler.fit_transform(simple_values)
        assert torch.allclose(result.mean(dim=0), torch.zeros(2), atol=1e-5)
        assert torch.allclose(result.std(dim=0), torch.ones(2), atol=1e-4)

    def test_zero_std_column(self):
        # Constant column => std=0 => replaced by 1 to avoid division by zero
        values = torch.tensor([[2.0, 1.0], [2.0, 2.0], [2.0, 3.0]])
        scaler = StdScaler()
        result = scaler.fit_transform(values)
        assert torch.allclose(result[:, 0], torch.zeros(3))

    def test_single_sample(self):
        # The corrected std of a single sample is NaN => replaced by 1
        scaler = StdScaler()
        result = scaler.fit_transform(torch.tensor([[2.0, 1.0]]))
        assert torch.equal(result, torch.zeros(1, 2))

    def test_inverse_transform_roundtrip(self, simple_values):
        scaler = StdScaler()
        scaled = scaler.fit_transform(simple_values)
        recovered = scaler.inverse_transform(scaled)
        assert torch.allclose(recovered, simple_values, atol=1e-5)


class TestQuantileScaler:
    def test_fit_transform_shape(self, simple_values):
        scaler = QuantileScaler()
        result = scaler.fit_transform(simple_values)
        assert result.shape == simple_values.shape

    def test_zero_scale(self):
        # All rows identical => quantile range is zero => scale replaced by 1
        values = torch.tensor([[4.0, 4.0], [4.0, 4.0], [4.0, 4.0]])
        scaler = QuantileScaler()
        result = scaler.fit_transform(values)
        assert torch.allclose(result, torch.zeros(3, 2))

    def test_inverse_transform_roundtrip(self, simple_values):
        scaler = QuantileScaler()
        scaled = scaler.fit_transform(simple_values)
        recovered = scaler.inverse_transform(scaled)
        assert torch.allclose(recovered, simple_values, atol=1e-4)


class TestNoneScaler:
    def test_fit_transform_identity(self, simple_values):
        scaler = NoneScaler()
        result = scaler.fit_transform(simple_values)
        assert torch.equal(result, simple_values)

    def test_inverse_transform_identity(self, simple_values):
        scaler = NoneScaler()
        scaler.fit(simple_values)
        result = scaler.inverse_transform(simple_values)
        assert torch.equal(result, simple_values)


@pytest.mark.parametrize("scaler_class", [MinMaxScaler, StdScaler, QuantileScaler, NoneScaler])
def test_transform_applies_the_fitted_scale_to_new_values(scaler_class, simple_values):
    """transform() reuses the scale fitted by fit(), and inverse_transform() undoes it."""
    scaler = scaler_class()
    torch.testing.assert_close(scaler.fit_transform(simple_values), scaler.transform(simple_values))
    new_values = torch.tensor([[0.0, 100.0], [4.0, -5.0]])
    scaled = scaler.transform(new_values)
    torch.testing.assert_close(scaler.inverse_transform(scaled), new_values)
    # The scale is affine, so transform() commutes with the convex combinations of a Bézier simploid.
    w = torch.tensor([[0.25, 0.75]])
    torch.testing.assert_close(scaler.transform(w @ new_values), w @ scaled)


@pytest.mark.parametrize(
    "normalize, scaler_class",
    [("max", MinMaxScaler), ("std", StdScaler), ("quantile", QuantileScaler), ("none", NoneScaler)],
)
def test_make_scaler(normalize, scaler_class):
    assert isinstance(make_scaler(normalize), scaler_class)


def test_make_scaler_rejects_unknown_names():
    with pytest.raises(ValueError, match="normalize"):
        make_scaler("minmax")  # type: ignore[arg-type]
