from fractions import Fraction
from typing import Literal, Protocol

import torch

from ._numeric import _round_fraction

#: The value normalization methods: ``"max"`` (:class:`MinMaxScaler`), ``"std"`` (:class:`StdScaler`),
#: ``"quantile"`` (:class:`QuantileScaler`) and ``"none"`` (:class:`NoneScaler`).
NormalizeType = Literal["max", "std", "quantile", "none"]


def _finite(values: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must be finite and representable in {values.dtype}; rescale the original data units.")
    return values


def _input(values: torch.Tensor, *, allow_integer: bool = False) -> None:
    integer = values.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    valid_dtype = values.is_floating_point() or (allow_integer and integer)
    if values.ndim != 2 or min(values.shape) < 1 or not valid_dtype:
        kind = "real numeric" if allow_integer else "floating"
        raise ValueError(f"values must be a nonempty {kind} tensor of shape (n_samples, n_features).")
    _finite(values, "values")


def _transform_input(values: torch.Tensor, offset: torch.Tensor) -> None:
    if values.ndim != 2:
        raise ValueError("values must be a 2-dimensional tensor of shape (n_samples, n_features).")
    if values.shape[1] != offset.shape[0]:
        raise ValueError(f"values must have {offset.shape[0]} features as fitted; got {values.shape[1]}.")
    _finite(values, "values")


def _integer_inverse(values: torch.Tensor, offset: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    dtype = torch.promote_types(torch.promote_types(values.dtype, offset.dtype), scale.dtype)
    values, offset, scale = torch.broadcast_tensors(values.to(dtype), offset.to(dtype), scale.to(dtype))
    bounds = torch.iinfo(dtype)
    product = values * scale
    risky = (values > bounds.max // scale) | (values < torch.div(bounds.min, scale, rounding_mode="trunc"))
    risky |= (product > bounds.max - offset.clamp_min(0)) | (product < bounds.min - offset.clamp_max(0))
    result = product + offset
    if bool(risky.any()):
        # Integer inverse inputs are uncommon. Exact arithmetic only for risky
        # entries preserves cancellation and the original integer output dtype.
        exact = [
            int(v) * int(s) + int(o)
            for v, s, o in zip(values[risky].tolist(), scale[risky].tolist(), offset[risky].tolist(), strict=True)
        ]
        if any(v < bounds.min or v > bounds.max for v in exact):
            raise ValueError("inverse-transformed integer values are not representable in the output dtype.")
        result[risky] = torch.tensor(exact, dtype=dtype, device=result.device)
    return result


def _integer_divide(values: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor | None = None) -> torch.Tensor:
    if offset is not None:
        dtype = torch.get_default_dtype()
        wide = torch.float64 if dtype == torch.float64 else torch.float32
        return ((values.to(wide) - offset.to(wide)) / scale.to(wide)).to(dtype)
    if torch.get_default_dtype() in (torch.float16, torch.bfloat16):
        # Half division needs floating opmath for large integer operands;
        # MPS also lacks the mixed long/half true-division kernel.
        return (values.to(torch.float32) / scale.to(torch.float32)).to(torch.get_default_dtype())
    return values / scale


def _transform(values: torch.Tensor, offset: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    _transform_input(values, offset)
    difference = values - offset
    integer = not difference.is_floating_point() and not difference.is_complex()
    result = _integer_divide(difference, scale) if integer else difference / scale
    if integer:
        bounds = torch.iinfo(difference.dtype)
        left, right = values.to(difference.dtype), offset.to(difference.dtype)
        risky = (left < bounds.min + right.clamp_min(0)) | (left > bounds.max + right.clamp_max(0))
        if bool(risky.any()):
            result = torch.where(risky, _integer_divide(values, scale, offset), result)
    if not torch.isfinite(result).all():
        fallback = _integer_divide(values, scale, offset) if integer else values / scale - offset / scale
        result = torch.where(torch.isfinite(result), result, fallback)
    if integer and not torch.isfinite(result).all():
        # Resolve the rare half overflow boundary without double rounding.
        selected = ~torch.isfinite(result)
        left, right, divisor = torch.broadcast_tensors(values, offset, scale)
        exact = [
            _round_fraction(Fraction(int(v) - int(o), int(s)), result.dtype)
            for v, o, s in zip(left[selected].tolist(), right[selected].tolist(), divisor[selected].tolist(), strict=True)
        ]
        result[selected] = torch.tensor(exact, dtype=result.dtype, device=result.device)
    return _finite(result, "transformed values")


def _inverse(values: torch.Tensor, offset: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    _transform_input(values, offset)
    if not any(t.is_floating_point() or t.is_complex() for t in (values, offset, scale)):
        return _integer_inverse(values, offset, scale)
    result = values * scale + offset
    if not torch.isfinite(result).all():
        fallback = (values + offset / scale) * scale
        result = torch.where(torch.isfinite(result), result, fallback)
    return _finite(result, "inverse-transformed values")


class Scaler(Protocol):
    """Protocol for data scalers with a fit/transform interface.

    All concrete scaler classes (:class:`MinMaxScaler`, :class:`StdScaler`,
    :class:`QuantileScaler`, :class:`NoneScaler`) satisfy this protocol via
    structural subtyping — no explicit inheritance is required.
    """

    def fit(self, values: torch.Tensor) -> None:
        """Fit the scaler to ``values``."""
        ...

    def fit_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Fit the scaler to ``values`` and return the transformed tensor."""
        ...

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """Transform ``values`` with the fitted statistics."""
        ...

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Reverse the transformation applied by :meth:`fit_transform`."""
        ...


class MinMaxScaler:
    """Min-max scaler that normalizes values to the [0, 1] range.

    Floating statistics and returned values must be finite in the input
    dtype. Integral inputs retain integer statistics and use PyTorch's default
    floating dtype for normalized values. An unrepresentable fitted range
    raises ``ValueError``; rescale the original data units first.

    Attributes
    ----------
    mins : torch.Tensor
        Minimum values per feature dimension.
    scales : torch.Tensor
        Scale factors (max - min) per feature dimension.
        Dimensions where max equals min are set to 1 to avoid division by zero.
    """

    mins: torch.Tensor
    scales: torch.Tensor

    def fit(self, values: torch.Tensor) -> None:
        """Fit the scaler to the data.

        Computes the per-feature minimum and scale (max - min) from ``values``.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor of shape ``(n_samples, n_features)``.
        """
        _input(values, allow_integer=True)
        mins = values.amin(dim=0)
        maxs = values.amax(dim=0)
        if not values.is_floating_point():
            # This comparison never subtracts endpoints or adds two positive
            # signed integers, so even a full int64 span cannot wrap.
            largest_range = torch.iinfo(values.dtype).max
            if bool((maxs > largest_range + mins.clamp_max(0)).any()):
                raise ValueError("fitted integer range (max - min) is not representable in the input dtype.")
        scales = _finite(maxs - mins, "fitted range (max - min)")
        scales[scales == 0.0] = 1.0  # Avoid division by zero
        self.mins = mins
        self.scales = scales

    def fit_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Fit the scaler to the data and return the normalized tensor.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor of shape ``(n_samples, n_features)``.

        Returns
        -------
        torch.Tensor
            Normalized tensor of the same shape as ``values``,
            with each feature scaled to the [0, 1] range.
        """
        self.fit(values)
        return self.transform(values)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """Normalize ``values`` with the fitted minimum and scale.

        Parameters
        ----------
        values : torch.Tensor
            Tensor of shape ``(n_samples, n_features)`` with the fitted feature count.
            Empty batches (zero samples) are accepted.

        Returns
        -------
        torch.Tensor
            Transformed tensor of the same shape as ``values``.

        Raises
        ------
        ValueError
            If the input is not two-dimensional, has a different feature count
            from the fitted data, or contains non-finite values or results.
        """
        return _transform(values, self.mins, self.scales)

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Reverse the normalization applied by :meth:`fit_transform`.

        Parameters
        ----------
        values : torch.Tensor
            Normalized tensor of shape ``(n_samples, n_features)`` with the fitted feature count.
            Empty batches (zero samples) are accepted.

        Returns
        -------
        torch.Tensor
            Tensor rescaled back to the original value range.

        Raises
        ------
        ValueError
            If the input is not two-dimensional, has a different feature count
            from the fitted data, or contains non-finite values or results.
        """
        return _inverse(values, self.mins, self.scales)


class StdScaler:
    """Standard-score (z-score) scaler that normalizes values to zero mean and unit variance.

    Statistics use a scaled fallback to avoid intermediate overflow. An
    unrepresentable standard deviation or result raises ``ValueError``.

    Attributes
    ----------
    means : torch.Tensor
        Per-feature means computed during :meth:`fit`.
    stds : torch.Tensor
        Per-feature standard deviations computed during :meth:`fit`.
        Dimensions with zero standard deviation are set to 1 to avoid division by zero.
    """

    means: torch.Tensor
    stds: torch.Tensor

    def fit(self, values: torch.Tensor) -> None:
        """Fit the scaler to the data.

        Computes the per-feature mean and standard deviation from ``values``.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor of shape ``(n_samples, n_features)``.
        """
        _input(values)
        if values.shape[0] == 1:
            stds, means = torch.ones_like(values[0]), values[0].clone()
        else:
            stds, means = torch.std_mean(values, dim=0)
            retry = ~torch.isfinite(stds) | ~torch.isfinite(means) | ((stds == 0) & (values.amax(0) != values.amin(0)))
            if retry.any():
                # Center first when differences fit: dividing a large offset
                # before centering would erase precision in a narrow spread.
                delta = values - values[0]
                centered = torch.isfinite(delta).all(dim=0)
                unit = torch.where(centered, delta.abs().amax(0), values.abs().amax(0))
                unit = unit.clamp_min(torch.finfo(values.dtype).tiny)
                normalized = torch.where(centered, delta / unit, values / unit)
                scaled_stds, scaled_means = torch.std_mean(normalized, dim=0)
                recovered_means = scaled_means * unit + torch.where(centered, values[0], 0)
                stds = torch.where(retry, scaled_stds * unit, stds)
                means = torch.where(retry, recovered_means, means)
        _finite(stds, "fitted standard deviation")
        _finite(means, "fitted mean")
        if ((stds == 0) & (values.amax(0) != values.amin(0))).any():
            raise ValueError("fitted standard deviation is too small to represent; rescale the original data units.")
        self.stds = torch.where(stds > 0, stds, torch.ones_like(stds))
        self.means = means

    def fit_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Fit the scaler to the data and return the standardized tensor.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor of shape ``(n_samples, n_features)``.

        Returns
        -------
        torch.Tensor
            Standardized tensor of the same shape as ``values``,
            with each feature having zero mean and unit standard deviation.
        """
        self.fit(values)
        return self.transform(values)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """Standardize ``values`` with the fitted mean and standard deviation.

        Parameters
        ----------
        values : torch.Tensor
            Tensor of shape ``(n_samples, n_features)`` with the fitted feature count.
            Empty batches (zero samples) are accepted.

        Returns
        -------
        torch.Tensor
            Transformed tensor of the same shape as ``values``.

        Raises
        ------
        ValueError
            If the input is not two-dimensional, has a different feature count
            from the fitted data, or contains non-finite values or results.
        """
        return _transform(values, self.means, self.stds)

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Reverse the standardization applied by :meth:`fit_transform`.

        Parameters
        ----------
        values : torch.Tensor
            Standardized tensor of shape ``(n_samples, n_features)`` with the fitted feature count.
            Empty batches (zero samples) are accepted.

        Returns
        -------
        torch.Tensor
            Tensor rescaled back to the original value range.

        Raises
        ------
        ValueError
            If the input is not two-dimensional, has a different feature count
            from the fitted data, or contains non-finite values or results.
        """
        return _inverse(values, self.means, self.stds)


class QuantileScaler:
    """Quantile-based scaler that normalizes values using percentile-based ranges.

    Values are scaled so that the ``q``-th percentile maps to 0 and the
    ``(1 - q)``-th percentile maps to 1, without clipping values to this range.
    This makes the scaler robust to outliers compared to :class:`MinMaxScaler`.
    Non-finite input, an unrepresentable fitted range, or an unrepresentable
    transformed result raises ``ValueError``. Rescale original units first.

    Attributes
    ----------
    q : float
        The lower quantile fraction used as the effective minimum.
        Defaults to ``0.05`` (5th percentile), ignoring the bottom and top 5% of values.
    mins : torch.Tensor
        Per-feature values at the ``q``-th quantile, computed during :meth:`fit`.
    scales : torch.Tensor
        Per-feature scale factors (``(1-q)``-quantile minus ``q``-quantile).
        Dimensions where the scale is zero are set to 1 to avoid division by zero.
    """

    q: float = 0.05  # Ignore 5% outliers
    mins: torch.Tensor
    scales: torch.Tensor

    def fit(self, values: torch.Tensor) -> None:
        """Fit the scaler to the data.

        Computes per-feature quantile bounds from ``values``.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor of shape ``(n_samples, n_features)``.
        """
        _input(values)
        mins = values.quantile(self.q, dim=0)
        maxs = values.quantile(1.0 - self.q, dim=0)
        if not (torch.isfinite(mins).all() and torch.isfinite(maxs).all()):
            unit = values.abs().amax(dim=0).clamp_min(torch.finfo(values.dtype).tiny)
            scaled = values / unit
            mins = scaled.quantile(self.q, dim=0) * unit
            maxs = scaled.quantile(1.0 - self.q, dim=0) * unit
        _finite(mins, "fitted lower quantile")
        _finite(maxs, "fitted upper quantile")
        scales = _finite(maxs - mins, "fitted range (max - min)")
        scales[scales == 0.0] = 1.0  # Avoid division by zero
        self.mins = mins
        self.scales = scales

    def fit_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Fit the scaler to the data and return the scaled tensor.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor of shape ``(n_samples, n_features)``.

        Returns
        -------
        torch.Tensor
            Scaled tensor of the same shape as ``values``.
            Values outside the fitted quantile range may exceed [0, 1].
        """
        self.fit(values)
        return self.transform(values)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """Scale ``values`` with the fitted quantile bounds.

        Parameters
        ----------
        values : torch.Tensor
            Tensor of shape ``(n_samples, n_features)`` with the fitted feature count.
            Empty batches (zero samples) are accepted.

        Returns
        -------
        torch.Tensor
            Transformed tensor of the same shape as ``values``.

        Raises
        ------
        ValueError
            If the input is not two-dimensional, has a different feature count
            from the fitted data, or contains non-finite values or results.
        """
        return _transform(values, self.mins, self.scales)

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Reverse the scaling applied by :meth:`fit_transform`.

        Parameters
        ----------
        values : torch.Tensor
            Scaled tensor of shape ``(n_samples, n_features)`` with the fitted feature count.
            Empty batches (zero samples) are accepted.

        Returns
        -------
        torch.Tensor
            Tensor rescaled back to the original value range.

        Raises
        ------
        ValueError
            If the input is not two-dimensional, has a different feature count
            from the fitted data, or contains non-finite values or results.
        """
        return _inverse(values, self.mins, self.scales)


class NoneScaler:
    """Pass-through scaler that leaves values unchanged.

    Useful as a no-op placeholder when normalization is not required,
    while still providing the same :meth:`fit`, :meth:`fit_transform`,
    and :meth:`inverse_transform` interface as the other scalers.
    """

    def fit(self, values: torch.Tensor) -> None:
        """No-op fit method included for API compatibility.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor (ignored).
        """
        pass

    def fit_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``values`` unchanged.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor of any shape.

        Returns
        -------
        torch.Tensor
            The same tensor as ``values`` without any modification.
        """
        return values

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``values`` unchanged.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor of any shape.

        Returns
        -------
        torch.Tensor
            The same tensor as ``values`` without any modification.
        """
        return values

    def inverse_transform(self, values: torch.Tensor) -> torch.Tensor:
        """Return ``values`` unchanged.

        Parameters
        ----------
        values : torch.Tensor
            Input tensor of any shape.

        Returns
        -------
        torch.Tensor
            The same tensor as ``values`` without any modification.
        """
        return values


def make_scaler(normalize: NormalizeType) -> Scaler:
    """The scaler of a normalization method.

    Every scaler is an affine map of each feature, so it commutes with Bézier
    simploids, whose Bernstein basis sums to one: transforming the control
    points transforms the values of the model.

    Parameters
    ----------
    normalize
        ``"max"``, ``"std"``, ``"quantile"`` or ``"none"``.

    Returns
    -------
    A new, unfitted scaler.

    Raises
    ------
    ValueError
        If ``normalize`` is not one of the methods.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf.preprocessing import make_scaler
    >>> scaler = make_scaler("max")
    >>> scaler.fit_transform(torch.tensor([[1.0], [3.0]])).tolist()
    [[0.0], [1.0]]
    """
    scalers: dict[str, type[Scaler]] = {
        "max": MinMaxScaler,
        "std": StdScaler,
        "quantile": QuantileScaler,
        "none": NoneScaler,
    }
    if normalize not in scalers:
        raise ValueError(f"{normalize=}. Must be one of {list(scalers)}.")
    return scalers[normalize]()
