"""Small numerical reductions shared by prediction and initialization."""

from fractions import Fraction
from math import frexp, ldexp

import torch


def _round_fraction(value: Fraction, dtype: torch.dtype) -> float:
    """Round once to the target binary format, including subnormal ties to even."""
    if value == 0:
        return 0.0
    numerator, denominator = abs(value.numerator), value.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    below_power = numerator < denominator << exponent if exponent >= 0 else numerator << -exponent < denominator
    if below_power:
        exponent -= 1
    info = torch.finfo(dtype)
    # eps = 2**(1-p); the smallest quantum is tiny * eps.
    precision = 2 - frexp(info.eps)[1]
    quantum = max(exponent - precision + 1, frexp(info.tiny)[1] - precision)
    if quantum >= 0:
        denominator <<= quantum
    else:
        numerator <<= -quantum
    significand, remainder = divmod(numerator, denominator)
    if 2 * remainder > denominator or (2 * remainder == denominator and significand % 2):
        significand += 1
    rounded = ldexp(float(significand), quantum)
    return -rounded if value < 0 else rounded


def _exact_means(values: torch.Tensor) -> torch.Tensor:
    """Evaluate finite column means with exact rationals in a rare CPU fallback."""
    columns = values.detach().cpu().T.tolist()
    means = [
        _round_fraction(sum((Fraction.from_float(x) for x in column), Fraction(0)) / len(column), values.dtype)
        for column in columns
    ]
    return torch.tensor(means, dtype=values.dtype, device=values.device)


def _finite_mean(values: torch.Tensor, dim: int, *, keepdim: bool = False) -> torch.Tensor:
    """Avoid overflowing finite means without scaling away cancelling residuals.

    Ordinary groups retain their vectorized device reduction. Finite groups
    near the sum-overflow boundary use an exact CPU rational reduction, which
    transfers only those groups. The risk check synchronizes accelerators.
    The gradient
    follows the ordinary mean rather than extreme intermediate scales.
    """
    detached = values.detach()
    magnitude = detached.abs().amax(dim=dim, keepdim=True)
    # A factor-two margin covers rounding near the sum-overflow boundary.
    risky = (magnitude >= torch.finfo(values.dtype).max / (2 * values.shape[dim])) & torch.isfinite(detached).all(
        dim=dim, keepdim=True
    )
    safe = detached.mean(dim=dim, keepdim=keepdim)
    if risky.any():
        columns = detached.movedim(dim, 0).reshape(values.shape[dim], -1)
        selected = risky.reshape(-1)
        safe = safe.clone().contiguous()
        safe.reshape(-1)[selected] = _exact_means(columns[:, selected])
    # A zero-valued ordinary-mean path supplies 1/count gradients without any
    # multiplication by an extreme scale. Mask nonfinite predictions before
    # forming that zero so native Inf/NaN forward results remain unchanged.
    finite_values = torch.where(torch.isfinite(detached), values, 0)
    return safe + (finite_values - finite_values.detach()).mean(dim=dim, keepdim=keepdim)
