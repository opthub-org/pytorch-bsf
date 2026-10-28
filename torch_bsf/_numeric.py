"""Small numerical reductions shared by prediction and initialization."""

import torch


def _finite_mean(values: torch.Tensor, dim: int, *, keepdim: bool = False) -> torch.Tensor:
    """Avoid overflowing a finite mean, preserving small values on the ordinary path."""
    detached = values.detach()
    scale = detached.abs().amax(dim=dim, keepdim=True).clamp_min(1)
    # Normalize only groups whose sum could overflow. Small groups keep their
    # native sum/division, so a subnormal value is never divided before summing.
    risky = scale >= torch.finfo(values.dtype).max / values.shape[dim]
    scale = torch.where(risky, scale, torch.ones_like(scale))
    safe = (detached / scale).mean(dim=dim, keepdim=True) * scale
    if not keepdim:
        safe = safe.squeeze(dim)
    # Supply only the safe forward value. The zero-valued ordinary-mean path
    # divides upstream gradients by count directly, without first multiplying
    # by an extreme scale (which can overflow even for a finite derivative).
    return safe + (values - detached).mean(dim=dim, keepdim=keepdim)
