"""Small numerical reductions shared by prediction and initialization."""

import torch


def _finite_mean(values: torch.Tensor, dim: int, *, keepdim: bool = False) -> torch.Tensor:
    """Avoid overflowing a finite mean, preserving small values on the ordinary path."""
    scale = values.detach().abs().amax(dim=dim, keepdim=True).clamp_min(1)
    # Normalize only groups whose sum could overflow. Small groups keep their
    # native sum/division, so a subnormal value is never divided before summing.
    risky = scale >= torch.finfo(values.dtype).max / values.shape[dim]
    scale = torch.where(risky, scale, torch.ones_like(scale))
    result = (values / scale).mean(dim=dim, keepdim=True) * scale
    return result if keepdim else result.squeeze(dim)
