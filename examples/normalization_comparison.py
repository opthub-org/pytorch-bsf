"""Compare explicit output scaling for unregularized least squares.

Core regression has no implicit transform: the application owns output units.
"""

import torch

from torch_bsf import BezierSpace, Samples, fit
from torch_bsf.sampling import grid


def main() -> None:
    space = BezierSpace.simplex(2, degree=2)
    points = grid(space, resolution=9)
    values = torch.stack([points.parameters[:, 0] ** 2, 1000 * points.parameters[:, 1]], 1)
    scale = values.abs().amax(0).clamp_min(1)
    raw = fit(space, Samples(points, values)).model(points)
    scaled = fit(space, Samples(points, values / scale)).model(points) * scale
    print("Maximum difference after explicit inverse scaling:", (raw - scaled).abs().max().item())
    assert torch.allclose(raw, scaled, atol=1e-9)


if __name__ == "__main__":
    main()
