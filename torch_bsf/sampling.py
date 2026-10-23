"""Domain-aware random and deterministic sampling, using local random generators."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import torch
from torch.nn import functional as F

from torch_bsf._kernels.combinatorics import _indices
from torch_bsf.core import BezierSpace, Points, _integer
from torch_bsf.geometry import Simploid


def _generator(seed: int | None) -> torch.Generator:
    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    return generator


def _probabilities(space: BezierSpace, weights: Mapping[int | Simploid, float] | None) -> torch.Tensor:
    probability = torch.zeros(space.n_cells, dtype=torch.float64)
    if weights is None:
        probability[list(space.main_cells)] = 1.0
    else:
        for cell, weight in weights.items():
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("cell_weights must be finite and nonnegative.")
            probability[space.cell_id(cell)] += weight
    if not torch.isfinite(probability.sum()) or not probability.sum() > 0:
        raise ValueError("cell_weights must have a positive sum.")
    return probability / probability.sum()


def sample(
    space: BezierSpace,
    n: int,
    *,
    method: Literal["random", "sobol"] = "random",
    seed: int | None = None,
    cell_weights: Mapping[int | Simploid, float] | None = None,
) -> Points:
    """Sample a space, uniformly within each selected cell's simplex factors.

    Cell probabilities default to equal weights on maximal cells, rather than
    an unspecified geometric volume. Sobol coordinates use exponential
    normalization of scrambled quasi-random uniforms. Cell selection remains
    categorical random sampling. Neither method resets global random state.

    Examples
    --------
    >>> from torch_bsf import BezierSpace
    >>> from torch_bsf.sampling import sample
    >>> sample(BezierSpace.simplex(2, degree=3), 5, seed=0).parameters.shape
    torch.Size([5, 3])
    """
    _integer(n, "n")
    if method not in ("random", "sobol"):
        raise ValueError("method must be 'random' or 'sobol'.")
    generator = _generator(seed)
    probability = _probabilities(space, cell_weights)
    ids = torch.multinomial(probability, n, replacement=True, generator=generator) if n else torch.empty(0, dtype=torch.long)
    parameters = torch.zeros(n, space.width, dtype=torch.float64)
    for cid in torch.unique(ids).tolist():
        info = space.cells[cid]
        rows = torch.nonzero(ids == cid).flatten()
        if not info.width:
            continue
        if method == "random":
            uniforms = torch.rand(len(rows), info.width, generator=generator, dtype=torch.float64)
        else:
            local_seed = int(torch.randint(2**31, (1,), generator=generator))
            uniforms = torch.quasirandom.SobolEngine(info.width, scramble=True, seed=local_seed).draw(
                len(rows), dtype=torch.float64
            )
        exponentials = -uniforms.clamp_min(torch.finfo(torch.float64).tiny).log()
        offset = 0
        for dimension in info.dimensions:
            block = exponentials[:, offset : offset + dimension + 1]
            parameters[rows, offset : offset + dimension + 1] = block / block.sum(1, keepdim=True)
            offset += dimension + 1
    return Points(parameters, ids)


def grid(space: BezierSpace, resolution: int = 10, *, cells: Sequence[int | Simploid] | None = None) -> Points:
    """Create tensor-product barycentric grids on the selected cells.

    ``resolution=0`` gives a cell's barycenter. Boundary points of different
    cells remain distinct parameter-domain points even when their images agree.
    """
    _integer(resolution, "resolution")
    selected = list(space.main_cells) if cells is None else [space.cell_id(c) for c in cells]
    coordinates, identifiers = [], []
    for cid in selected:
        info = space.cells[cid]
        if resolution:
            indices = _indices(info.dimensions, (resolution,) * len(info.dimensions))
            t = torch.tensor([[v / resolution for factor in alpha for v in factor] for alpha in indices], dtype=torch.float64)
            t = t.reshape(len(indices), info.width)
        else:
            t = torch.tensor([[1 / (d + 1) for d in info.dimensions for _ in range(d + 1)]], dtype=torch.float64)
        coordinates.append(F.pad(t, (0, space.width - info.width)))
        identifiers.append(torch.full((len(t),), cid, dtype=torch.long))
    if not coordinates:
        return Points(torch.empty(0, space.width), torch.empty(0, dtype=torch.long))
    return Points(torch.cat(coordinates), torch.cat(identifiers))
