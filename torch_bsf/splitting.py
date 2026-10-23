"""Exact local subdivision with explicit maps back to the parent domain."""

from dataclasses import dataclass
from typing import Literal

import torch

from torch_bsf._kernels._splitting import longest_edge_criterion, max_error_criterion, reparametrize, split
from torch_bsf._kernels.bezier import BezierSimploid
from torch_bsf.core import BezierModel, BezierSpace, Points, Samples


@dataclass(frozen=True)
class EdgeSplit:
    """Split one factor edge at a fraction strictly between zero and one."""

    i: int
    j: int
    fraction: float = 0.5
    factor: int = 0


@dataclass(frozen=True)
class LongestEdge:
    """Select the longest chord of the selected cell's control-net vertices."""

    fraction: float = 0.5


@dataclass(frozen=True)
class ResidualSplit:
    """Minimize the sum of unweighted child MSEs over a split-fraction grid.

    Exact subdivision preserves the represented function. This policy balances
    residual grouping; it does not refit the children or guarantee error reduction.
    """

    samples: Samples
    resolution: int = 10


@dataclass(frozen=True)
class ParameterMap:
    """An affine child-to-parent map and its partial inverse."""

    parent_cell: int
    dimensions: tuple[int, ...]
    split: EdgeSplit
    part: Literal["A", "B"]

    def to_parent(self, points: Points | torch.Tensor) -> Points:
        """Map child barycentric points to their parent cell."""
        batch = points if isinstance(points, Points) else Points(points)
        # Validate against the child's domain, including its sole cell ID.
        BezierSpace.simploid(self.dimensions, degree=1).validate(batch)
        t = batch.parameters.clone()
        offset = sum(a + 1 for a in self.dimensions[: self.split.factor])
        i, j, s = offset + self.split.i, offset + self.split.j, self.split.fraction
        if self.part == "A":
            t[:, i] = batch.parameters[:, i] + (1 - s) * batch.parameters[:, j]
            t[:, j] = s * batch.parameters[:, j]
        else:
            t[:, i] = (1 - s) * batch.parameters[:, i]
            t[:, j] = s * batch.parameters[:, i] + batch.parameters[:, j]
        return Points(t, torch.full((len(t),), self.parent_cell, device=t.device, dtype=torch.long))

    def from_parent(self, points: Points) -> tuple[Points, torch.Tensor]:
        """Return child coordinates and a mask selecting parent rows in this piece."""
        s = self.split
        eligible = torch.nonzero(points.cell_ids == self.parent_cell).flatten()
        width = sum(d + 1 for d in self.dimensions)
        transformed, local_mask = reparametrize(
            points.parameters[eligible, :width],
            s.i,
            s.j,
            s.fraction,
            self.part,
            simploid_type=self.dimensions,
            factor=s.factor,
        )
        mask = torch.zeros(len(points), device=points.cell_ids.device, dtype=torch.bool)
        mask[eligible] = local_mask
        return Points(transformed[local_mask]), mask


@dataclass(frozen=True)
class SubdivisionResult:
    """Two exact local models and their parent parameter maps.

    This does not remesh neighboring cells of an assembled domain.
    """

    models: tuple[BezierModel, BezierModel]
    maps: tuple[ParameterMap, ParameterMap]


def subdivide(
    model: BezierModel, *, cell: int = 0, policy: EdgeSplit | LongestEdge | ResidualSplit = LongestEdge()
) -> SubdivisionResult:
    """Split a selected cell exactly, retaining its output dtype and device."""
    cid = model.space.cell_id(cell)
    info = model.space.cells[cid]
    local = BezierSimploid(info.dimensions, info.degree, model.coefficients[list(info.controls)])
    if isinstance(policy, LongestEdge):
        factor, i, j, fraction = longest_edge_criterion(local, policy.fraction)
        chosen = EdgeSplit(i, j, fraction, factor)
    elif isinstance(policy, ResidualSplit):
        selected = policy.samples.points.cell_ids == cid
        points = policy.samples.points.parameters[selected, : info.width]
        values = policy.samples.values[selected.to(policy.samples.values.device)]
        weights = policy.samples.weights[selected.cpu()]
        if not len(weights) or not torch.all(weights == weights[0]):
            raise ValueError("ResidualSplit requires equally weighted observations on the selected cell.")
        factor, i, j, fraction = max_error_criterion(points, values, policy.resolution)(local)
        chosen = EdgeSplit(i, j, fraction, factor)
    elif isinstance(policy, EdgeSplit):
        chosen = policy
    else:
        raise TypeError("policy must be EdgeSplit, LongestEdge or ResidualSplit.")
    a, b = split(local, chosen.i, chosen.j, chosen.fraction, factor=chosen.factor)
    space = BezierSpace.simploid(info.dimensions, degree=info.degree)
    models = (BezierModel(space, a.control_points), BezierModel(space, b.control_points))
    maps = (ParameterMap(cid, info.dimensions, chosen, "A"), ParameterMap(cid, info.dimensions, chosen, "B"))
    return SubdivisionResult(models, maps)
