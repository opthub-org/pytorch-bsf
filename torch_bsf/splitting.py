"""Exact local subdivision with explicit maps back to the parent domain."""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

import torch

from torch_bsf._kernels._conforming import Refinement, locate, nodes_of, split_nets
from torch_bsf._kernels._splitting import (
    _check_edge,
    _check_s,
    longest_edge_criterion,
    max_error_criterion,
    reparametrize,
    split,
)
from torch_bsf._kernels.bezier import BezierSimploid
from torch_bsf._kernels.sets import Simploid
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


def _choose(model: BezierModel, cid: int, policy: EdgeSplit | LongestEdge | ResidualSplit) -> EdgeSplit:
    """The edge split selected by ``policy`` on cell ``cid``."""
    info = model.space.cells[cid]
    local = BezierSimploid(info.dimensions, info.degree, model.coefficients[list(info.controls)])
    if isinstance(policy, LongestEdge):
        factor, i, j, fraction = longest_edge_criterion(local, policy.fraction)
        return EdgeSplit(i, j, fraction, factor)
    if isinstance(policy, ResidualSplit):
        selected = policy.samples.points.cell_ids == cid
        points = policy.samples.points.parameters[selected, : info.width]
        values = policy.samples.values[selected.to(policy.samples.values.device)]
        weights = policy.samples.weights[selected.cpu()]
        if not len(weights) or not torch.all(weights == weights[0]):
            raise ValueError("ResidualSplit requires equally weighted observations on the selected cell.")
        factor, i, j, fraction = max_error_criterion(points, values, policy.resolution)(local)
        return EdgeSplit(i, j, fraction, factor)
    if isinstance(policy, EdgeSplit):
        return policy
    raise TypeError("policy must be EdgeSplit, LongestEdge or ResidualSplit.")


def subdivide(
    model: BezierModel, *, cell: int = 0, policy: EdgeSplit | LongestEdge | ResidualSplit = LongestEdge()
) -> SubdivisionResult:
    """Split a selected cell exactly, retaining its output dtype and device."""
    cid = model.space.cell_id(cell)
    info = model.space.cells[cid]
    local = BezierSimploid(info.dimensions, info.degree, model.coefficients[list(info.controls)])
    chosen = _choose(model, cid, policy)
    a, b = split(local, chosen.i, chosen.j, chosen.fraction, factor=chosen.factor)
    space = BezierSpace.simploid(info.dimensions, degree=info.degree)
    models = (BezierModel(space, a.control_points), BezierModel(space, b.control_points))
    maps = (ParameterMap(cid, info.dimensions, chosen, "A"), ParameterMap(cid, info.dimensions, chosen, "B"))
    return SubdivisionResult(models, maps)


@dataclass(frozen=True, eq=False)
class SpaceSubdivision:
    """An exact conforming subdivision of a whole space and the map of points onto it.

    Attributes
    ----------
    model
        The subdivided model, representing the same function on a refined space.
    parents
        For every cell of the refined space, the cell of the original space whose relative interior contains it
        (``0`` for a single-cell space, whose faces are not cells).
    split
        The split of the selected cell chosen by the policy.
    """

    model: BezierModel
    parents: tuple[int, ...]
    split: EdgeSplit
    _refinement: Refinement = field(repr=False)
    _original: BezierSpace = field(repr=False)
    _nodes: tuple[Simploid, ...] = field(repr=False)

    def transfer(self, points: Points) -> Points:
        """Map points of the original space to the refined space, preserving their order.

        A point on a boundary shared by several pieces goes to one of them; every such choice evaluates to the
        same value, since the subdivision is conforming.
        """
        self._original.validate(points)
        parameters = torch.zeros(len(points), self.model.space.width, dtype=points.parameters.dtype)
        cell_ids = torch.zeros(len(points), dtype=torch.long)
        # The pieces are located on the CPU in float64; the result is moved back to the input's device.
        ids_cpu = points.cell_ids.cpu()
        for cid in torch.unique(ids_cpu).tolist():
            rows = torch.nonzero(ids_cpu == cid).flatten()
            info = self._original.cells[cid]
            node = self._nodes[cid]
            pieces = [lists for lists, _ in self._refinement.pieces(node)]
            t = points.parameters.cpu()[rows, : info.width].double()
            which, local = locate(t, info.dimensions, pieces)
            tops = self._refinement.tops[node]
            ids = torch.tensor([self.model.space.cell_id(tops[k]) for k in which.tolist()], dtype=torch.long)
            parameters[rows, : info.width] = local.to(parameters.dtype)
            cell_ids[rows] = ids
        return Points(parameters.to(points.parameters.device), cell_ids.to(points.cell_ids.device))


def subdivide_space(
    model: BezierModel, *, cell: int = 0, policy: EdgeSplit | LongestEdge | ResidualSplit = LongestEdge()
) -> SpaceSubdivision:
    """Split a cell exactly and propagate the split so that the whole space stays conforming.

    The selected factor edge is split at ``fraction``, and so is every factor edge of the space parallel to it
    through shared faces: the edges of the same factor in neighboring cells across shared faces and, through a
    degenerate face, the edge of its base it maps onto.  Each affected cell is bisected along all of its marked
    edges, so a shared face is subdivided in the same way from every side and the result is again a simploidal
    Bézier space: closed under splitting, including spaces with degenerate cells, for any fraction in
    :math:`(0, 1)`.  The coefficients are computed by de Casteljau subdivision, so the refined model represents
    the same function.

    Parameters
    ----------
    model
        The model to subdivide.
    cell
        The cell whose edge selects the split.
    policy
        How to choose the edge of ``cell``, as in :func:`subdivide`.

    Returns
    -------
    The refined model with the parent of every refined cell and a map of points onto the refined space.

    Examples
    --------
    Split one triangle of a square made of two triangles; the edge shared by both triangles is split,
    so the neighbor is split too:

    >>> import torch
    >>> from torch_bsf import BezierModel, BezierSpace, Points
    >>> from torch_bsf.geometry import SimploidalSet
    >>> from torch_bsf.splitting import EdgeSplit, subdivide_space
    >>> S = SimploidalSet()
    >>> a, b = S.add_simplex(2), S.add_simplex(2)
    >>> _ = S.identify(a.face(0, 0), b.face(0, 0))
    >>> space = BezierSpace.on_set(S, degree=2)
    >>> model = BezierModel(space, torch.randn(space.n_controls, 1, dtype=torch.float64))
    >>> result = subdivide_space(model, cell=space.cell_id(a), policy=EdgeSplit(1, 2, 0.3))
    >>> len(result.model.space.main_cells)
    4
    >>> points = Points(torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float64), [space.cell_id(b)])
    >>> torch.allclose(model(points), result.model(result.transfer(points)))
    True
    """
    space = model.space
    cid = space.cell_id(cell)
    chosen = _choose(model, cid, policy)
    info = space.cells[cid]
    _check_edge(info.dimensions, chosen.factor, chosen.i, chosen.j)
    _check_s(chosen.fraction)
    # The refinement orients every edge from its lower to its higher vertex; the fraction along that
    # orientation is kept exact, since 1 - s rounds to one for tiny s.
    reverse = chosen.i > chosen.j
    low, high = sorted((chosen.i, chosen.j))
    fraction = 1 - Fraction(chosen.fraction) if reverse else Fraction(chosen.fraction)
    nodes, ids = nodes_of([c.dimensions for c in space.cells], space._topology)
    cells = tuple(sorted(ids, key=ids.__getitem__))
    refinement = Refinement(nodes, (cells[cid], chosen.factor, low, high), fraction)
    degrees = {top: space.cells[k].degree for node, k in ids.items() for top in refinement.tops[node]}
    refined = BezierSpace.on_set(refinement.domain, degree=degrees)
    coefficients = torch.full(
        (refined.n_controls, model.n_outputs), torch.nan, dtype=model.coefficients.dtype, device=model.coefficients.device
    )
    for k in space.main_cells:
        node = cells[k]
        cell_info = space.cells[k]
        local = BezierSimploid(cell_info.dimensions, cell_info.degree, model.coefficients.detach()[list(cell_info.controls)])
        marks = refinement.marks.get(node, [])
        paths = [path for _, path in refinement.pieces(node)]
        for top, net in zip(refinement.tops[node], split_nets(local, marks, chosen.fraction, paths, reverse), strict=True):
            rows = list(refined.cells[refined.cell_id(top)].controls)
            coefficients[rows] = net.to(coefficients.dtype)
    # Only a single-cell space has simploids (the faces of its cell) that are not cells.
    by_id = sorted(refined._source_ids, key=refined.cell_id)
    parents = tuple(ids.get(refinement.support[s], 0) for s in by_id)
    return SpaceSubdivision(BezierModel(refined, coefficients), parents, chosen, refinement, space, cells)
