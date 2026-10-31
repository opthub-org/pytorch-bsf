"""Faces and controls shared by skeleton sampling, its risk analysis and the inductive skeleton fitters.

A face of a simploid is a product of one face (a nonempty vertex subset) per simplex factor. It is keyed by the
sorted global coordinate indices of its vertices, so a row lies on the face spanned by its nonzero coordinates, and
the keys of a single simplex are its sorted vertex tuples. The dimension of a face is its number of vertices minus
the number of factors.
"""

from collections.abc import Sequence
from itertools import combinations, product

import torch

from torch_bsf._kernels.combinatorics import _indices
from torch_bsf.core import BezierSpace, Cell, Points, _point_groups


def _check_skeleton_space(space: BezierSpace) -> None:
    """Validate the common contract without imposing risk-computation limits."""
    if not isinstance(space, BezierSpace):
        raise TypeError("space must be a BezierSpace.")
    if any(d < 1 for cell in space.cells for d in cell.degree):
        raise ValueError("skeleton methods require positive degree in every positive-dimensional factor.")


def _single_skeleton(space: BezierSpace) -> bool:
    """Whether the original single-cell algorithms apply, including their face keys and RNG order."""
    _check_skeleton_space(space)
    return space.n_cells == 1 and bool(space.cells[0].dimensions) and not space._topology


def _skeleton_stages(space: BezierSpace) -> list[tuple[int, tuple[int, ...]]]:
    """Active nondegenerate cells and their proper global controls, in fitting order."""
    _check_skeleton_space(space)
    stages = []
    for cid, cell in enumerate(space.cells):
        proper = tuple(
            cell.controls[i]
            for i, alpha in enumerate(_indices(cell.dimensions, cell.degree))
            if all(v > 0 for factor in alpha for v in factor)
        )
        if proper:
            stages.append((cid, proper))
    stages.sort(key=lambda stage: (sum(space.cells[stage[0]].dimensions), stage[0]))
    if sorted(i for _, proper in stages for i in proper) != list(range(space.n_controls)):
        raise ValueError("cell allocation requires addressable faces; construct the space with BezierSpace.on_set.")
    return stages


def _push_face(space: BezierSpace, cid: int, factor: int, vertex: int, t: torch.Tensor) -> tuple[int, torch.Tensor]:
    """Restrict coordinates to one face and push its degeneracy map to the nondegenerate base."""
    cell = space.cells[cid]
    blocks = [t[:, block] for block in _blocks(cell.dimensions)]
    blocks[factor] = torch.cat((blocks[factor][:, :vertex], blocks[factor][:, vertex + 1 :]), dim=1)
    if cell.dimensions[factor] == 1:
        blocks.pop(factor)
    _, _, base, dmap = next(face for face in space._topology[cid] if face[:2] == (factor, vertex))
    out = t.new_zeros(len(t), space.cells[base].width)
    base_blocks = _blocks(space.cells[base].dimensions)
    for block, (src, mapping) in zip(blocks, dmap, strict=True):
        if src >= 0:
            columns = torch.tensor(mapping, dtype=torch.long) + base_blocks[src].start
            out.index_add_(1, columns, block)
    return base, out


def _skeleton_observations(space: BezierSpace, points: Points) -> tuple[Points, dict[int, torch.Tensor]]:
    """Canonicalize exact-zero boundary rows, using each observation once, including degenerate faces."""
    space.validate(points)
    parameters = points.parameters.detach().cpu().double()
    out = torch.zeros(len(points), space.width, dtype=torch.float64)
    ids = torch.empty(len(points), dtype=torch.long)
    pending = [
        (cid, rows.cpu(), parameters[rows.cpu(), : space.cells[cid].width]) for cid, rows in _point_groups(points.cell_ids)
    ]
    while pending:
        cid, rows, t = pending.pop()
        remaining = torch.ones(len(rows), dtype=torch.bool)
        for factor, block in enumerate(_blocks(space.cells[cid].dimensions)):
            for vertex, column in enumerate(block):
                boundary = remaining & (t[:, column] == 0)
                if boundary.any():
                    base, moved = _push_face(space, cid, factor, vertex, t[boundary])
                    pending.append((base, rows[boundary], moved))
                    remaining &= ~boundary
        out[rows[remaining], : t.shape[1]] = t[remaining]
        ids[rows[remaining]] = cid
    canonical = Points(out, ids)
    return canonical, dict(_point_groups(ids))


def _skeleton_cell(space: BezierSpace) -> Cell:
    """The cell of a single simploid whose factors all have positive dimension and degree."""
    if not isinstance(space, BezierSpace):
        raise TypeError("space must be a BezierSpace.")
    cell = space.cells[0]
    if space.n_cells != 1 or not cell.dimensions or min(cell.degree) < 1:
        raise ValueError("skeleton methods require a single simplex or simploid of positive dimension and degree.")
    return cell


def _blocks(dimensions: Sequence[int]) -> list[range]:
    """The global coordinate indices of each factor."""
    offsets = [0]
    for a in dimensions:
        offsets.append(offsets[-1] + a + 1)
    return [range(start, stop) for start, stop in zip(offsets[:-1], offsets[1:], strict=True)]


def _faces_with_interior_controls(dimensions: Sequence[int], degree: Sequence[int]) -> list[tuple[int, ...]]:
    """The faces with at most ``degree`` vertices in every factor, by increasing dimension, then lexicographically.

    Other faces have no interior controls, so face-by-face fitting skips them.
    """
    per_factor = [
        [face for size in range(1, min(d, len(block)) + 1) for face in combinations(block, size)]
        for block, d in zip(_blocks(dimensions), degree, strict=True)
    ]
    return sorted((sum(faces, ()) for faces in product(*per_factor)), key=lambda face: (len(face), face))


def _control_faces(dimensions: Sequence[int], degree: Sequence[int]) -> list[tuple[int, ...]]:
    """The face each control is interior to (the support of its multi-index), in control order."""
    return [
        tuple(i for i, a in enumerate(v for factor in alpha for v in factor) if a)
        for alpha in _indices(tuple(dimensions), tuple(degree))
    ]


def _interior_and_boundary(control_faces: list[tuple[int, ...]], face: tuple[int, ...]) -> tuple[list[int], list[int]]:
    """Indices of the controls interior to ``face`` and of those on its boundary."""
    vertices = set(face)
    interior = [k for k, f in enumerate(control_faces) if f == face]
    boundary = [k for k, f in enumerate(control_faces) if f != face and vertices.issuperset(f)]
    return interior, boundary
