"""Faces and controls shared by skeleton sampling, its risk analysis and the inductive skeleton fitters.

A face of a simploid is a product of one face (a nonempty vertex subset) per simplex factor. It is keyed by the
sorted global coordinate indices of its vertices, so a row lies on the face spanned by its nonzero coordinates, and
the keys of a single simplex are its sorted vertex tuples. The dimension of a face is its number of vertices minus
the number of factors.
"""

from collections.abc import Sequence
from itertools import combinations, product

from torch_bsf._kernels.combinatorics import _indices
from torch_bsf.core import BezierSpace, Cell


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
