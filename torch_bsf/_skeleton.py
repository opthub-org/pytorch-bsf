"""Faces and controls shared by skeleton sampling, its risk analysis and the inductive skeleton fitters."""

from itertools import combinations

from torch_bsf._kernels.combinatorics import _simplex_indices
from torch_bsf.core import BezierSpace


def _skeleton_shape(space: BezierSpace) -> tuple[int, int]:
    """The vertex count and degree of a single simplex of positive dimension and degree."""
    if not isinstance(space, BezierSpace):
        raise TypeError("space must be a BezierSpace.")
    cell = space.cells[0]
    if space.n_cells != 1 or len(cell.dimensions) != 1 or cell.degree[0] < 1:
        raise ValueError("skeleton methods require a single simplex of positive dimension and degree.")
    return cell.dimensions[0] + 1, cell.degree[0]


def _faces_with_interior_controls(width: int, degree: int) -> list[tuple[int, ...]]:
    """Sorted vertex tuples of the faces with at most ``degree`` vertices, in increasing size.

    Larger faces of the skeletons have no interior controls, so face-by-face fitting skips them.
    """
    return [face for size in range(1, min(degree, width) + 1) for face in combinations(range(width), size)]


def _control_faces(width: int, degree: int) -> list[tuple[int, ...]]:
    """The face each control is interior to (the support of its multi-index), in control order."""
    return [tuple(i for i, a in enumerate(alpha) if a) for alpha in _simplex_indices(width, degree)]


def _interior_and_boundary(control_faces: list[tuple[int, ...]], face: tuple[int, ...]) -> tuple[list[int], list[int]]:
    """Indices of the controls interior to ``face`` and of those on its boundary."""
    vertices = set(face)
    interior = [k for k, f in enumerate(control_faces) if f == face]
    boundary = [k for k, f in enumerate(control_faces) if f != face and vertices.issuperset(f)]
    return interior, boundary
