"""Macro-element interpolants and the refinement indicator of :mod:`torch_bsf.interpolation`."""

from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any

import torch

from torch_bsf._interpolation.constructions import (
    _CONSTRUCTIONS,
    _CONTINUITIES,
    _SPLITS,
    C0,
    C1,
    CloughTocher,
    NoSplit,
    _supported_constructions,
)
from torch_bsf._interpolation.exact import _times_power_of_two
from torch_bsf._interpolation.hermite import _rounded
from torch_bsf._interpolation.mesh import SimplicialMesh
from torch_bsf.core import BezierModel, Points


@dataclass(frozen=True, eq=False)
class MacroInterpolant:
    """A piecewise cubic interpolant on a mesh, defined on pieces of its simplices (see :func:`macro_interpolant`).

    Attributes
    ----------
    mesh
        The mesh carrying the data.
    pieces
        The mesh of the pieces, on whose :attr:`~SimplicialMesh.space` the model is defined. It is ``mesh`` itself
        without a split.
    parent
        ``parent[k]`` is the simplex of ``mesh`` containing piece ``k``.
    model
        The float64 model. Evaluate it at :meth:`locate`, or at :meth:`transfer` of points of ``mesh.space``.
    """

    mesh: SimplicialMesh
    pieces: SimplicialMesh
    parent: torch.Tensor
    model: BezierModel
    _transfer: Callable[[Points], Points] = field(repr=False, kw_only=True)

    def transfer(self, points: Points) -> Points:
        """Map points of ``mesh.space`` to the pieces containing them, preserving their order.

        Parameters
        ----------
        points
            Barycentric coordinates in simplices of ``mesh`` and their cell IDs (from :attr:`SimplicialMesh.cell_ids`),
            e.g. from :meth:`SimplicialMesh.locate`.

        Returns
        -------
        Points
            Barycentric coordinates and cell IDs in ``pieces.space``.

        Raises
        ------
        ValueError
            If ``points`` are not valid points of ``mesh.space`` (see :meth:`~torch_bsf.core.BezierSpace.validate`),
            or a cell ID is not that of a simplex of ``mesh`` (e.g. a face, or a simplex replaced by :func:`refine`).

        Examples
        --------
        The interpolant of the first coordinate reproduces it, here at the centroids of the triangles:

        >>> import torch
        >>> from torch_bsf.core import Points
        >>> from torch_bsf.interpolation import C1, CloughTocher, lattice_mesh, macro_interpolant
        >>> mesh = lattice_mesh(2, 2)
        >>> interpolant = macro_interpolant(mesh, mesh.vertices[:, 0], split=CloughTocher(), continuity=C1())
        >>> centroids = Points(torch.full((mesh.n_simplices, 3), 1 / 3, dtype=torch.float64), mesh.cell_ids)
        >>> values = interpolant.model(interpolant.transfer(centroids))[:, 0]
        >>> torch.allclose(values, mesh.vertices[mesh.simplices, 0].mean(dim=1))
        True
        """
        if not torch.isin(points.cell_ids.cpu(), self.mesh.cell_ids).all():
            raise ValueError("points must lie in simplices of the mesh: their cell IDs must be in mesh.cell_ids.")
        self.mesh.space.validate(points)
        return self._transfer(points)

    def locate(self, x: Any, *, tol: float = 1e-8, batch_size: int | None = None) -> Points:
        """Find the piece containing each point and its barycentric coordinates there.

        The points are located in :attr:`mesh` by :meth:`SimplicialMesh.locate`, and then in the pieces of their
        simplex by :meth:`transfer`.

        Parameters
        ----------
        x
            Points of shape ``(n_points, d)``.
        tol
            Tolerance on the barycentric coordinates, as in :meth:`SimplicialMesh.locate`.
        batch_size
            Number of points located at a time, as in :meth:`SimplicialMesh.locate`, which also chooses the
            default.

        Returns
        -------
        Points
            Barycentric coordinates and cell IDs in ``pieces.space``, in the order of ``x``.

        Raises
        ------
        ValueError
            If ``tol`` or a point is invalid, or a point is outside the mesh (see :meth:`SimplicialMesh.locate`).
        """
        return self.transfer(self.mesh.locate(x, tol=tol, batch_size=batch_size))

    def __call__(self, x: Any, *, tol: float = 1e-8, batch_size: int | None = None) -> torch.Tensor:
        """Evaluate the interpolant at points ``x``.

        Parameters
        ----------
        x
            Points of shape ``(n_points, d)``.
        tol, batch_size
            As in :meth:`locate`.

        Returns
        -------
        torch.Tensor
            Float64 predictions of shape ``(n_points, n_outputs)``.
        """
        prediction: torch.Tensor = self.model(self.locate(x, tol=tol, batch_size=batch_size))
        return prediction


def macro_interpolant(
    mesh: SimplicialMesh,
    values: Any,
    *,
    split: NoSplit | CloughTocher = NoSplit(),
    continuity: C0 | C1 = C0(),
    gradients: Any = None,
    clamp: bool = True,
) -> MacroInterpolant:
    r"""Interpolate vertex data by a piecewise cubic Bézier model on pieces of the simplices of a mesh.

    ``split`` says how each simplex is divided into pieces, and ``continuity`` how smoothly neighbouring pieces
    join. The supported constructions are

    * ``NoSplit()`` with ``C0()`` (any dimension): :func:`hermite_interpolant`;
    * ``CloughTocher()`` with ``C1()`` (two dimensions): the Clough–Tocher macro-element, continuously
      differentiable, on three cubic pieces per triangle.

    Both interpolate the data and reproduce quadratic data without clamping when the gradients do. With
    ``clamp=True``, on every simplex of ``mesh`` and for every output, the prediction lies between the smallest and
    largest data at its vertices. For the C1 construction, the gradients are first limited for this, each projected
    onto the gradients keeping the controls next to its vertex in the range of every triangle containing them,
    which keeps the interpolant C1. In the spirit of :cite:p:`fritsch1980monotone`, this is a two-dimensional
    range-restricted C1 interpolant. The set of such gradients can shrink abruptly as the mesh moves. For example,
    take a vertex whose datum is the largest datum of the triangle on one side of each of two of its edges and the
    smallest of the triangle on the other side. Its gradient must be orthogonal to both edges: while they are
    parallel, it is free along their common normal; once they are not, it must vanish. The clamped C1 interpolant
    therefore need not depend continuously on the vertex coordinates.

    The construction is not differentiated with respect to the mesh: like every :class:`~torch_bsf.core.BezierModel`,
    the model copies its coefficients without their autograd history, as for :func:`hermite_interpolant`, and the
    split points are computed from the vertices without it too. Derivatives with respect to the vertices therefore
    pass only through :meth:`SimplicialMesh.locate`.

    Parameters
    ----------
    mesh
        The mesh.
    values
        Vertex data of shape ``(n_vertices, n_outputs)`` or ``(n_vertices,)``. For the C1 construction, an output
        with data beyond about 2**1000 (or, without ``clamp``, controls beyond it) is processed in units of a power
        of two up to 2**24, so that its sums stay finite; its subnormal data are then interpolated only up to an
        absolute error below 2**-1050.
    split
        How each simplex is divided into pieces: :class:`NoSplit` or :class:`CloughTocher`.
    continuity
        How smoothly neighbouring pieces join: :class:`C0` or :class:`C1`.
    gradients
        Vertex gradients of shape ``(n_vertices, d, n_outputs)``. Estimated with :func:`estimate_gradients` if
        ``None``. With ``clamp`` and the C1 construction, a gradient whose controls exceed about 2**1000 (near
        float64 overflow) is divided by a power of two before it is limited.
    clamp
        Whether to keep the prediction on every simplex within the range of its vertex data.

    Returns
    -------
    MacroInterpolant
        The interpolant: call it at points, or evaluate its ``model`` at its :meth:`~MacroInterpolant.locate`.

    Raises
    ------
    TypeError
        If ``split`` or ``continuity`` is not one of the classes above.
    ValueError
        If the construction is not supported, the mesh is not two-dimensional for :class:`CloughTocher` or has a
        triangle whose split point rounds onto one of its sides, the data or supplied gradients are invalid, or
        the interpolant is not finite in float64 (see :func:`hermite_interpolant`).

    Examples
    --------
    >>> import torch
    >>> from torch_bsf.interpolation import C1, CloughTocher, lattice_mesh, macro_interpolant
    >>> mesh = lattice_mesh(2, 3)
    >>> x = mesh.vertices
    >>> interpolant = macro_interpolant(mesh, x[:, 0] * x[:, 1], split=CloughTocher(), continuity=C1())
    >>> interpolant.pieces.n_simplices == 3 * mesh.n_simplices
    True
    >>> y = interpolant(x)
    >>> torch.allclose(y[:, 0], x[:, 0] * x[:, 1])
    True
    """
    # Exact types, as the constructions are looked up by them.
    if type(split) not in _SPLITS or type(continuity) not in _CONTINUITIES:
        splits, continuities = (" or ".join(c.__name__ for c in options) for options in (_SPLITS, _CONTINUITIES))
        raise TypeError(f"split must be {splits}, and continuity {continuities}.")
    construction = _CONSTRUCTIONS.get((type(split), type(continuity)))
    if construction is None:
        raise ValueError(
            f"split={split!r} with continuity={continuity!r} is not supported; supported: {_supported_constructions()}."
        )
    f = mesh._check_values(values)
    pieces = _SPLITS[type(split)](mesh)
    model = construction(mesh, pieces, f, gradients, bool(clamp))
    return MacroInterpolant(mesh, pieces.mesh, pieces.parent, model, _transfer=pieces.transfer)


def _refinement_scores(
    cubic: torch.Tensor, values: torch.Tensor, volume_mantissa: torch.Tensor, volume_exponent: torch.Tensor
) -> torch.Tensor:
    """Volume-weighted mean correction, retaining tiny residuals and avoiding intermediate overflow.

    Ordinary rows keep the vectorized reductions. Near either float64 boundary, compute the score from
    the finite centroid predictions and vertex data as exact rationals, rounding only the final score.
    This also combines a correction outside float64 with a small volume before rounding either of them.
    """
    if not torch.isfinite(cubic).all():
        raise ValueError("the interpolant has a non-finite centroid prediction.")
    linear = values.mean(dim=1)
    correction = (cubic - linear).abs().mean(dim=1)
    mantissa, exponent = torch.frexp(correction)
    score = _times_power_of_two(mantissa * volume_mantissa, exponent.to(torch.long) + volume_exponent)
    score = torch.where(correction == 0, 0.0, score)
    entries = torch.cat([cubic, values.flatten(1)], dim=1).abs()
    info = torch.finfo(torch.float64)
    # A factor-two margin bounds both vertex/output sums; small terms must survive
    # division by their counts until a possibly huge volume has been applied.
    count = max(values.shape[1], values.shape[2])
    risky = (entries >= info.max / (2 * count)).any(dim=1)
    risky |= ((entries > 0) & (entries < info.tiny * count)).any(dim=1)
    risky |= ~torch.isfinite(score) | ((score == 0) & (correction > 0))
    for row in risky.nonzero().flatten().tolist():
        columns = zip(cubic[row].tolist(), values[row].T.tolist(), strict=True)
        exact = (
            sum(
                (
                    abs(Fraction(prediction) - sum(map(Fraction, column), Fraction(0)) / len(column))
                    for prediction, column in columns
                ),
                Fraction(0),
            )
            / cubic.shape[1]
        )
        volume = Fraction(volume_mantissa[row].item()) * Fraction(2) ** int(volume_exponent[row])
        score[row] = _rounded(exact * volume)
    return score


def refinement_indicator(
    mesh: SimplicialMesh,
    values: Any,
    *,
    gradients: Any = None,
    clamp: bool = True,
    split: NoSplit | CloughTocher = NoSplit(),
    continuity: C0 | C1 = C0(),
) -> torch.Tensor:
    """Score simplices for refinement by the cubic correction to linear interpolation.

    The score of a simplex is its volume times the mean over outputs of the absolute difference between
    the cubic interpolant (:func:`macro_interpolant`, by default :func:`hermite_interpolant`) and linear
    interpolation at its centroid. It is large where the data bend
    within a large simplex. Refine the highest-scoring simplices first, for example
    with :func:`refine`, and evaluate the response at the new vertices.

    Parameters
    ----------
    mesh
        The mesh.
    values
        Vertex data of shape ``(n_vertices, n_outputs)`` or ``(n_vertices,)``.
    gradients
        Vertex gradients, as in :func:`hermite_interpolant`.
    clamp
        Whether the cubic interpolant is clamped, as in :func:`hermite_interpolant`.
    split, continuity
        The construction of the cubic interpolant, as in :func:`macro_interpolant`.

    Returns
    -------
    torch.Tensor
        Nonnegative float64 scores of shape ``(n_simplices,)``, one per simplex of ``mesh``.
        Finite centroid predictions and extreme vertex data use exact reductions near the float64
        limits, applying the volume before final rounding. A score beyond float64 is positive infinity.

    Raises
    ------
    TypeError
        If ``split`` or ``continuity`` is not one of the classes of :func:`macro_interpolant`.
    ValueError
        If :func:`macro_interpolant` cannot construct a finite interpolant, including when the
        vertex data or gradients cause its controls to overflow float64, or if a centroid
        prediction is non-finite.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf.interpolation import SimplicialMesh, refinement_indicator
    >>> x = torch.arange(6.0)[:, None]
    >>> mesh = SimplicialMesh(x, [[k, k + 1] for k in range(5)])
    >>> [round(s, 6) for s in refinement_indicator(mesh, x.square()).tolist()]
    [0.25, 0.25, 0.25, 0.25, 0.25]
    """
    f = mesh._check_values(values)
    interpolant = macro_interpolant(mesh, f, split=split, continuity=continuity, gradients=gradients, clamp=clamp)
    n = mesh.dimension + 1
    centroid = Points(torch.full((mesh.n_simplices, n), 1 / n, dtype=torch.float64), mesh.cell_ids)
    with torch.no_grad():
        cubic: torch.Tensor = interpolant.model(interpolant.transfer(centroid))
    return _refinement_scores(cubic, f[mesh.simplices], mesh._volume_mantissa, mesh._volume_exponent)
