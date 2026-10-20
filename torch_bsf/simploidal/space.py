r"""Simploidal Bézier spaces :cite:p:`peltier2018simploidal`."""

from typing import Mapping, Sequence, TypeAlias, cast

import torch
import torch.nn as nn

from torch_bsf.simploidal.bezier import BezierSimploid
from torch_bsf.simploidal.combinatorics import (
    SimploidIndex,
    SimploidType,
    _check_degree,
    _face_index,
    _indices,
    _positions,
    _pullback_index,
    bernstein_basis,
    simploid_n_params,
)
from torch_bsf.simploidal.sets import Simploid, SimploidalSet

#: The degree assignment of a :class:`SimploidalBezierSpace`: one degree for every factor of every simploid,
#: or the degrees of some non-degenerate simploids (an integer for all their factors, or one per factor).
DegreeAssignment: TypeAlias = int | Mapping[Simploid, int | Sequence[int]]

#: A factor ``(simploid, i)`` of a non-degenerate simploid.
_Factor: TypeAlias = tuple[Simploid, int]


def _check_min_degree(degree: int, what: str) -> int:
    if int(degree) < 1:
        raise ValueError(f"{what} must be at least 1, but {degree} is given.")
    return int(degree)


def _face_factors(simploid_type: SimploidType, i: int) -> list[int]:
    """For each factor of a face ``(i, j)`` of a simploid of type ``simploid_type``, the factor it comes from."""
    n = len(simploid_type) if simploid_type[i] > 1 else len(simploid_type) - 1
    return [k if simploid_type[i] > 1 or k < i else k + 1 for k in range(n)]


def _degree_classes(simploids: Sequence[Simploid]) -> dict[_Factor, _Factor]:
    """Groups the factors of non-degenerate simploids whose degrees must agree, returning the representative of each.

    A factor of a simploid and the corresponding factor of each of its faces must have the same degree; for
    a degenerate face, this applies to the factor of its base that the face's factor comes from, whereas
    an inserted edge factor imposes no constraint (the Bézier simploid is constant along it).
    """
    parent: dict[_Factor, _Factor] = {(s, i): (s, i) for s in simploids for i in range(s.length)}

    def find(x: _Factor) -> _Factor:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for s in simploids:
        for (i, _), f in s.faces():
            factors = _face_factors(s.simploid_type, i)
            for k, (src, _) in enumerate(f.degeneracy_map):
                if src >= 0:
                    parent[find((f.base, src))] = find((s, factors[k]))
    return {x: find(x) for x in parent}


def _fixed_degrees(
    simploids: Sequence[Simploid], degree: Mapping[Simploid, int | Sequence[int]], roots: dict[_Factor, _Factor]
) -> dict[_Factor, tuple[int, Simploid, int]]:
    """The degree given to each class of factors, with the factor it was first given to."""
    members = set(simploids)
    fixed: dict[_Factor, tuple[int, Simploid, int]] = {}
    for s, ds in degree.items():
        if s not in members:
            raise ValueError(f"{s!r} is not a non-degenerate simploid of the simploidal set.")
        for i, d in enumerate(_check_degree(ds, s.length)):
            d = _check_min_degree(d, "degree")
            prev = fixed.setdefault(roots[(s, i)], (d, s, i))
            if prev[0] != d:
                raise ValueError(
                    f"Inconsistent degrees: factor {i} of {s!r} is given degree {d}, but factor {prev[2]} of "
                    f"{prev[1]!r}, whose degree must agree with it through shared faces, is given degree {prev[0]}."
                )
    return fixed


def _assign_degrees(
    simploids: Sequence[Simploid], degree: DegreeAssignment, default_degree: int | None
) -> dict[Simploid, tuple[int, ...]]:
    """The degree of each factor of each non-degenerate simploid."""
    if isinstance(degree, int):
        d = _check_min_degree(degree, "degree")
        return {s: (d,) * s.length for s in simploids}
    if default_degree is not None:
        default_degree = _check_min_degree(default_degree, "default_degree")
    roots = _degree_classes(simploids)
    fixed = _fixed_degrees(simploids, degree, roots)
    result: dict[Simploid, tuple[int, ...]] = {}
    for s in simploids:
        ds = []
        for i in range(s.length):
            given = fixed.get(roots[(s, i)])
            if given is None and default_degree is None:
                raise ValueError(
                    f"The degree of factor {i} of {s!r} is not determined; specify it in degree or give default_degree."
                )
            ds.append(given[0] if given is not None else cast(int, default_degree))
        result[s] = tuple(ds)
    return result


class SimploidalBezierSpace(nn.Module):
    r"""A simploidal Bézier space :cite:p:`peltier2018simploidal`.

    A Bézier simploid is associated with every simploid of a
    :class:`SimploidalSet`.  Only the *proper* control points of the
    non-degenerate simploids (those whose multi-indices have no zero
    component, and the single point of each vertex) are stored, in
    :attr:`proper_control_points`.  The full control net of any simploid is
    rebuilt from its proper points and those of its boundary (Property 2-1),
    and the control net of a degenerate simploid from that of its base
    (Property 2-2).  Hence every control point is stored exactly once and
    simploids sharing a face agree on it.

    The structure of the simploidal set is captured when the space is
    created; modifying the set afterwards is not reflected in the space.

    Every factor of every non-degenerate simploid has its own degree.  The
    degrees must be consistent: a factor of a simploid and the corresponding
    factor of each of its faces have the same degree, so that a shared face
    has a single control net.  In particular, the degree of a factor of a
    simploid is the degree of its edges in the direction of that factor.  An
    edge factor inserted by a degeneracy (``degeneracy(i, -1)``), along which
    the Bézier simploid is constant, imposes no constraint; it gets degree
    :attr:`degree` in :meth:`simploid_degree`.

    Parameters
    ----------
    simploidal_set
        The simploidal set.
    degree
        Either the degree of every factor of every simploid (at least 1), or a
        mapping from non-degenerate simploids to their degrees (an integer for
        all their factors, or one degree per factor).  With a mapping, the
        degrees propagate through shared faces to the simploids it does not
        mention; a factor whose degree is still undetermined gets ``default_degree``.
    n_values
        The number of values (target dimension).
    proper_control_points
        Initial proper control points of shape ``(n_proper_control_points, n_values)``.  Zeros if ``None``.
    default_degree
        The degree of the factors left undetermined by a ``degree`` mapping (at least 1).
        Ignored when ``degree`` is an integer.

    Raises
    ------
    ValueError
        If a degree is less than 1, if the degree mapping is inconsistent across
        shared faces or mentions a simploid that is not a non-degenerate simploid
        of the set, or if a degree is left undetermined without ``default_degree``.

    Examples
    --------
    Two triangles sharing an edge share the control points of that edge:

    >>> import torch
    >>> from torch_bsf.simploidal import SimploidalSet, SimploidalBezierSpace
    >>> S = SimploidalSet()
    >>> t1, t2 = S.add_simplex(2), S.add_simplex(2)
    >>> e = S.identify(t1.face(0, 0), t2.face(0, 0))
    >>> space = SimploidalBezierSpace(S, degree=3, n_values=2)
    >>> space.n_proper_control_points  # 4 vertices, 5 edges x 2, 2 triangles x 1
    16
    >>> _ = torch.manual_seed(0)
    >>> _ = torch.nn.init.normal_(space.proper_control_points)
    >>> u = torch.tensor([[0.0, 0.3, 0.7]])  # a point on the shared edge
    >>> torch.allclose(space.evaluate(t1, u), space.evaluate(t2, u), atol=1e-6)
    True

    A prism that is linear across its triangles and cubic along its height,
    glued to a cube along a square face.  The degrees of the cube's factors
    along that face follow from the prism; the remaining one is the default:

    >>> S = SimploidalSet()
    >>> prism, cube = S.add_simploid((2, 1)), S.add_simploid((1, 1, 1))
    >>> _ = S.identify(prism.face(0, 0), cube.face(0, 0))
    >>> space = SimploidalBezierSpace(S, {prism: (1, 3)}, n_values=1, default_degree=2)
    >>> space.simploid_degree(cube)
    (2, 1, 3)
    """

    def __init__(
        self,
        simploidal_set: SimploidalSet,
        degree: DegreeAssignment,
        n_values: int,
        proper_control_points: torch.Tensor | None = None,
        *,
        default_degree: int | None = None,
    ):
        super().__init__()
        self.simploidal_set = simploidal_set
        self.simploids: list[Simploid] = simploidal_set.simploids()
        self._ids = {s: k for k, s in enumerate(self.simploids)}
        self._degrees = _assign_degrees(self.simploids, degree, default_degree)
        all_degrees = [d for ds in self._degrees.values() for d in ds]
        #: The degree of every factor if they are all equal, and their maximum otherwise.
        self.degree = int(degree) if isinstance(degree, int) else max(all_degrees, default=default_degree or 1)
        self._uniform = all(d == self.degree for d in all_degrees)
        self._proper: dict[Simploid, dict[SimploidIndex, int]] = {}
        offset = 0
        for s in self.simploids:
            proper = [alpha for alpha in _indices(s.simploid_type, self._degrees[s]) if all(c > 0 for m in alpha for c in m)]
            self._proper[s] = {alpha: offset + k for k, alpha in enumerate(proper)}
            offset += len(proper)
        self._rows: dict[tuple[Simploid, tuple[int, ...]], list[int]] = {}
        if proper_control_points is None:
            proper_control_points = torch.zeros(offset, n_values)
        if tuple(proper_control_points.shape) != (offset, n_values):
            raise ValueError(
                f"proper_control_points must have shape ({offset}, {n_values}), "
                f"but {tuple(proper_control_points.shape)} is given."
            )
        matrix = torch.as_tensor(proper_control_points)
        if not matrix.is_floating_point():
            matrix = matrix.to(torch.get_default_dtype())
        self.proper_control_points = nn.Parameter(matrix.detach().clone())

    def extra_repr(self) -> str:
        degree = f"degree={self.degree}" if self._uniform else f"max_degree={self.degree}"
        return f"n_simploids={len(self.simploids)}, {degree}, n_values={self.n_values}"

    @property
    def n_values(self) -> int:
        """The number of values (target dimension)."""
        return int(self.proper_control_points.shape[1])

    @property
    def n_proper_control_points(self) -> int:
        """The number of stored (proper) control points."""
        return int(self.proper_control_points.shape[0])

    def simploid_id(self, simploid: Simploid) -> int:
        """The position of a non-degenerate simploid in :attr:`simploids`, used by :meth:`forward`."""
        try:
            return self._ids[simploid]
        except KeyError:
            raise ValueError(f"{simploid!r} is not a non-degenerate simploid of this space.") from None

    def simploid_degree(self, simploid: Simploid) -> tuple[int, ...]:
        """The degree of each factor of the Bézier simploid of a simploid.

        A degenerate simploid has the degrees of the factors of its base it comes
        from, and degree :attr:`degree` on the edge factors inserted by degeneracies.

        Parameters
        ----------
        simploid
            Any simploid of the set, degenerate or not.

        Returns
        -------
        One degree per factor.

        Examples
        --------
        >>> from torch_bsf.simploidal import SimploidalSet, SimploidalBezierSpace
        >>> S = SimploidalSet()
        >>> square = S.add_simploid((1, 1))
        >>> space = SimploidalBezierSpace(S, {square: (2, 3)}, n_values=1)
        >>> space.simploid_degree(square), space.simploid_degree(square.face(1, 0))
        ((2, 3), (2,))
        >>> space.simploid_degree(square.degeneracy(0, -1))
        (3, 2, 3)
        """
        base = simploid.base
        if base not in self._ids:
            raise ValueError(f"{simploid!r} does not belong to the simploidal set of this space.")
        if not simploid.is_degenerate:
            return self._degrees[simploid]
        ds = self._degrees[base]
        return tuple(ds[src] if src >= 0 else self.degree for src, _ in simploid.degeneracy_map)

    def control_point_rows(self, simploid: Simploid) -> list[int]:
        """The rows of :attr:`proper_control_points` holding the full control net of a simploid.

        Parameters
        ----------
        simploid
            Any simploid of the set, degenerate or not.

        Returns
        -------
        One row per control point, in the order of :func:`simploid_indices`.
        """
        return self._rows_of(simploid, self.simploid_degree(simploid))

    def _rows_of(self, s: Simploid, degree: tuple[int, ...]) -> list[int]:
        """The rows of the control net of ``s`` at ``degree``.

        ``degree`` is :meth:`simploid_degree` except on the inserted edge factors of a degenerate
        face, which take the degree of the corresponding factor of the simploid it is a face of.
        """
        rows = self._rows.get((s, degree))
        if rows is None:
            rows = self._degenerate_rows(s, degree) if s.is_degenerate else self._nondegenerate_rows(s)
            self._rows[(s, degree)] = rows
        return rows

    def _degenerate_rows(self, s: Simploid, degree: tuple[int, ...]) -> list[int]:
        base = s.base
        base_degree = self._degrees[base]
        base_rows = self._rows_of(base, base_degree)
        pos = _positions(base.simploid_type, base_degree)
        return [base_rows[pos[_pullback_index(alpha, s._map, base.length)]] for alpha in _indices(s.simploid_type, degree)]

    def _nondegenerate_rows(self, s: Simploid) -> list[int]:
        proper = self._proper[s]
        degree = self._degrees[s]
        rows = []
        for alpha in _indices(s.simploid_type, degree):
            if alpha in proper:
                rows.append(proper[alpha])
                continue
            i, j = next((i, m.index(0)) for i, m in enumerate(alpha) if 0 in m)
            f = s.face(i, j)
            face_degree = degree if s.simploid_type[i] > 1 else degree[:i] + degree[i + 1 :]
            pos = _positions(f.simploid_type, face_degree)
            rows.append(self._rows_of(f, face_degree)[pos[_face_index(alpha, i, j)]])
        return rows

    def control_points(self, simploid: Simploid) -> torch.Tensor:
        """The full control net of a simploid, of shape ``(n_indices, n_values)`` (differentiable)."""
        rows = torch.tensor(self.control_point_rows(simploid), dtype=torch.long, device=self.proper_control_points.device)
        return self.proper_control_points[rows]

    def bezier_simploid(self, simploid: Simploid) -> BezierSimploid:
        """A standalone copy of the Bézier simploid associated with a simploid."""
        return BezierSimploid(simploid.simploid_type, self.simploid_degree(simploid), self.control_points(simploid).detach())

    def evaluate(self, simploid: Simploid, t: torch.Tensor) -> torch.Tensor:
        r"""Evaluates the Bézier simploid of a simploid.

        Parameters
        ----------
        simploid
            Any simploid of the set, degenerate or not.
        t
            Parameters of shape ``(batch, sum(a_i + 1))``.

        Returns
        -------
        Values of shape ``(batch, n_values)``.
        """
        cp = self.control_points(simploid)
        t = torch.as_tensor(t, dtype=cp.dtype, device=cp.device)
        return bernstein_basis(simploid.simploid_type, self.simploid_degree(simploid), t) @ cp

    def forward(self, t: torch.Tensor, simploid_ids: torch.Tensor) -> torch.Tensor:
        r"""Evaluates a batch mixing several simploids.

        Parameters
        ----------
        t
            Parameters of shape ``(batch, width)``: row ``k`` holds the parameters of simploid
            ``simploids[simploid_ids[k]]`` in its first ``sum(a_i + 1)`` columns (the rest is ignored).
        simploid_ids
            Indices into :attr:`simploids` (see :meth:`simploid_id`) of shape ``(batch,)``.

        Returns
        -------
        Values of shape ``(batch, n_values)``.
        """
        p = self.proper_control_points
        t = torch.as_tensor(t, dtype=p.dtype, device=p.device)
        ids = torch.as_tensor(simploid_ids, dtype=torch.long, device=p.device)
        out = torch.zeros(ids.shape[0], self.n_values, dtype=p.dtype, device=p.device)
        for sid in torch.unique(ids).tolist():
            s = self.simploids[sid]
            (rows,) = torch.nonzero(ids == sid, as_tuple=True)
            out = out.index_put((rows,), self.evaluate(s, t[rows, : simploid_n_params(s.simploid_type)]))
        return out

    def design_matrix(self, t: torch.Tensor, simploid_ids: torch.Tensor) -> torch.Tensor:
        r"""The linear map from the proper control points to the values of a batch mixing several simploids.

        ``forward(t, simploid_ids) == design_matrix(t, simploid_ids) @ proper_control_points``.

        Parameters
        ----------
        t, simploid_ids
            As in :meth:`forward`.

        Returns
        -------
        A matrix of shape ``(batch, n_proper_control_points)`` on the device of
        :attr:`proper_control_points`, in the dtype of ``t`` (the default dtype if ``t`` is not floating).

        Examples
        --------
        >>> import torch
        >>> from torch_bsf.simploidal import SimploidalSet, SimploidalBezierSpace
        >>> S = SimploidalSet()
        >>> tri = S.add_simplex(2)
        >>> space = SimploidalBezierSpace(S, degree=2, n_values=1)
        >>> _ = torch.nn.init.normal_(space.proper_control_points)
        >>> t, ids = torch.tensor([[0.2, 0.3, 0.5]]), torch.tensor([space.simploid_id(tri)])
        >>> torch.allclose(space.design_matrix(t, ids) @ space.proper_control_points, space(t, ids))
        True
        """
        p = self.proper_control_points
        t = torch.as_tensor(t, device=p.device)
        if not t.is_floating_point():
            t = t.to(torch.get_default_dtype())
        ids = torch.as_tensor(simploid_ids, dtype=torch.long, device=p.device)
        out = torch.zeros(ids.shape[0], self.n_proper_control_points, dtype=t.dtype, device=p.device)
        for sid in torch.unique(ids).tolist():
            s = self.simploids[sid]
            (rows,) = torch.nonzero(ids == sid, as_tuple=True)
            basis = bernstein_basis(s.simploid_type, self.simploid_degree(s), t[rows, : simploid_n_params(s.simploid_type)])
            cols = torch.tensor(self.control_point_rows(s), dtype=torch.long, device=p.device)
            block = torch.zeros(len(rows), self.n_proper_control_points, dtype=t.dtype, device=p.device)
            out[rows] = block.index_add_(1, cols, basis)
        return out
