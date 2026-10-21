r"""Simploids and simploidal sets :cite:p:`peltier2018simploidal`."""

from collections.abc import Iterator, Sequence
from itertools import combinations
from math import gcd
from typing import Literal, cast

import torch

from torch_bsf.simploidal.combinatorics import DegeneracyMap, SimploidType, _check_type


def _identity_map(simploid_type: SimploidType) -> DegeneracyMap:
    return tuple((p, tuple(range(a + 1))) for p, a in enumerate(simploid_type))


def _compose(inner: DegeneracyMap, outer: DegeneracyMap) -> DegeneracyMap:
    """Composes ``outer`` (relative to a degenerate simploid) with ``inner`` (that simploid's own map)."""
    out = []
    for src, g in outer:
        if src < 0:
            out.append((src, g))
        else:
            src2, g2 = inner[src]
            out.append((src2, tuple(g2[v] for v in g)))
    return tuple(out)


def _canonical(base: "Simploid", dmap: DegeneracyMap) -> "Simploid":
    """The unique simploid obtained from ``base`` by the degeneracy map ``dmap``."""
    while base._base is not None:
        dmap = _compose(base._map, dmap)
        base = base._base
    if dmap == _identity_map(base._type):
        return base
    s = base._degeneracies.get(dmap)
    if s is None:
        s = Simploid(tuple(len(g) - 1 for _, g in dmap))
        s._base = base
        s._map = dmap
        base._degeneracies[dmap] = s
    return s


def _shift_factors(dmap: DegeneracyMap, removed: int) -> DegeneracyMap:
    return tuple((s - 1 if s > removed else s, g) for s, g in dmap)


class Simploid:
    r"""A simploid of a :class:`SimploidalSet`.

    Simploids are created by the construction methods of :class:`SimploidalSet`,
    not directly.  A non-degenerate simploid stores its faces explicitly.  A
    degenerate simploid is stored in canonical form, as a non-degenerate *base*
    simploid together with a :data:`DegeneracyMap`, so that two equal
    degenerate simploids are the same object.

    Examples
    --------
    >>> from torch_bsf.simploidal import SimploidalSet
    >>> S = SimploidalSet()
    >>> prism = S.add_simploid((2, 1))
    >>> prism.simploid_type, prism.dim
    ((2, 1), 3)
    >>> prism.face(1, 0).simploid_type  # a triangle
    (2,)
    >>> prism.face(0, 0).simploid_type  # a square
    (1, 1)
    >>> e = prism.face(1, 0).face(0, 0)  # an edge
    >>> e.degeneracy(0, 0).is_degenerate
    True
    >>> e.degeneracy(0, 0).face(0, 0) is e
    True
    """

    def __init__(self, simploid_type: Sequence[int]):
        self._type: SimploidType = _check_type(simploid_type)
        self._faces: dict[tuple[int, int], Simploid] = {}
        self._base: Simploid | None = None
        self._map: DegeneracyMap = ()
        self._degeneracies: dict[DegeneracyMap, Simploid] = {}

    def __repr__(self) -> str:
        kind = ", degenerate" if self.is_degenerate else ""
        return f"Simploid(type={self._type}{kind}, id=0x{id(self):x})"

    @property
    def simploid_type(self) -> SimploidType:
        r"""The type :math:`(a_1, \ldots, a_n)`."""
        return self._type

    @property
    def length(self) -> int:
        r"""The length :math:`n` of the type."""
        return len(self._type)

    @property
    def dim(self) -> int:
        r"""The dimension :math:`\sum_i a_i`."""
        return sum(self._type)

    @property
    def is_vertex(self) -> bool:
        """Whether this is a vertex, i.e., its type is ``()``."""
        return not self._type

    @property
    def is_degenerate(self) -> bool:
        """Whether this simploid is degenerate."""
        return self._base is not None

    @property
    def base(self) -> "Simploid":
        """The non-degenerate simploid this one is degenerate from, or itself if it is not degenerate."""
        return self._base if self._base is not None else self

    @property
    def degeneracy_map(self) -> DegeneracyMap:
        """The degeneracy map from :attr:`base` to this simploid (the identity map if not degenerate)."""
        return self._map if self._base is not None else _identity_map(self._type)

    def _check_face(self, i: int, j: int) -> None:
        if not (0 <= i < self.length and 0 <= j <= self._type[i]):
            raise IndexError(f"face ({i}, {j}) is not defined for a simploid of type {self._type}.")

    def face(self, i: int, j: int) -> "Simploid":
        r"""The face :math:`d^{i+1}_j`: removes vertex ``j`` of factor ``i``.

        Parameters
        ----------
        i
            The factor index, ``0 <= i < length``.
        j
            The vertex index, ``0 <= j <= a_i``.

        Returns
        -------
        The face.
        """
        self._check_face(i, j)
        if self._base is None:
            return self._faces[(i, j)]
        return self._degenerate_face(i, j)

    def _degenerate_face(self, k: int, v: int) -> "Simploid":
        base, dmap = cast(Simploid, self._base), self._map
        src, g = dmap[k]
        g2 = g[:v] + g[v + 1 :]
        before, after = dmap[:k], dmap[k + 1 :]
        if src < 0:
            return _canonical(base, before + (((-1, g2),) if len(g2) > 1 else ()) + after)
        if len(set(g2)) == len(set(g)):
            return _canonical(base, before + ((src, g2),) + after)
        # The face misses vertex m of factor src: it is degenerate from a face of the base.
        m = g[v]
        g3 = tuple(v if v < m else v - 1 for v in g2)
        if base._type[src] > 1:
            return _canonical(base.face(src, m), before + ((src, g3),) + after)
        middle = ((-1, g3),) if len(g3) > 1 else ()
        return _canonical(base.face(src, m), _shift_factors(before, src) + middle + _shift_factors(after, src))

    def faces(self) -> Iterator[tuple[tuple[int, int], "Simploid"]]:
        """Iterates ``((i, j), face(i, j))`` over all faces."""
        for i, a in enumerate(self._type):
            for j in range(a + 1):
                yield (i, j), self.face(i, j)

    def degeneracy(self, i: int, j: int) -> "Simploid":
        r"""The degenerate simploid :math:`s^{i+1}_j` (``j >= 0``) or :math:`s^i_{-1}` (``j == -1``).

        Parameters
        ----------
        i
            The factor index: ``0 <= i < length`` if ``j >= 0``, ``0 <= i <= length`` if ``j == -1``.
        j
            The vertex of factor ``i`` to repeat, or ``-1`` to insert a degenerate edge as the new factor ``i``.

        Returns
        -------
        The degenerate simploid.
        """
        dmap = self.degeneracy_map
        if j == -1:
            if not 0 <= i <= self.length:
                raise IndexError(f"degeneracy ({i}, -1) is not defined for a simploid of type {self._type}.")
            return _canonical(self.base, dmap[:i] + ((-1, (0, 0)),) + dmap[i:])
        self._check_face(i, j)
        src, g = dmap[i]
        g2 = tuple(g[x] if x <= j else g[x - 1] for x in range(len(g) + 1))
        return _canonical(self.base, dmap[:i] + ((src, g2),) + dmap[i + 1 :])

    def vertices(self) -> list["Simploid"]:
        """The distinct vertices of this simploid."""
        if self.is_vertex:
            return [self]
        seen: dict[Simploid, None] = {}
        for _, f in self.faces():
            seen.update(dict.fromkeys(f.vertices()))
        return list(seen)


class SimploidalSet:
    r"""A simploidal set :cite:p:`peltier2018simploidal`.

    A simploidal set is a set of simploids equipped with face and degeneracy
    operators satisfying the commutation relations of Definition 1 of the
    paper.  Only non-degenerate simploids are stored; degenerate simploids are
    created on demand by :meth:`Simploid.degeneracy` and by the faces of
    degenerate simploids.

    Simploidal sets are built with the basic operations of the paper:

    * :meth:`add_vertex`, :meth:`add_simplex`, :meth:`add_simploid`: complete cells with their boundary;
    * :meth:`product`: cartesian product of two simploidal sets;
    * :meth:`identify`: identification of two simploids (and of their boundaries);
    * :meth:`degenerate`: degeneracy of a simploid onto a degenerate simploid of one of its faces;
    * :meth:`cone`: cone with a new vertex.

    Examples
    --------
    A tetrahedron and a cube connected by a prism (Figure 1(c) of the paper):

    >>> from torch_bsf.simploidal import SimploidalSet
    >>> S = SimploidalSet()
    >>> tetra = S.add_simplex(3)
    >>> prism = S.add_simploid((2, 1))
    >>> cube = S.add_simploid((1, 1, 1))
    >>> _ = S.identify(tetra.face(0, 0), prism.face(1, 0))
    >>> _ = S.identify(prism.face(0, 0), cube.face(0, 0))
    >>> S.check()
    >>> [s.simploid_type for s in S.main_simploids()]
    [(3,), (2, 1), (1, 1, 1)]
    >>> len(S.simploids(0))  # 4 + 6 + 8 vertices, minus 3 and 4 identified ones
    11
    """

    def __init__(self) -> None:
        self._simploids: list[Simploid] = []

    def __len__(self) -> int:
        return len(self._simploids)

    def __iter__(self) -> Iterator[Simploid]:
        return iter(self._simploids)

    def __contains__(self, simploid: object) -> bool:
        if not isinstance(simploid, Simploid):
            return False
        base = simploid.base
        return any(s is base for s in self._simploids)

    def __repr__(self) -> str:
        return f"SimploidalSet(n_simploids={len(self)}, dim={self.dim})"

    @property
    def dim(self) -> int:
        """The maximum dimension of the simploids (``-1`` if empty)."""
        return max((s.dim for s in self._simploids), default=-1)

    def simploids(self, dim: int | None = None) -> list[Simploid]:
        """The non-degenerate simploids, optionally only those of a given dimension."""
        return [s for s in self._simploids if dim is None or s.dim == dim]

    def main_simploids(self) -> list[Simploid]:
        """The non-degenerate simploids that are not a face of another non-degenerate simploid."""
        faces = {id(f.base) for s in self._simploids for _, f in s.faces()}
        return [s for s in self._simploids if id(s) not in faces]

    def _add(self, simploids: Sequence[Simploid]) -> None:
        self._simploids.extend(simploids)
        self._simploids.sort(key=lambda s: s.dim)

    def add_vertex(self) -> Simploid:
        """Adds a new vertex.

        Returns
        -------
        The vertex.
        """
        v = Simploid(())
        self._add([v])
        return v

    def add_simplex(self, dim: int) -> Simploid:
        r"""Adds a complete simplex (a simploid of type :math:`(\mathrm{dim})`) with its boundary.

        Face ``face(0, j)`` of a simplex :math:`(v_0, \ldots, v_k)` is :math:`(v_0, \ldots, \hat v_j, \ldots, v_k)`.

        Parameters
        ----------
        dim
            The dimension of the simplex.

        Returns
        -------
        The simplex.

        Examples
        --------
        >>> from torch_bsf.simploidal import SimploidalSet
        >>> S = SimploidalSet()
        >>> tri = S.add_simplex(2)
        >>> len(S), len(tri.vertices())
        (7, 3)
        """
        if dim < 0:
            raise ValueError(f"dim must be non-negative, but {dim} is given.")
        cells: dict[tuple[int, ...], Simploid] = {}
        for k in range(dim + 1):
            for combo in combinations(range(dim + 1), k + 1):
                s = Simploid((k,) if k else ())
                for j in range(k + 1 if k else 0):
                    s._faces[(0, j)] = cells[combo[:j] + combo[j + 1 :]]
                cells[combo] = s
        self._add(list(cells.values()))
        return cells[tuple(range(dim + 1))]

    def add_simploid(self, simploid_type: Sequence[int]) -> Simploid:
        r"""Adds a complete simploid of the given type with its boundary.

        It is built as the cartesian product of complete simplices.

        Parameters
        ----------
        simploid_type
            The type :math:`(a_1, \ldots, a_n)`.

        Returns
        -------
        The simploid.

        Examples
        --------
        >>> from torch_bsf.simploidal import SimploidalSet
        >>> S = SimploidalSet()
        >>> square = S.add_simploid((1, 1))
        >>> len(S.simploids(0)), len(S.simploids(1)), len(S.simploids(2))
        (4, 4, 1)
        """
        st = _check_type(simploid_type)
        result = SimploidalSet()
        top = result.add_vertex()
        for a in st:
            factor = SimploidalSet()
            simplex = factor.add_simplex(a)
            result, pairs = result.product(factor)
            top = pairs[(top, simplex)]
        self._add(result._simploids)
        return top

    def product(self, other: "SimploidalSet") -> tuple["SimploidalSet", dict[tuple[Simploid, Simploid], Simploid]]:
        r"""The cartesian product of two simploidal sets (Algorithm 7 of the paper).

        The product of simploids of types :math:`(a_1, \ldots, a_k)` and
        :math:`(b_1, \ldots, b_l)` is a simploid of type
        :math:`(a_1, \ldots, a_k, b_1, \ldots, b_l)`.

        Parameters
        ----------
        other
            The second factor.

        Returns
        -------
        product
            A new simploidal set.
        pairs
            Maps each pair ``(sigma1, sigma2)`` of non-degenerate simploids to ``sigma1 x sigma2``.

        Examples
        --------
        >>> from torch_bsf.simploidal import SimploidalSet
        >>> A, B = SimploidalSet(), SimploidalSet()
        >>> tri, edge = A.add_simplex(2), B.add_simplex(1)
        >>> P, pairs = A.product(B)
        >>> pairs[(tri, edge)].simploid_type
        (2, 1)
        """
        result = SimploidalSet()
        pairs: dict[tuple[Simploid, Simploid], Simploid] = {}
        for s1 in self._simploids:
            for s2 in other._simploids:
                pairs[(s1, s2)] = Simploid(s1._type + s2._type)

        def prod(x: Simploid, y: Simploid) -> Simploid:
            n = x.base.length
            dmap = x.degeneracy_map + tuple((s + n if s >= 0 else s, g) for s, g in y.degeneracy_map)
            return _canonical(pairs[(x.base, y.base)], dmap)

        for (s1, s2), s in pairs.items():
            for (i, j), f in s1.faces():
                s._faces[(i, j)] = prod(f, s2)
            for (i, j), f in s2.faces():
                s._faces[(s1.length + i, j)] = prod(s1, f)
        result._add(list(pairs.values()))
        return result, pairs

    def _closure(self, simploids: Sequence[Simploid] | None) -> list[Simploid]:
        """The non-degenerate simploids of ``simploids`` and of their boundaries."""
        if simploids is None:
            return list(self._simploids)
        members = {id(s) for s in self._simploids}
        seen: dict[Simploid, None] = {}
        stack = [s.base for s in simploids]
        while stack:
            s = stack.pop()
            if s in seen:
                continue
            if id(s) not in members:
                raise ValueError(f"{s!r} does not belong to this simploidal set.")
            seen[s] = None
            stack.extend(f.base for _, f in s.faces())
        return sorted(seen, key=lambda s: s.dim)

    def cone(self, simploids: Sequence[Simploid] | None = None) -> tuple[Simploid, dict[Simploid, Simploid]]:
        r"""Cones simploids with a new vertex (Algorithm 8 of the paper).

        The cone of a simploid of type :math:`(a_1, \ldots, a_k)` is a simploid
        of type :math:`(a_1, \ldots, a_k, 1)`: an extrusion whose far end is
        degenerated into the new vertex ``v``.  For example, the cone of a
        square is a square-based pyramid of type :math:`(1, 1, 1)`.  The cone
        ``c`` of ``sigma`` satisfies ``c.face(k, 1) is sigma`` and
        ``c.face(k, 0)`` is the degeneracy of ``v`` with the type of ``sigma``,
        where ``k = sigma.length``.

        Parameters
        ----------
        simploids
            The simploids to cone (their boundaries are coned too).  All simploids if ``None``.

        Returns
        -------
        vertex
            The new vertex.
        cones
            Maps each coned non-degenerate simploid to its cone.

        Examples
        --------
        >>> from torch_bsf.simploidal import SimploidalSet
        >>> S = SimploidalSet()
        >>> square = S.add_simploid((1, 1))
        >>> v, cones = S.cone([square])
        >>> pyramid = cones[square]
        >>> pyramid.simploid_type
        (1, 1, 1)
        >>> pyramid.face(2, 0).is_degenerate, pyramid.face(2, 0).base is v
        (True, True)
        """
        base = self._closure(simploids)
        v = Simploid(())
        cones = {s: Simploid(s._type + (1,)) for s in base}

        def cone_of(x: Simploid) -> Simploid:
            if x.is_degenerate:
                return _canonical(cones[x.base], x._map + ((x.base.length, (0, 1)),))
            return cones[x]

        for s, c in cones.items():
            for (i, j), f in s.faces():
                c._faces[(i, j)] = cone_of(f)
            c._faces[(s.length, 1)] = s
            c._faces[(s.length, 0)] = _canonical(v, tuple((-1, (0,) * (a + 1)) for a in s._type))
        self._add([v, *cones.values()])
        return v, cones

    def identify(self, a: Simploid, b: Simploid) -> Simploid:
        r"""Identifies two simploids of the same type.

        This is the higher-level identification of the paper: the boundaries
        of ``a`` and ``b`` are identified face by face (``a.face(i, j)`` with
        ``b.face(i, j)``), and the degenerate simploids induced by the merged
        simploids are identified too.  If one of the identified simploids is
        degenerate, the result is degenerate (this is how :meth:`degenerate`
        works).

        References to simploids that are merged away become stale; use the
        returned simploid and the faces of the surviving simploids instead.

        Parameters
        ----------
        a, b
            The simploids to identify.

        Returns
        -------
        The simploid resulting from the identification.

        Raises
        ------
        ValueError
            If the simploids (or faces to be identified) have different types, do not
            belong to this set, or if two distinct degenerate simploids would be identified.

        Examples
        --------
        Glue two triangles along an edge:

        >>> from torch_bsf.simploidal import SimploidalSet
        >>> S = SimploidalSet()
        >>> t1, t2 = S.add_simplex(2), S.add_simplex(2)
        >>> e = S.identify(t1.face(0, 0), t2.face(0, 2))
        >>> t1.face(0, 0) is e and t2.face(0, 2) is e
        True
        >>> len(S.simploids(0)), len(S.simploids(1))
        (4, 5)
        """
        for s in (a, b):
            if s not in self:
                raise ValueError(f"{s!r} does not belong to this simploidal set.")
        classes = self._identification_classes(a, b)
        order = {id(s): k for k, s in enumerate(self._simploids)}
        final: dict[Simploid, Simploid] = {}
        for cls in sorted(classes, key=lambda c: c[0].dim):
            final.update(dict.fromkeys(cls, _representative(cls, final, order)))
        self._simploids = [s for s in self._simploids if final.get(s, s) is s]
        for s in self._simploids:
            for key, f in s._faces.items():
                s._faces[key] = _resolve(f, final)
        return _resolve(a, final)

    @staticmethod
    def _identification_classes(a: Simploid, b: Simploid) -> list[list[Simploid]]:
        """The equivalence classes of the simploids identified by identifying ``a`` and ``b``."""
        parent: dict[Simploid, Simploid] = {}

        def find(x: Simploid) -> Simploid:
            while parent.get(x, x) is not x:
                x = parent[x]
            return x

        queue = [(a, b)]
        while queue:
            x, y = queue.pop()
            if x._type != y._type:
                raise ValueError(f"Cannot identify simploids of different types {x._type} and {y._type}.")
            rx, ry = find(x), find(y)
            if rx is ry:
                continue
            parent.setdefault(rx, rx)
            parent[ry] = rx
            queue.extend((fx, y.face(i, j)) for (i, j), fx in x.faces())
        classes: dict[Simploid, list[Simploid]] = {}
        for x in parent:
            classes.setdefault(find(x), []).append(x)
        return list(classes.values())

    def degenerate(self, simploid: Simploid, face: Simploid, i: int, j: int) -> Simploid:
        r"""Degenerates a simploid onto a degenerate simploid of one of its faces.

        This identifies ``simploid`` with ``face.degeneracy(i, j)``; the two must have the same type,
        and their boundaries are identified accordingly.

        Parameters
        ----------
        simploid
            The simploid to degenerate.
        face
            The simploid it degenerates into.
        i, j
            The degeneracy operator, as in :meth:`Simploid.degeneracy`.

        Returns
        -------
        The resulting degenerate simploid.

        Examples
        --------
        Collapse an edge of a square into a vertex, which turns the square
        into a triangle-shaped simploid of type ``(1, 1)``:

        >>> from torch_bsf.simploidal import SimploidalSet
        >>> S = SimploidalSet()
        >>> square = S.add_simploid((1, 1))
        >>> edge = square.face(0, 0)  # an edge of type (1,)
        >>> d = S.degenerate(edge, edge.face(0, 0), 0, -1)
        >>> d.is_degenerate, square.face(0, 0) is d
        (True, True)
        >>> len(S.simploids(0))
        3
        """
        return self.identify(simploid, face.degeneracy(i, j))

    def check(self) -> None:
        r"""Checks the face commutation relations (II) of Definition 1 for every simploid.

        Raises
        ------
        ValueError
            If a relation does not hold or a face does not belong to this set.
        """
        members = {id(s) for s in self._simploids}
        for s in self._simploids:
            for (i, j), f in s.faces():
                if id(f.base) not in members:
                    raise ValueError(f"Face ({i}, {j}) of {s!r} does not belong to this simploidal set.")
                if f._type != _face_type(s._type, i):
                    raise ValueError(f"Face ({i}, {j}) of {s!r} has a wrong type {f._type}.")
            _check_commutation(s)

    def boundary_matrix(self, dim: int) -> torch.Tensor:
        r"""The matrix of the boundary operator :math:`\partial_{\mathrm{dim}}` of the normalized chain complex.

        Following Section 4.4 of :cite:t:`peltier2018simploidal`, the chain
        group :math:`C_k` is spanned by the non-degenerate simploids of
        dimension :math:`k`, and the boundary of a simploid :math:`\sigma` of
        type :math:`(a_1, \ldots, a_n)` is

        .. math::

           \partial \sigma = \sum_{i=1}^n \sum_{j=0}^{a_i} (-1)^{a_1 + \cdots + a_{i-1} + j} d^i_j \sigma,

        the Koszul sign rule for the product :math:`\Delta^{a_1} \times \cdots \times \Delta^{a_n}`,
        where the terms whose face :math:`d^i_j \sigma` is degenerate are dropped.

        Parameters
        ----------
        dim
            The dimension :math:`k \ge 0` of the simploids whose boundary is taken.

        Returns
        -------
        An integer (``torch.long``) matrix of shape ``(len(simploids(dim - 1)), len(simploids(dim)))``:
        column ``c`` holds the boundary of ``simploids(dim)[c]`` in the basis ``simploids(dim - 1)``.
        ``boundary_matrix(0)`` has no rows.

        Raises
        ------
        ValueError
            If ``dim`` is negative or a face does not belong to this set.

        Examples
        --------
        >>> from torch_bsf.simploidal import SimploidalSet
        >>> S = SimploidalSet()
        >>> square = S.add_simploid((1, 1))
        >>> [[key for key, f in square.faces() if f is e] for e in S.simploids(1)]
        [[(0, 1)], [(0, 0)], [(1, 1)], [(1, 0)]]
        >>> S.boundary_matrix(2)
        tensor([[-1],
                [ 1],
                [ 1],
                [-1]])
        >>> bool((S.boundary_matrix(1) @ S.boundary_matrix(2) == 0).all())
        True
        """
        if dim < 0:
            raise ValueError(f"dim must be non-negative, but {dim} is given.")
        rows = {s: r for r, s in enumerate(self.simploids(dim - 1))}
        columns = self.simploids(dim)
        matrix = torch.zeros(len(rows), len(columns), dtype=torch.long)
        for c, s in enumerate(columns):
            for (i, j), f in s.faces():
                if f.is_degenerate:
                    continue
                if f not in rows:
                    raise ValueError(f"Face ({i}, {j}) of {s!r} does not belong to this simploidal set.")
                matrix[rows[f], c] += (-1) ** (sum(s.simploid_type[:i]) + j)
        return matrix

    def betti_numbers(self, field: Literal["Q", "Z2"] = "Q") -> list[int]:
        r"""The Betti numbers :math:`\beta_0, \ldots, \beta_{\mathrm{dim}}` of the simploidal set.

        They are the dimensions of the homology groups of the normalized chain
        complex (see :meth:`boundary_matrix`),
        :math:`\beta_k = \dim C_k - \operatorname{rank} \partial_k - \operatorname{rank} \partial_{k+1}`.
        The ranks are computed exactly by integer Gaussian elimination, so the
        result does not depend on a floating-point tolerance.

        Parameters
        ----------
        field
            The coefficient field: ``"Q"`` (rational numbers) or ``"Z2"``
            (integers modulo 2).  They differ only when the integral homology has
            torsion, e.g., for non-orientable spaces.

        Returns
        -------
        The Betti numbers, one per dimension from ``0`` to :attr:`dim` (empty if the set is empty).

        Raises
        ------
        ValueError
            If ``field`` is neither ``"Q"`` nor ``"Z2"``.

        Examples
        --------
        A torus, a square whose opposite edges are identified:

        >>> from torch_bsf.simploidal import SimploidalSet
        >>> S = SimploidalSet()
        >>> square = S.add_simploid((1, 1))
        >>> _ = S.identify(square.face(0, 0), square.face(0, 1))
        >>> _ = S.identify(square.face(1, 0), square.face(1, 1))
        >>> S.betti_numbers()
        [1, 2, 1]
        """
        if field not in ("Q", "Z2"):
            raise ValueError(f'field must be "Q" or "Z2", but {field!r} is given.')
        modulus = 2 if field == "Z2" else None
        ranks = [_rank(self.boundary_matrix(k), modulus) for k in range(self.dim + 2)]
        return [len(self.simploids(k)) - ranks[k] - ranks[k + 1] for k in range(self.dim + 1)]


def _rank(matrix: torch.Tensor, modulus: int | None) -> int:
    """The exact rank of an integer matrix over the rationals (``modulus=None``) or over the integers modulo a prime."""
    rows = [_reduce(r, modulus) for r in cast(list[list[int]], matrix.tolist())]
    rows = [r for r in rows if any(r)]
    rank = 0
    while rows:
        pivot = rows.pop()
        c = next(k for k, x in enumerate(pivot) if x)
        rows = [_eliminate(r, pivot, c, modulus) for r in rows]
        rows = [r for r in rows if any(r)]
        rank += 1
    return rank


def _eliminate(row: list[int], pivot: list[int], c: int, modulus: int | None) -> list[int]:
    """Cancels the entry of ``row`` in column ``c`` with ``pivot``, whose entry there is nonzero."""
    if not row[c]:
        return row
    p, q = pivot[c], row[c]
    return _reduce([p * x - q * y for x, y in zip(row, pivot, strict=True)], modulus)


def _reduce(row: list[int], modulus: int | None) -> list[int]:
    """Reduces a row modulo ``modulus``, or divides it by the gcd of its entries if ``modulus`` is ``None``."""
    if modulus is not None:
        return [x % modulus for x in row]
    g = gcd(*row)
    return [x // g for x in row] if g > 1 else row


def _resolve(x: Simploid, final: dict[Simploid, Simploid]) -> Simploid:
    """The simploid that ``x`` becomes after an identification described by ``final``."""
    if x in final:
        return final[x]
    if x._base is not None:
        r = _resolve(x._base, final)
        return x if r is x._base else _canonical(r, x._map)
    return x


def _representative(cls: list[Simploid], final: dict[Simploid, Simploid], order: dict[int, int]) -> Simploid:
    """The simploid an identification class is merged into; degenerate if any member is degenerate."""
    degenerate = [_canonical(_resolve(cast(Simploid, m._base), final), m._map) for m in cls if m.is_degenerate]
    if not degenerate:
        return min(cls, key=lambda m: order.get(id(m), len(order)))
    if any(d is not degenerate[0] for d in degenerate):  # pragma: no cover - defensive
        raise ValueError("Two distinct degenerate simploids cannot be identified.")
    return degenerate[0]


def _face_type(simploid_type: SimploidType, i: int) -> SimploidType:
    a = simploid_type[i]
    if a > 1:
        return simploid_type[:i] + (a - 1,) + simploid_type[i + 1 :]
    return simploid_type[:i] + simploid_type[i + 1 :]


def _check_commutation(s: Simploid) -> None:
    """Checks the relations (II) of Definition 1 on all pairs of faces of ``s``."""
    st = s.simploid_type
    for i, a in enumerate(st):
        for j in range(a + 1):
            for q in range(j if a > 1 else 0):
                if s.face(i, j).face(i, q) is not s.face(i, q).face(i, j - 1):
                    raise ValueError(f"d({i},{j}) d({i},{q}) != d({i},{q}) d({i},{j - 1}) on {s!r}.")
            for k in range(i):
                i2 = i - 1 if st[k] == 1 else i
                for q in range(st[k] + 1):
                    if s.face(i, j).face(k, q) is not s.face(k, q).face(i2, j):
                        raise ValueError(f"d({i},{j}) d({k},{q}) != d({k},{q}) d({i2},{j}) on {s!r}.")
