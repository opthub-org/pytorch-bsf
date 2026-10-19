r"""Simploidal sets and simploidal Bézier spaces.

This module implements the simploidal sets of :cite:t:`peltier2018simploidal`
and the fitting of the Bézier spaces they carry.

A *simploid* of type :math:`(a_1, \ldots, a_n)` is a cartesian product
:math:`\Delta^{a_1} \times \cdots \times \Delta^{a_n}` of simplices; simplices
(type :math:`(a)`) and cubes (type :math:`(1, \ldots, 1)`) are special cases.
A *Bézier simploid* of type :math:`(a_1, \ldots, a_n)` and degree
:math:`(d_1, \ldots, d_n)` is the tensor product of Bézier simplices

.. math::

   B(\mathbf u^1, \ldots, \mathbf u^n)
   = \sum_{\boldsymbol\alpha^1 \in \Gamma^{a_1}_{d_1}} \cdots \sum_{\boldsymbol\alpha^n \in \Gamma^{a_n}_{d_n}}
     P_{(\boldsymbol\alpha^1, \ldots, \boldsymbol\alpha^n)}
     \prod_{i=1}^n B^{d_i}_{\boldsymbol\alpha^i}(\mathbf u^i),

where each :math:`\mathbf u^i` is a point of the standard :math:`a_i`-simplex
given in barycentric coordinates.  Throughout this module the parameter of a
simploid is the concatenation :math:`(\mathbf u^1, \ldots, \mathbf u^n)`, a
vector of length :math:`\sum_i (a_i + 1)`.

A *simploidal set* (:class:`SimploidalSet`) is an assembly of simploids glued
along common faces, described by face operators :math:`d^i_j` and degeneracy
operators :math:`s^i_j`.  A *simploidal Bézier space*
(:class:`SimploidalBezierSpace`) associates a Bézier simploid with every
simploid of a simploidal set.  As in Section 4.3 of the paper, only the
*proper* control points of each non-degenerate simploid are stored, so control
points on shared faces are stored once and the space is continuous across
them by construction.

Indexing convention
-------------------
Factor indices are 0-based, whereas the paper uses 1-based ones:

* ``face(i, j)`` is the paper's :math:`d^{i+1}_j` (remove vertex ``j`` of factor ``i``);
* ``degeneracy(i, j)`` with ``j >= 0`` is :math:`s^{i+1}_j` (repeat vertex ``j`` of factor ``i``);
* ``degeneracy(i, -1)`` is :math:`s^i_{-1}` (insert a degenerate edge as the new factor ``i``).

Examples
--------
Glue a triangle and a square along an edge and fit the resulting space:

>>> import torch
>>> from torch_bsf.simploidal import SimploidalSet, fit_simploidal_set, sample_simploid
>>> S = SimploidalSet()
>>> triangle = S.add_simplex(2)
>>> square = S.add_simploid((1, 1))
>>> edge = S.identify(triangle.face(0, 0), square.face(0, 0))
>>> S.check()
>>> t1, t2 = sample_simploid((2,), 50, seed=0), sample_simploid((1, 1), 50, seed=1)
>>> space = fit_simploidal_set(S, [(triangle, t1, t1[:, :1]), (square, t2, t2[:, 1:2])], degree=2)
>>> space.n_values
1
"""

from functools import lru_cache
from itertools import combinations
from itertools import product as iproduct
from math import gcd
from typing import Any, Iterator, Literal, Mapping, Sequence, TypeAlias, cast

import lightning.pytorch as L
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from torch_bsf.bezier_simplex import polynom
from torch_bsf.control_points import simplex_indices

#: The type :math:`(a_1, \ldots, a_n)` of a simploid; ``()`` is a vertex.
SimploidType: TypeAlias = tuple[int, ...]

#: The index of a control point of a Bézier simploid: a tuple of multi-indices, one per factor.
SimploidIndex: TypeAlias = tuple[tuple[int, ...], ...]

#: A degeneracy map: for each factor of a degenerate simploid, the factor of its base it comes
#: from (``-1`` for an inserted edge factor) and the surjection on the vertices of that factor.
DegeneracyMap: TypeAlias = tuple[tuple[int, tuple[int, ...]], ...]

Solver = Literal["lstsq", "trainer"]


# ---------------------------------------------------------------------------
# Index utilities
# ---------------------------------------------------------------------------


def _check_type(simploid_type: Sequence[int]) -> SimploidType:
    t = tuple(int(a) for a in simploid_type)
    if any(a < 1 for a in t):
        raise ValueError(f"Every component of a simploid type must be positive, but {t} is given.")
    return t


def _check_degree(degree: int | Sequence[int], length: int) -> tuple[int, ...]:
    ds = (int(degree),) * length if isinstance(degree, int) else tuple(int(d) for d in degree)
    if len(ds) != length:
        raise ValueError(f"degree must have {length} components, but {ds} is given.")
    if any(d < 0 for d in ds):
        raise ValueError(f"degree must be non-negative, but {ds} is given.")
    return ds


@lru_cache(maxsize=None)
def _indices(simploid_type: SimploidType, degree: tuple[int, ...]) -> tuple[SimploidIndex, ...]:
    return tuple(iproduct(*(tuple(simplex_indices(a + 1, d)) for a, d in zip(simploid_type, degree))))


@lru_cache(maxsize=None)
def _positions(simploid_type: SimploidType, degree: tuple[int, ...]) -> dict[SimploidIndex, int]:
    return {alpha: pos for pos, alpha in enumerate(_indices(simploid_type, degree))}


def simploid_indices(simploid_type: Sequence[int], degree: int | Sequence[int]) -> list[SimploidIndex]:
    r"""Lists the control point indices of a Bézier simploid in canonical order.

    The order is the lexicographic order of the tuples of multi-indices, the
    first factor varying slowest (Algorithm 4 of :cite:t:`peltier2018simploidal`);
    the multi-indices of each factor follow :func:`torch_bsf.control_points.simplex_indices`.

    Parameters
    ----------
    simploid_type
        The type :math:`(a_1, \ldots, a_n)`.
    degree
        The degree of every factor, or one degree per factor.

    Returns
    -------
    The indices.

    Examples
    --------
    >>> from torch_bsf.simploidal import simploid_indices
    >>> simploid_indices((1, 1), 1)
    [((1, 0), (1, 0)), ((1, 0), (0, 1)), ((0, 1), (1, 0)), ((0, 1), (0, 1))]
    >>> simploid_indices((), 3)
    [()]
    """
    t = _check_type(simploid_type)
    return list(_indices(t, _check_degree(degree, len(t))))


def simploid_n_params(simploid_type: Sequence[int]) -> int:
    r"""The length :math:`\sum_i (a_i + 1)` of the parameter vector of a simploid.

    Examples
    --------
    >>> from torch_bsf.simploidal import simploid_n_params
    >>> simploid_n_params((2, 1))
    5
    """
    return sum(a + 1 for a in _check_type(simploid_type))


def bernstein_basis(simploid_type: Sequence[int], degree: int | Sequence[int], t: torch.Tensor) -> torch.Tensor:
    r"""Evaluates the tensor-product Bernstein basis of a Bézier simploid.

    Parameters
    ----------
    simploid_type
        The type :math:`(a_1, \ldots, a_n)`.
    degree
        The degree of every factor, or one degree per factor.
    t
        Parameters of shape ``(batch, sum(a_i + 1))``: the concatenated
        barycentric coordinates of every factor.

    Returns
    -------
    The basis of shape ``(batch, n_indices)``, columns in the order of :func:`simploid_indices`.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf.simploidal import bernstein_basis
    >>> bernstein_basis((1, 1), 1, torch.tensor([[0.5, 0.5, 1.0, 0.0]]))
    tensor([[0.5000, 0.0000, 0.5000, 0.0000]])
    """
    st = _check_type(simploid_type)
    ds = _check_degree(degree, len(st))
    if t.dim() != 2 or t.shape[1] != sum(a + 1 for a in st):
        raise ValueError(f"t must have shape (batch, {sum(a + 1 for a in st)}), but {tuple(t.shape)} is given.")
    dtype = t.dtype if t.is_floating_point() else torch.get_default_dtype()
    t = t.to(dtype)
    basis = torch.ones(t.shape[0], 1, dtype=dtype, device=t.device)
    offset = 0
    for a, d in zip(st, ds):
        idx = list(simplex_indices(a + 1, d))
        powers = torch.tensor(idx, dtype=torch.long, device=t.device)
        coeffs = torch.tensor([polynom(d, i) for i in idx], dtype=dtype, device=t.device)
        block = t[:, offset : offset + a + 1]
        factor = coeffs * torch.pow(block.unsqueeze(1), powers.unsqueeze(0)).prod(dim=-1)
        basis = (basis.unsqueeze(2) * factor.unsqueeze(1)).reshape(t.shape[0], -1)
        offset += a + 1
    return basis


def sample_simploid(simploid_type: Sequence[int], n_samples: int, seed: int | None = None) -> torch.Tensor:
    r"""Samples points uniformly on a simploid.

    Each factor is sampled from the flat Dirichlet distribution.

    Parameters
    ----------
    simploid_type
        The type :math:`(a_1, \ldots, a_n)`.
    n_samples
        The number of samples.
    seed
        Random seed for reproducibility.

    Returns
    -------
    Parameters of shape ``(n_samples, sum(a_i + 1))``.

    Examples
    --------
    >>> from torch_bsf.simploidal import sample_simploid
    >>> t = sample_simploid((2, 1), 4, seed=0)
    >>> t.shape
    torch.Size([4, 5])
    >>> bool(((t[:, :3].sum(dim=1) - 1).abs() < 1e-6).all())
    True
    """
    st = _check_type(simploid_type)
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, but {n_samples} is given.")
    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    blocks = [torch.empty(n_samples, 0)]
    for a in st:
        e = -torch.log(torch.rand(n_samples, a + 1, generator=generator).clamp_min(1e-12))
        blocks.append(e / e.sum(dim=1, keepdim=True))
    return torch.cat(blocks, dim=1).to(torch.get_default_dtype())


def _face_index(alpha: SimploidIndex, i: int, j: int) -> SimploidIndex:
    """The index in the face ``d^i_j`` of a control point index whose ``alpha[i][j]`` is zero."""
    a = alpha[i]
    if len(a) == 2:
        return alpha[:i] + alpha[i + 1 :]
    return alpha[:i] + (a[:j] + a[j + 1 :],) + alpha[i + 1 :]


def _pullback_index(alpha: SimploidIndex, dmap: DegeneracyMap, base_length: int) -> SimploidIndex:
    """The control point index of the base simploid that equals ``alpha`` of a degenerate simploid (Property 2)."""
    beta: list[tuple[int, ...]] = [()] * base_length
    for a, (src, g) in zip(alpha, dmap):
        if src < 0:
            continue
        b = [0] * (max(g) + 1)
        for x, v in enumerate(g):
            b[v] += a[x]
        beta[src] = tuple(b)
    return tuple(beta)


# ---------------------------------------------------------------------------
# Combinatorial structure
# ---------------------------------------------------------------------------


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
    return _reduce([p * x - q * y for x, y in zip(row, pivot)], modulus)


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


# ---------------------------------------------------------------------------
# Bézier models
# ---------------------------------------------------------------------------


class _LeastSquaresModule(L.LightningModule):
    """Training, validation and test steps shared by the Bézier simploid models."""

    def _predict_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def training_step(self, batch, batch_idx) -> dict[str, Any]:
        y_hat, y = self._predict_batch(batch)
        mse = F.mse_loss(y_hat, y)
        self.log("train_mse", mse, sync_dist=True)
        return {"loss": mse}

    def validation_step(self, batch, batch_idx) -> None:
        y_hat, y = self._predict_batch(batch)
        self.log("val_mse", F.mse_loss(y_hat, y), sync_dist=True)
        self.log("val_mae", F.l1_loss(y_hat, y), sync_dist=True)

    def test_step(self, batch, batch_idx) -> dict[str, Any]:
        y_hat, y = self._predict_batch(batch)
        mse = F.mse_loss(y_hat, y)
        self.log("test_mse", mse, sync_dist=True)
        self.log("test_mae", F.l1_loss(y_hat, y), sync_dist=True)
        return {"test_loss": mse}

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.LBFGS(self.parameters())


class BezierSimploid(_LeastSquaresModule):
    r"""A Bézier simploid.

    Parameters
    ----------
    simploid_type
        The type :math:`(a_1, \ldots, a_n)`.  ``()`` gives a constant (a vertex).
    degree
        The degree of every factor, or one degree per factor.
    control_points
        The control points: a tensor of shape ``(n_indices, n_values)`` whose rows follow
        :func:`simploid_indices`, or a mapping from every index to its value.
        Zeros if ``None``, in which case ``n_values`` is required.
    n_values
        The number of values (target dimension); used only when ``control_points`` is ``None``.

    Examples
    --------
    A bilinear patch (a simploid of type ``(1, 1)``):

    >>> import torch
    >>> from torch_bsf.simploidal import BezierSimploid
    >>> patch = BezierSimploid((1, 1), 1, {
    ...     ((1, 0), (1, 0)): [0.0, 0.0, 0.0],
    ...     ((1, 0), (0, 1)): [0.0, 1.0, 0.0],
    ...     ((0, 1), (1, 0)): [1.0, 0.0, 0.0],
    ...     ((0, 1), (0, 1)): [1.0, 1.0, 1.0],
    ... })
    >>> patch(torch.tensor([[0.5, 0.5, 0.5, 0.5]]))
    tensor([[0.5000, 0.5000, 0.2500]], grad_fn=<MmBackward0>)
    """

    def __init__(
        self,
        simploid_type: Sequence[int],
        degree: int | Sequence[int],
        control_points: torch.Tensor | Mapping[SimploidIndex, Sequence[float] | torch.Tensor] | None = None,
        *,
        n_values: int | None = None,
    ):
        super().__init__()
        st = _check_type(simploid_type)
        ds = _check_degree(degree, len(st))
        indices = _indices(st, ds)
        if control_points is None:
            if n_values is None:
                raise TypeError("n_values is required when control_points is None.")
            matrix = torch.zeros(len(indices), n_values)
        elif isinstance(control_points, Mapping):
            missing = [idx for idx in indices if idx not in control_points]
            if missing:
                raise ValueError(f"Control points are missing for indices {missing}.")
            matrix = torch.stack([torch.as_tensor(control_points[idx]) for idx in indices])
        else:
            matrix = torch.as_tensor(control_points)
        if matrix.dim() != 2 or matrix.shape[0] != len(indices):
            raise ValueError(f"control_points must have shape ({len(indices)}, n_values), but {tuple(matrix.shape)} is given.")
        if not matrix.is_floating_point():
            matrix = matrix.to(torch.get_default_dtype())
        self._simploid_type = st
        self._degree = ds
        self.control_points = nn.Parameter(matrix.detach().clone())
        # Store the actual shape so that load_from_checkpoint can rebuild a placeholder.
        simploid_type = list(st)
        degree = list(ds)
        n_values = int(matrix.shape[1])
        self.save_hyperparameters(ignore=["control_points"])

    @classmethod
    def zeros(cls, simploid_type: Sequence[int], degree: int | Sequence[int], n_values: int) -> "BezierSimploid":
        """A Bézier simploid whose control points are all zero."""
        return cls(simploid_type, degree, n_values=n_values)

    @classmethod
    def rand(cls, simploid_type: Sequence[int], degree: int | Sequence[int], n_values: int) -> "BezierSimploid":
        r"""A Bézier simploid whose control points are uniformly distributed on :math:`[0, 1)`."""
        n = len(simploid_indices(simploid_type, degree))
        return cls(simploid_type, degree, torch.rand(n, n_values))

    @classmethod
    def randn(cls, simploid_type: Sequence[int], degree: int | Sequence[int], n_values: int) -> "BezierSimploid":
        """A Bézier simploid whose control points follow the standard normal distribution."""
        n = len(simploid_indices(simploid_type, degree))
        return cls(simploid_type, degree, torch.randn(n, n_values))

    def extra_repr(self) -> str:
        return f"simploid_type={self._simploid_type}, degree={self._degree}, n_values={self.n_values}"

    @property
    def simploid_type(self) -> SimploidType:
        r"""The type :math:`(a_1, \ldots, a_n)`."""
        return self._simploid_type

    @property
    def degree(self) -> tuple[int, ...]:
        """The degree of each factor."""
        return self._degree

    @property
    def n_params(self) -> int:
        r"""The length :math:`\sum_i (a_i + 1)` of the parameter vector."""
        return simploid_n_params(self._simploid_type)

    @property
    def n_values(self) -> int:
        """The number of values (target dimension)."""
        return int(self.control_points.shape[1])

    def indices(self) -> list[SimploidIndex]:
        """The control point indices in the order of the rows of :attr:`control_points`."""
        return list(_indices(self._simploid_type, self._degree))

    def __getitem__(self, index: SimploidIndex) -> torch.Tensor:
        return self.control_points[_positions(self._simploid_type, self._degree)[tuple(map(tuple, index))]]

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        r"""Evaluates the Bézier simploid.

        Parameters
        ----------
        t
            Parameters of shape ``(batch, n_params)``.

        Returns
        -------
        Values of shape ``(batch, n_values)``.
        """
        t = torch.as_tensor(t, dtype=self.control_points.dtype, device=self.control_points.device)
        return bernstein_basis(self._simploid_type, self._degree, t) @ self.control_points

    def _predict_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        return self.forward(x), y

    def _derived(self, simploid_type: SimploidType, degree: tuple[int, ...], parent: list[SimploidIndex]) -> "BezierSimploid":
        pos = _positions(self._simploid_type, self._degree)
        rows = torch.tensor([pos[alpha] for alpha in parent], dtype=torch.long, device=self.control_points.device)
        return BezierSimploid(simploid_type, degree, self.control_points.detach()[rows])

    def face(self, i: int, j: int) -> "BezierSimploid":
        r"""The Bézier simploid of the face ``d^{i+1}_j`` (Property 2-1 of :cite:t:`peltier2018simploidal`).

        Its control points are those whose multi-index of factor ``i`` has a zero at position ``j``.

        Examples
        --------
        >>> import torch
        >>> from torch_bsf.simploidal import BezierSimploid
        >>> prism = BezierSimploid.randn((2, 1), 3, n_values=2)
        >>> bottom = prism.face(1, 1)  # the triangle at u^2 = (1, 0)
        >>> u = torch.tensor([[0.2, 0.3, 0.5]])
        >>> torch.allclose(prism(torch.cat([u, torch.tensor([[1.0, 0.0]])], dim=1)), bottom(u))
        True
        """
        if not (0 <= i < len(self._simploid_type) and 0 <= j <= self._simploid_type[i]):
            raise IndexError(f"face ({i}, {j}) is not defined for a simploid of type {self._simploid_type}.")
        a, d = self._simploid_type[i], self._degree[i]
        if a > 1:
            st = self._simploid_type[:i] + (a - 1,) + self._simploid_type[i + 1 :]
            ds = self._degree
        else:
            st = self._simploid_type[:i] + self._simploid_type[i + 1 :]
            ds = self._degree[:i] + self._degree[i + 1 :]

        def parent(alpha: SimploidIndex) -> SimploidIndex:
            if a > 1:
                m = alpha[i]
                return alpha[:i] + (m[:j] + (0,) + m[j:],) + alpha[i + 1 :]
            return alpha[:i] + (((0, d) if j == 0 else (d, 0)),) + alpha[i:]

        return self._derived(st, ds, [parent(alpha) for alpha in _indices(st, ds)])

    def degenerate(self, i: int, j: int, degree: int | None = None) -> "BezierSimploid":
        r"""The Bézier simploid of the degenerate simploid ``s^{i+1}_j`` or ``s^i_{-1}`` (Property 2-2).

        The result has the same shape as this Bézier simploid.

        Parameters
        ----------
        i, j
            The degeneracy operator, as in :meth:`Simploid.degeneracy`.
        degree
            The degree of the inserted edge factor when ``j == -1``.
            Defaults to the common degree of the factors.

        Examples
        --------
        >>> import torch
        >>> from torch_bsf.simploidal import BezierSimploid
        >>> curve = BezierSimploid.randn((1,), 3, n_values=2)
        >>> triangle = curve.degenerate(0, 1)  # repeat vertex 1: (v0, v1, v1)
        >>> torch.allclose(triangle(torch.tensor([[0.2, 0.3, 0.5]])), curve(torch.tensor([[0.2, 0.8]])))
        True
        """
        st, ds = self._simploid_type, self._degree
        dmap = _identity_map(st)
        if j == -1:
            if not 0 <= i <= len(st):
                raise IndexError(f"degeneracy ({i}, -1) is not defined for a simploid of type {st}.")
            if degree is None:
                if len(set(ds)) != 1:
                    raise ValueError("degree is required for a vertex or when the factors have different degrees.")
                degree = ds[0]
            new_type, new_degree = st[:i] + (1,) + st[i:], ds[:i] + (degree,) + ds[i:]
            new_map = dmap[:i] + ((-1, (0, 0)),) + dmap[i:]
        else:
            if not (0 <= i < len(st) and 0 <= j <= st[i]):
                raise IndexError(f"degeneracy ({i}, {j}) is not defined for a simploid of type {st}.")
            g = dmap[i][1]
            new_type, new_degree = st[:i] + (st[i] + 1,) + st[i + 1 :], ds
            new_map = dmap[:i] + ((i, tuple(g[x] if x <= j else g[x - 1] for x in range(len(g) + 1))),) + dmap[i + 1 :]
        parents = [_pullback_index(alpha, new_map, len(st)) for alpha in _indices(new_type, new_degree)]
        return self._derived(new_type, new_degree, parents)

    def product(self, other: "BezierSimploid") -> "BezierSimploid":
        r"""The cartesian product with another Bézier simploid (Minkowski sum of the control points).

        The product of Bézier simploids :math:`B_1` and :math:`B_2` is
        :math:`B(\mathbf u, \mathbf v) = B_1(\mathbf u) + B_2(\mathbf v)`.

        Examples
        --------
        >>> import torch
        >>> from torch_bsf.simploidal import BezierSimploid
        >>> b1, b2 = BezierSimploid.randn((2,), 2, n_values=3), BezierSimploid.randn((1,), 3, n_values=3)
        >>> u, v = torch.tensor([[0.2, 0.3, 0.5]]), torch.tensor([[0.4, 0.6]])
        >>> torch.allclose(b1.product(b2)(torch.cat([u, v], dim=1)), b1(u) + b2(v))
        True
        """
        if other.n_values != self.n_values:
            raise ValueError(f"n_values mismatch: {self.n_values} != {other.n_values}.")
        p1 = self.control_points.detach()
        p2 = other.control_points.detach().to(p1)
        matrix = (p1.unsqueeze(1) + p2.unsqueeze(0)).reshape(-1, self.n_values)
        return BezierSimploid(self._simploid_type + other._simploid_type, self._degree + other._degree, matrix)


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


class SimploidalBezierSpace(_LeastSquaresModule):
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
    >>> _ = torch.nn.init.normal_(space.proper_control_points)
    >>> u = torch.tensor([[0.0, 0.3, 0.7]])  # a point on the shared edge
    >>> torch.allclose(space.evaluate(t1, u), space.evaluate(t2, u))
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

    def _predict_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        ids, t, y = batch
        return self.forward(t, ids), y


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def _solve_least_squares(a: torch.Tensor, y: torch.Tensor, ridge: float) -> torch.Tensor:
    r"""Minimizes :math:`\|A X - Y\|^2 + \mathrm{ridge} \|X\|^2` in double precision (minimum-norm if singular)."""
    if ridge < 0:
        raise ValueError(f"ridge must be non-negative, but {ridge} is given.")
    a64, y64 = a.detach().cpu().double(), y.detach().cpu().double()
    if ridge > 0:
        lhs = a64.T @ a64 + ridge * torch.eye(a64.shape[1], dtype=torch.float64)
        return cast(torch.Tensor, torch.linalg.solve(lhs, a64.T @ y64))
    return cast(torch.Tensor, torch.linalg.lstsq(a64, y64, driver="gelsd").solution)


def _train(model: L.LightningModule, dataset: TensorDataset, batch_size: int | None, kwargs: dict[str, Any]) -> None:
    dl = DataLoader(dataset, batch_size=batch_size or len(dataset))
    L.Trainer(**kwargs).fit(model, dl)


def _floating_dtype(tensors: Sequence[torch.Tensor]) -> torch.dtype:
    """The promoted floating dtype of ``tensors``, or the default dtype if none is floating."""
    dtypes = [t.dtype for t in tensors if t.is_floating_point()]
    if not dtypes:
        return torch.get_default_dtype()
    dtype = dtypes[0]
    for d in dtypes[1:]:
        dtype = torch.promote_types(dtype, d)
    return dtype


def _check_values(params: torch.Tensor, values: torch.Tensor, n_params: int) -> None:
    if params.dim() != 2 or params.shape[1] != n_params:
        raise ValueError(f"params must have shape (n, {n_params}), but {tuple(params.shape)} is given.")
    if values.dim() != 2 or values.shape[0] != params.shape[0]:
        raise ValueError(f"values must have shape ({params.shape[0]}, n_values), but {tuple(values.shape)} is given.")


def fit_simploid(
    params: torch.Tensor,
    values: torch.Tensor,
    simploid_type: Sequence[int],
    degree: int | Sequence[int],
    *,
    solver: Solver = "lstsq",
    ridge: float = 0.0,
    batch_size: int | None = None,
    seed: int | None = None,
    **kwargs,
) -> BezierSimploid:
    r"""Fits a Bézier simploid to data.

    The control points minimize the squared error
    :math:`\sum_k \|B(\mathbf t_k) - \mathbf x_k\|^2`.

    Parameters
    ----------
    params
        Parameters of shape ``(n, sum(a_i + 1))``.
    values
        Values of shape ``(n, n_values)``.
    simploid_type
        The type :math:`(a_1, \ldots, a_n)`.
    degree
        The degree of every factor, or one degree per factor.
    solver
        ``"lstsq"`` solves the (linear) least-squares problem exactly;
        ``"trainer"`` trains with :class:`lightning.pytorch.Trainer` from random control points.
    ridge
        Weight of the Tikhonov penalty :math:`\mathrm{ridge}\,\|P\|^2` (``"lstsq"`` only).
        Without it, under-determined problems get the minimum-norm solution.
    batch_size
        Minibatch size for ``"trainer"`` (full batch by default).
    seed
        Random seed passed to :func:`lightning.pytorch.seed_everything` (``"trainer"`` only).
    kwargs
        Arguments for :class:`lightning.pytorch.Trainer` (``"trainer"`` only).

    Returns
    -------
    The fitted Bézier simploid.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf.simploidal import fit_simploid, sample_simploid
    >>> t = sample_simploid((2, 1), 100, seed=0)
    >>> x = t[:, :3] * t[:, 3:4]  # a function on a prism
    >>> bs = fit_simploid(t, x, (2, 1), degree=(1, 1))
    >>> torch.allclose(bs(t), x, atol=1e-5)
    True
    """
    st = _check_type(simploid_type)
    ds = _check_degree(degree, len(st))
    _check_values(params, values, simploid_n_params(st))
    if solver == "lstsq":
        basis = bernstein_basis(st, ds, params)
        solution = _solve_least_squares(basis, values, ridge)
        return BezierSimploid(st, ds, solution.to(values.dtype if values.is_floating_point() else basis.dtype))
    if solver != "trainer":
        raise ValueError(f"Unknown solver {solver!r}.")
    if ridge:
        raise ValueError("ridge is supported only by the 'lstsq' solver.")
    if seed is not None:
        L.seed_everything(seed)
    dtype = _floating_dtype([values])
    model = BezierSimploid.randn(st, ds, int(values.shape[1])).to(dtype)
    _train(model, TensorDataset(params.to(dtype), values.to(dtype)), batch_size, kwargs)
    return model


def _normalize_data(
    space: SimploidalBezierSpace, data: Sequence[tuple[Simploid, torch.Tensor, torch.Tensor]]
) -> list[tuple[Simploid, torch.Tensor, torch.Tensor]]:
    out = []
    for simploid, params, values in data:
        space.simploid_id(simploid)
        params, values = torch.as_tensor(params), torch.as_tensor(values)
        _check_values(params, values, simploid_n_params(simploid.simploid_type))
        if values.shape[1] != space.n_values:
            raise ValueError(f"All values must have {space.n_values} columns, but {values.shape[1]} is given.")
        out.append((simploid, params, values))
    return out


def fit_simploidal_set(
    simploidal_set: SimploidalSet,
    data: Sequence[tuple[Simploid, torch.Tensor, torch.Tensor]],
    degree: DegreeAssignment,
    *,
    default_degree: int | None = None,
    solver: Solver = "lstsq",
    ridge: float = 0.0,
    batch_size: int | None = None,
    seed: int | None = None,
    **kwargs,
) -> SimploidalBezierSpace:
    r"""Fits a simploidal Bézier space to data on its simploids.

    Each sample is attached to a non-degenerate simploid (of any dimension)
    and given in that simploid's parameters.  Since simploids sharing a face
    share its control points, all pieces are fitted jointly and the result is
    continuous across shared faces.  The proper control points minimize the
    total squared error over all samples.

    Parameters
    ----------
    simploidal_set
        The simploidal set.
    data
        A sequence of ``(simploid, params, values)``, where ``params`` has shape
        ``(n, sum(a_i + 1))`` for the type of ``simploid`` and ``values`` has shape ``(n, n_values)``.
        A simploid may appear several times.
    degree
        The degree of every factor of every simploid (at least 1), or a mapping from
        non-degenerate simploids to their degrees; see :class:`SimploidalBezierSpace`.
    default_degree
        The degree of the factors left undetermined by a ``degree`` mapping.
    solver
        ``"lstsq"`` solves the (linear) least-squares problem exactly;
        ``"trainer"`` trains with :class:`lightning.pytorch.Trainer` from zero control points.
    ridge
        Weight of the Tikhonov penalty :math:`\mathrm{ridge}\,\|P\|^2` on the proper control points
        (``"lstsq"`` only).  Without it, under-determined problems get the minimum-norm solution.
    batch_size
        Minibatch size for ``"trainer"`` (full batch by default).
    seed
        Random seed passed to :func:`lightning.pytorch.seed_everything` (``"trainer"`` only).
    kwargs
        Arguments for :class:`lightning.pytorch.Trainer` (``"trainer"`` only).

    Returns
    -------
    The fitted simploidal Bézier space.

    Examples
    --------
    Fit a piecewise model on a triangle glued to a square, with data that is
    continuous across the shared edge:

    >>> import torch
    >>> from torch_bsf.simploidal import SimploidalSet, fit_simploidal_set, sample_simploid
    >>> S = SimploidalSet()
    >>> tri, sq = S.add_simplex(2), S.add_simploid((1, 1))
    >>> _ = S.identify(tri.face(0, 2), sq.face(1, 1))
    >>> t1, t2 = sample_simploid((2,), 30, seed=0), sample_simploid((1, 1), 30, seed=1)
    >>> x1 = t1[:, 0:1] - t1[:, 1:2]  # the shared edge is (w0, w1, 0) on the triangle
    >>> x2 = (t2[:, 0:1] - t2[:, 1:2]) * t2[:, 2:3]  # and (w0, w1, 1, 0) on the square
    >>> space = fit_simploidal_set(S, [(tri, t1, x1), (sq, t2, x2)], degree=2)
    >>> torch.allclose(space.evaluate(tri, t1), x1, atol=1e-5)
    True

    The same with a quadratic triangle and a square that is cubic across the
    shared edge (the degree along the shared edge is common to both pieces):

    >>> space = fit_simploidal_set(S, [(tri, t1, x1), (sq, t2, x2)], degree={tri: 2, sq: (2, 3)})
    >>> space.simploid_degree(sq)
    (2, 3)
    """
    if not data:
        raise ValueError("data must not be empty.")
    n_values = int(torch.as_tensor(data[0][2]).shape[1])
    space = SimploidalBezierSpace(simploidal_set, degree, n_values, default_degree=default_degree)
    samples = _normalize_data(space, data)
    dtype = _floating_dtype([v for _, _, v in samples])
    space = space.to(dtype)
    if solver == "lstsq":
        blocks, targets = [], []
        for simploid, params, values in samples:
            basis = bernstein_basis(simploid.simploid_type, space.simploid_degree(simploid), params.cpu()).double()
            rows = torch.tensor(space.control_point_rows(simploid), dtype=torch.long)
            blocks.append(
                torch.zeros(len(params), space.n_proper_control_points, dtype=torch.float64).index_add_(1, rows, basis)
            )
            targets.append(values.cpu().double())
        solution = _solve_least_squares(torch.cat(blocks), torch.cat(targets), ridge)
        with torch.no_grad():
            space.proper_control_points.copy_(solution.to(space.proper_control_points.dtype))
        return space
    if solver != "trainer":
        raise ValueError(f"Unknown solver {solver!r}.")
    if ridge:
        raise ValueError("ridge is supported only by the 'lstsq' solver.")
    if seed is not None:
        L.seed_everything(seed)
    width = max(p.shape[1] for _, p, _ in samples)
    ids = torch.cat([torch.full((len(p),), space.simploid_id(s), dtype=torch.long) for s, p, _ in samples])
    params = torch.cat([F.pad(p.to(dtype), (0, width - p.shape[1])) for _, p, _ in samples])
    values = torch.cat([v.to(dtype) for _, _, v in samples])
    _train(space, TensorDataset(ids, params, values), batch_size, kwargs)
    return space
