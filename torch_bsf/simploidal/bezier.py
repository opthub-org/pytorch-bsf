r"""Bézier simploids."""

from typing import Mapping, Sequence

import torch
import torch.nn as nn

from torch_bsf.simploidal.combinatorics import (
    SimploidIndex,
    SimploidType,
    _check_degree,
    _check_type,
    _factor_tensors,
    _indices,
    _positions,
    _pullback_index,
    _tensor_product_basis,
    simploid_indices,
    simploid_n_params,
)
from torch_bsf.simploidal.sets import _identity_map


class BezierSimploid(nn.Module):
    r"""A Bézier simploid.

    This is a plain :class:`torch.nn.Module` holding the control points; fit one to data with
    :func:`~torch_bsf.simploidal.fit_simploid`.

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
        # Cache the powers and coefficients of the Bernstein basis of every factor.  They follow the
        # module across devices and dtypes, and are rebuilt from the type and degree when loading.
        for k, (a, d) in enumerate(zip(st, ds)):
            powers, coeffs = _factor_tensors(a, d, matrix.dtype, matrix.device)
            self.register_buffer(f"_powers{k}", powers, persistent=False)
            self.register_buffer(f"_coeffs{k}", coeffs, persistent=False)

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
        cp = self.control_points
        return self.design_matrix(torch.as_tensor(t, dtype=cp.dtype, device=cp.device)) @ cp

    def design_matrix(self, t: torch.Tensor) -> torch.Tensor:
        r"""The Bernstein basis at ``t``, so that ``forward(t) == design_matrix(t) @ control_points``.

        Parameters
        ----------
        t
            Parameters of shape ``(batch, n_params)``.

        Returns
        -------
        The basis of shape ``(batch, n_indices)`` on the device of :attr:`control_points`, in the dtype
        of ``t`` (the default dtype if ``t`` is not floating).

        Examples
        --------
        >>> import torch
        >>> from torch_bsf.simploidal import BezierSimploid
        >>> bs = BezierSimploid.randn((2, 1), 2, n_values=3)
        >>> t = torch.tensor([[0.2, 0.3, 0.5, 0.4, 0.6]])
        >>> torch.allclose(bs.design_matrix(t) @ bs.control_points, bs(t))
        True
        """
        t = torch.as_tensor(t, device=self.control_points.device)
        if not t.is_floating_point():
            t = t.to(torch.get_default_dtype())
        if t.dim() != 2 or t.shape[1] != self.n_params:
            raise ValueError(f"t must have shape (batch, {self.n_params}), but {tuple(t.shape)} is given.")
        factors = [(getattr(self, f"_powers{k}"), getattr(self, f"_coeffs{k}")) for k in range(len(self._simploid_type))]
        return _tensor_product_basis(t, factors)

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
        >>> _ = torch.manual_seed(0)
        >>> prism = BezierSimploid.randn((2, 1), 3, n_values=2)
        >>> bottom = prism.face(1, 1)  # the triangle at u^2 = (1, 0)
        >>> u = torch.tensor([[0.2, 0.3, 0.5]])
        >>> torch.allclose(prism(torch.cat([u, torch.tensor([[1.0, 0.0]])], dim=1)), bottom(u), atol=1e-6)
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
        >>> _ = torch.manual_seed(0)
        >>> curve = BezierSimploid.randn((1,), 3, n_values=2)
        >>> triangle = curve.degenerate(0, 1)  # repeat vertex 1: (v0, v1, v1)
        >>> torch.allclose(triangle(torch.tensor([[0.2, 0.3, 0.5]])), curve(torch.tensor([[0.2, 0.8]])), atol=1e-6)
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
        >>> _ = torch.manual_seed(0)
        >>> b1, b2 = BezierSimploid.randn((2,), 2, n_values=3), BezierSimploid.randn((1,), 3, n_values=3)
        >>> u, v = torch.tensor([[0.2, 0.3, 0.5]]), torch.tensor([[0.4, 0.6]])
        >>> torch.allclose(b1.product(b2)(torch.cat([u, v], dim=1)), b1(u) + b2(v), atol=1e-6)
        True
        """
        if other.n_values != self.n_values:
            raise ValueError(f"n_values mismatch: {self.n_values} != {other.n_values}.")
        p1 = self.control_points.detach()
        p2 = other.control_points.detach().to(p1)
        matrix = (p1.unsqueeze(1) + p2.unsqueeze(0)).reshape(-1, self.n_values)
        return BezierSimploid(self._simploid_type + other._simploid_type, self._degree + other._degree, matrix)
