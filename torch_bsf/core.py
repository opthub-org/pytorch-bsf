"""Spaces, observations and differentiable models shared by every fitting backend."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from types import MappingProxyType
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from torch_bsf._kernels.combinatorics import (
    _adjacent_pairs,
    _canonical_index,
    _check_degree,
    _check_type,
    _indices,
    _structural_integer,
    bernstein_basis,
)
from torch_bsf._kernels.sets import Simploid, SimploidalSet
from torch_bsf._kernels.space import DegreeAssignment, SimploidalBezierSpace


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


@dataclass(frozen=True)
class Cell:
    """A compiled cell's factor dimensions, degrees and global coefficient rows."""

    dimensions: tuple[int, ...]
    degree: tuple[int, ...]
    controls: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimensions", tuple(self.dimensions))
        object.__setattr__(self, "degree", tuple(self.degree))
        object.__setattr__(self, "controls", tuple(self.controls))

    @property
    def width(self) -> int:
        """Number of concatenated barycentric coordinates."""
        return sum(a + 1 for a in self.dimensions)


@dataclass(frozen=True, eq=False)
class BezierSpace:
    """Immutable polynomial space, independent of learned values and output dimension.

    Construct a simplex with :meth:`simplex`, a product with :meth:`simploid`,
    or compile a mutable topology builder with :meth:`on_set`.
    Dimensions, degrees and control multi-indices accept Python and NumPy integers;
    booleans, strings and floats (including integer-valued floats) raise ``ValueError``.
    Factor dimensions are positive; :meth:`simplex` also accepts dimension zero.
    Single-cell degrees are non-negative; shared-space degrees are positive.

    Examples
    --------
    >>> from torch_bsf import BezierSpace
    >>> space = BezierSpace.simplex(dimension=2, degree=3)
    >>> space.n_controls
    10
    """

    cells: tuple[Cell, ...]
    n_controls: int
    main_cells: tuple[int, ...]
    _edges: tuple[tuple[int, int], ...] = field(repr=False)
    _source_ids: Mapping[Simploid, int] = field(default_factory=dict, repr=False, compare=False)
    _topology: tuple[Any, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cells", tuple(self.cells))
        object.__setattr__(self, "main_cells", tuple(self.main_cells))
        object.__setattr__(self, "_edges", tuple(tuple(edge) for edge in self._edges))
        _integer(self.n_controls, "n_controls", 1)
        if not self.cells or not self.main_cells:
            raise ValueError("a space requires cells and at least one maximal cell.")
        for cell in self.cells:
            _check_type(cell.dimensions)
            _check_degree(cell.degree, len(cell.dimensions))
            if len(cell.controls) != len(_indices(cell.dimensions, cell.degree)):
                raise ValueError("cell coefficient layout does not match its degree.")
            if any(isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < self.n_controls for k in cell.controls):
                raise ValueError("cell coefficient index is outside the space.")
        if any(isinstance(c, bool) or not isinstance(c, int) or not 0 <= c < len(self.cells) for c in self.main_cells):
            raise ValueError("invalid maximal cell ID.")
        if any(not (0 <= a < b < self.n_controls) for a, b in self._edges):
            raise ValueError("invalid coefficient adjacency.")
        object.__setattr__(self, "_source_ids", MappingProxyType(dict(self._source_ids)))

    @classmethod
    def simplex(cls, dimension: int, *, degree: int) -> BezierSpace:
        """Create a single simplex, including a zero-dimensional constant."""
        dimension = _structural_integer(dimension, "dimension")
        return cls.simploid((dimension,) if dimension else (), degree=degree)

    @classmethod
    def simploid(cls, dimensions: Sequence[int], *, degree: int | Sequence[int]) -> BezierSpace:
        """Create a tensor product of simplices with one degree per factor."""
        st = _check_type(dimensions)
        ds = _check_degree(degree, len(st))
        n = len(_indices(st, ds))
        cell = Cell(st, ds, tuple(range(n)))
        return cls((cell,), n, (0,), _adjacent_pairs(st, ds))

    @classmethod
    def on_set(cls, domain: SimploidalSet, *, degree: DegreeAssignment, default_degree: int | None = None) -> BezierSpace:
        """Snapshot a simploidal set, propagating compatible degrees across shared faces.

        The builder's cell handles remain usable for addressing this snapshot;
        subsequent changes to the builder cannot change its basis or layout.
        """
        domain.check()
        if not len(domain):
            raise ValueError("domain must contain at least one cell.")
        original = domain.simploids()
        frozen, degrees = copy.deepcopy((domain, degree))
        # Compile the existing, tested incidence/degree algorithm, retaining only
        # immutable geometry. Its temporary parameter tensor is not kept by the space.
        compiled = SimploidalBezierSpace(frozen, degrees, n_values=1, default_degree=default_degree)
        ids = {s: k for k, s in enumerate(compiled.simploids)}
        cells = tuple(
            Cell(s.simploid_type, compiled.simploid_degree(s), tuple(compiled.control_point_rows(s)))
            for s in compiled.simploids
        )
        faces = {ids[f.base] for s in compiled.simploids for _, f in s.faces()}
        main = tuple(k for k in range(len(cells)) if k not in faces)
        topology = tuple(tuple((i, j, ids[f.base], f.degeneracy_map) for (i, j), f in s.faces()) for s in compiled.simploids)
        return cls(
            cells,
            compiled.n_proper_control_points,
            main,
            tuple((int(row[0]), int(row[1])) for row in compiled.adjacency().unbind()),
            MappingProxyType({s: k for k, s in enumerate(original)}),
            topology,
        )

    @property
    def n_cells(self) -> int:
        """Number of addressable non-degenerate cells, including faces."""
        return len(self.cells)

    @property
    def width(self) -> int:
        """Width used when packing mixed-cell coordinate batches."""
        return max(c.width for c in self.cells)

    def cell_id(self, cell: int | Simploid) -> int:
        """Resolve an integer or an original builder handle to its frozen cell ID."""
        if isinstance(cell, Simploid):
            try:
                return self._source_ids[cell]
            except KeyError:
                raise ValueError("cell does not belong to this space snapshot.") from None
        if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell < self.n_cells:
            raise ValueError("cell ID is outside this space.")
        return cell

    def control_index(self, index: Sequence[Any], *, cell: int | Simploid = 0) -> int:
        """Resolve a cell-local multi-index to a globally shared coefficient row."""
        info = self.cells[self.cell_id(cell)]
        alpha = _canonical_index(info.dimensions, index)
        try:
            return info.controls[_indices(info.dimensions, info.degree).index(alpha)]
        except ValueError:
            raise ValueError("invalid control index for this cell.") from None

    def adjacency(self) -> torch.Tensor:
        """Return a copy of the undirected control-net edges."""
        return torch.tensor(self._edges, dtype=torch.long).reshape(-1, 2)

    def compatible(self, other: BezierSpace) -> bool:
        """Whether coefficient ordering and all addressable cells agree."""
        return (self.cells, self.n_controls, self.main_cells, self._topology) == (
            other.cells,
            other.n_controls,
            other.main_cells,
            other._topology,
        )

    def with_degree(self, degree: int | Sequence[int]) -> BezierSpace:
        """Recompile the same frozen domain at another degree for model selection."""
        if not self._topology:
            return BezierSpace.simploid(self.cells[0].dimensions, degree=degree)
        if not isinstance(degree, Integral):
            raise ValueError("a multi-cell degree candidate must be a scalar degree.")
        degree = _structural_integer(degree, "degree", 1)
        from torch_bsf._kernels.sets import _canonical

        nodes = [Simploid(cell.dimensions) for cell in self.cells]
        for node, faces in zip(nodes, self._topology, strict=True):
            for i, j, base, mapping in faces:
                node._faces[(i, j)] = _canonical(nodes[base], mapping)
        domain = SimploidalSet()
        domain._simploids = nodes
        return BezierSpace.on_set(domain, degree=degree)

    def validate(self, points: Points) -> None:
        """Check IDs, finite coordinates and simplex membership of each factor."""
        if self.n_cells == 1 and points.parameters.shape[1] != self.width:
            raise ValueError("parameter width does not match the single-cell space.")
        if (points.cell_ids < 0).any() or (points.cell_ids >= self.n_cells).any():
            raise ValueError("points contain an invalid cell ID.")
        self._validate_groups(points, _point_groups(points.cell_ids))

    def _validate_groups(self, points: Points, groups: list[tuple[int, torch.Tensor]]) -> None:
        """Validate each active cell after a single grouping of observation rows."""
        for cid, rows in groups:
            info = self.cells[cid]
            if points.parameters.shape[1] < info.width:
                raise ValueError(f"cell {cid} requires {info.width} barycentric coordinates.")
            t = points.parameters[rows, : info.width]
            if not torch.isfinite(t).all():
                raise ValueError("parameters must be finite.")
            offset = 0
            for a in info.dimensions:
                block = t[:, offset : offset + a + 1]
                if (block < -1e-6).any() or not torch.allclose(
                    block.sum(1), torch.ones(len(t), device=t.device, dtype=t.dtype), atol=1e-6, rtol=0
                ):
                    raise ValueError("parameters must be barycentric coordinates on each simplex factor.")
                offset += a + 1

    def basis(self, points: Points | torch.Tensor) -> BasisOperator:
        """Build a cell-local linear operator without a global dense allocation."""
        if not isinstance(points, Points) and self.n_cells != 1:
            raise ValueError("multi-cell spaces require explicit Points with cell IDs.")
        batch = points if isinstance(points, Points) else Points(points)
        groups = _point_groups(batch.cell_ids)
        # Validate IDs and dimensions before indexing the cell table.
        if self.n_cells == 1 and batch.parameters.shape[1] != self.width:
            raise ValueError("parameter width does not match the single-cell space.")
        if (batch.cell_ids < 0).any() or (batch.cell_ids >= self.n_cells).any():
            raise ValueError("points contain an invalid cell ID.")
        self._validate_groups(batch, groups)
        return BasisOperator(self, batch, _groups=groups)


@dataclass(frozen=True, init=False)
class Points:
    """Barycentric coordinate rows and their cell IDs, in prediction order.

    Parameters
    ----------
    parameters : array-like, shape (n_samples, coordinate_width)
        Real coordinate rows. For a simplex of dimension ``d``, the width
        is ``d + 1``; for a product, concatenate one barycentric block per
        factor. Mixed-cell rows may be padded on the right to ``space.width``.
        Coordinates are copied on their input device.
    cell_ids : array-like of int, shape (n_samples,), optional
        One integer cell ID per row. Defaults to all zeros on the coordinate
        device. IDs are copied as ``torch.long`` on that device.

    Raises
    ------
    ValueError
        If coordinates are complex or not two-dimensional, or IDs are not
        a matching one-dimensional integer vector (booleans are rejected).

    Notes
    -----
    Floating tensors and NumPy arrays retain their dtype. Python float lists
    use ``torch.get_default_dtype()``; integer coordinates become float64.
    Pass float64 tensors or arrays to preserve double-precision coordinates.
    Construction checks shapes and types only. Cell membership, finite active
    coordinates and barycentric blocks are checked by ``space.validate``,
    basis construction or model evaluation. Inactive padding is ignored.

    Examples
    --------
    >>> from torch_bsf import BezierSpace, Points
    >>> points = Points([[1.0, 0.0], [0.25, 0.75]])
    >>> tuple(points.parameters.shape), points.cell_ids.tolist()
    ((2, 2), [0, 0])
    >>> BezierSpace.simplex(1, degree=1).validate(points)
    """

    parameters: torch.Tensor
    cell_ids: torch.Tensor

    def __init__(self, parameters: Any, cell_ids: Any = None):
        t = torch.as_tensor(parameters)
        if t.is_complex():
            raise ValueError("parameters must be real.")
        if t.ndim != 2:
            raise ValueError("parameters must have shape (n_samples, n_parameters).")
        if not t.is_floating_point():
            t = t.to(torch.float64)
        ids = (
            torch.zeros(len(t), dtype=torch.long, device=t.device)
            if cell_ids is None
            else torch.as_tensor(cell_ids, device=t.device)
        )
        if ids.ndim != 1 or len(ids) != len(t) or ids.is_floating_point() or ids.dtype == torch.bool or ids.is_complex():
            raise ValueError("cell_ids must be an integer vector matching parameters.")
        object.__setattr__(self, "parameters", t.clone())
        object.__setattr__(self, "cell_ids", ids.to(torch.long).clone())

    @classmethod
    def from_cells(cls, space: BezierSpace, data: Sequence[tuple[int | Simploid, Any]]) -> Points:
        """Pack ordered cell-local coordinates without manual padding.

        Parameters
        ----------
        space : BezierSpace
            Compiled space used to resolve cell references and validate rows.
        data : sequence of (cell, parameters) pairs
            Each ``cell`` is an integer ID or an original builder handle.
            Its coordinates have shape ``(n_cell_samples, cell.width)``.
            Entries and rows within each entry retain their supplied order;
            repeated cell references are allowed.

        Returns
        -------
        Points
            Coordinates of shape ``(n_samples, space.width)``, where
            ``n_samples`` is the sum of entry row counts. Nonempty input is
            converted to CPU float64 and zero-padded on the right. IDs are
            CPU ``torch.long``. An empty sequence returns zero rows with
            coordinates in PyTorch's default floating dtype on its default
            device.

        Raises
        ------
        ValueError
            If a cell reference is invalid, a coordinate batch has the wrong
            shape or local width, or its coordinates are not finite valid
            barycentric blocks for that cell.

        Notes
        -----
        Inputs first follow :class:`Points` dtype inference, then convert to
        float64; this cannot recover precision already lost to float32.
        See :meth:`Samples.from_cells` for a mixed-width packing example with
        aligned observations and weights.

        Examples
        --------
        >>> from torch_bsf import BezierSpace, Points
        >>> space = BezierSpace.simplex(1, degree=1)
        >>> points = Points.from_cells(space, [(0, [[0.25, 0.75], [1.0, 0.0]])])
        >>> points.parameters.tolist(), points.cell_ids.tolist()
        ([[0.25, 0.75], [1.0, 0.0]], [0, 0])
        >>> str(points.parameters.dtype), str(points.parameters.device)
        ('torch.float64', 'cpu')
        """
        if not data:
            return cls(torch.empty(0, space.width), torch.empty(0, dtype=torch.long))
        coordinates, identifiers = [], []
        for cell, parameters in data:
            cid = space.cell_id(cell)
            batch = cls(parameters, torch.full((len(parameters),), cid, dtype=torch.long))
            if batch.parameters.shape[1] != space.cells[cid].width:
                raise ValueError("cell-local coordinates have the wrong width.")
            space.validate(batch)
            coordinates.append(F.pad(batch.parameters.cpu().double(), (0, space.width - batch.parameters.shape[1])))
            identifiers.append(batch.cell_ids.cpu())
        return cls(torch.cat(coordinates), torch.cat(identifiers))

    def __len__(self) -> int:
        return len(self.parameters)

    def take(self, indices: torch.Tensor) -> Points:
        """Select rows while preserving their cell identities."""
        rows = indices.to(self.parameters.device)
        return Points(self.parameters[rows], self.cell_ids[rows])

    def to(self, *, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float64) -> Points:
        """Copy coordinates to an explicit compute dtype and device."""
        return Points(self.parameters.to(device=device, dtype=dtype), self.cell_ids.to(device))


@dataclass(frozen=True, init=False)
class Samples:
    """Points with finite observed outputs and optional nonnegative weights.

    Parameters
    ----------
    parameters : Points or array-like, shape (n_samples, coordinate_width)
        Existing points, retained as supplied, or coordinates used to build
        :class:`Points`. Raw coordinates and cell IDs are copied by Points.
    values : array-like, shape (n_samples, n_outputs) or (n_samples,)
        Real, finite observations in point row order. One-dimensional values
        become one output column. Both ``n_samples`` and ``n_outputs`` must
        be positive. Values are copied and retain their input device.
    cell_ids : array-like of int, shape (n_samples,), optional
        IDs for raw coordinates; defaults to all zeros. Must be omitted when
        ``parameters`` is already a :class:`Points` instance.
    weights : array-like, shape (n_samples,), optional
        Observation weights in the same row order, copied to CPU float64.
        Defaults to all ones. Entries must be finite and nonnegative, and
        their sum must be finite and strictly positive. Individual zeros
        are allowed, but all observations must still be finite, including
        observations with zero weight.

    Raises
    ------
    ValueError
        If both Points and separate IDs are supplied, values are complex,
        nonfinite, empty or have a mismatched shape, or weights violate the
        shape, finiteness or positivity requirements. Raw coordinates also
        follow the construction checks of :class:`Points`.

    Notes
    -----
    Coordinates follow :class:`Points`. Observed values use
    ``torch.as_tensor`` dtype inference: tensors and NumPy arrays retain
    their dtype, Python float lists use ``torch.get_default_dtype()``, and
    integer values remain integers. For double precision, pass float64
    tensors or arrays. Weights are stored as CPU float64 tensors.
    As with direct Points construction, space membership is checked when
    the points are validated or used with a space, rather than here.

    Examples
    --------
    >>> from torch_bsf import Points, Samples
    >>> points = Points([[1.0, 0.0], [0.0, 1.0]])
    >>> samples = Samples(points, [2.0, 4.0], weights=[0.0, 1.0])
    >>> tuple(samples.values.shape), samples.weights.tolist()
    ((2, 1), [0.0, 1.0])
    >>> Samples(points, [2.0, 4.0], cell_ids=[0, 0])
    Traceback (most recent call last):
        ...
    ValueError: cell_ids are already specified by Points.
    >>> Samples(points, [float("nan"), 4.0], weights=[0.0, 1.0])
    Traceback (most recent call last):
        ...
    ValueError: values must be finite with shape (n_samples, n_outputs), both positive.
    """

    points: Points
    values: torch.Tensor
    weights: torch.Tensor

    def __init__(self, parameters: Any, values: Any, *, cell_ids: Any = None, weights: Any = None):
        if isinstance(parameters, Points):
            if cell_ids is not None:
                raise ValueError("cell_ids are already specified by Points.")
            points = parameters
        else:
            points = Points(parameters, cell_ids)
        y = torch.as_tensor(values)
        if y.is_complex():
            raise ValueError("values must be real.")
        if y.ndim == 1:
            y = y[:, None]
        if y.ndim != 2 or len(y) != len(points) or y.shape[1] < 1 or not len(y) or not torch.isfinite(y).all():
            raise ValueError("values must be finite with shape (n_samples, n_outputs), both positive.")
        w = torch.ones(len(y), dtype=torch.float64) if weights is None else torch.as_tensor(weights, dtype=torch.float64).cpu()
        if (
            w.shape != (len(y),)
            or not torch.isfinite(w).all()
            or (w < 0).any()
            or not torch.isfinite(w.sum())
            or not w.sum() > 0
        ):
            raise ValueError("weights must be finite, nonnegative and have a positive sum.")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "values", y.clone())
        object.__setattr__(self, "weights", w.clone())

    def __len__(self) -> int:
        return len(self.points)

    def take(self, indices: torch.Tensor) -> Samples:
        """Select observations, keeping coordinates, outputs and weights aligned."""
        return Samples(
            self.points.take(indices), self.values[indices.to(self.values.device)], weights=self.weights[indices.cpu()]
        )

    @classmethod
    def from_cells(
        cls, space: BezierSpace, data: Sequence[tuple[int | Simploid, Any, Any]], *, weights: Any = None
    ) -> Samples:
        """Pack ordered cell-local observations with different coordinate widths.

        Parameters
        ----------
        space : BezierSpace
            Compiled space used to resolve cell references and validate rows.
        data : sequence of (cell, parameters, values) triples
            Nonempty sequence of cell IDs or original builder handles,
            coordinates of shape ``(n_cell_samples, cell.width)``, and
            corresponding finite observations of shape
            ``(n_cell_samples, n_outputs)`` or ``(n_cell_samples,)``.
            Each entry must contain at least one row and all entries must
            have the same ``n_outputs``.
        weights : array-like, shape (n_samples,), optional
            One flat vector for all concatenated rows, in entry order then
            row order within each entry. Defaults to ones; follows the
            finite, nonnegative, positive-sum contract of :class:`Samples`.

        Returns
        -------
        Samples
            Coordinates of shape ``(n_samples, space.width)`` and observations
            of shape ``(n_samples, n_outputs)``, both on CPU in float64.
            Narrower coordinates have zero padding on the right. Cell IDs,
            values and weights follow the same concatenation order.

        Raises
        ------
        ValueError
            If data is empty, cell references or local coordinates are
            invalid, an entry has invalid observations, or weights violate
            the :class:`Samples` contract.
        RuntimeError
            If entries have inconsistent output widths, so their observed
            values cannot be concatenated.

        Notes
        -----
        Unlike direct construction, this helper converts coordinates and
        observations to CPU float64 after their usual dtype inference.
        Pass float64 inputs to avoid first rounding Python float lists to
        the default dtype. Outputs and weights are copied.

        Examples
        --------
        Pack a triangle row before two edge rows. The edge needs two active
        coordinates; its third column is padding. Weights follow this order,
        regardless of the numeric cell IDs.

        >>> from torch_bsf import BezierSpace, Samples, SimploidalSet
        >>> domain = SimploidalSet()
        >>> edge, triangle = domain.add_simplex(1), domain.add_simplex(2)
        >>> space = BezierSpace.on_set(domain, degree=1)
        >>> samples = Samples.from_cells(space, [
        ...     (triangle, [[0.25, 0.25, 0.5]], [10.0]),
        ...     (edge, [[1.0, 0.0], [0.0, 1.0]], [20.0, 30.0]),
        ... ], weights=[2.0, 0.0, 3.0])
        >>> samples.points.parameters.tolist()
        [[0.25, 0.25, 0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        >>> samples.points.cell_ids.tolist() == [space.cell_id(triangle), space.cell_id(edge), space.cell_id(edge)]
        True
        >>> tuple(samples.values.shape), samples.values[:, 0].tolist(), samples.weights.tolist()
        ((3, 1), [10.0, 20.0, 30.0], [2.0, 0.0, 3.0])
        >>> str(samples.points.parameters.dtype), str(samples.values.dtype), str(samples.values.device)
        ('torch.float64', 'torch.float64', 'cpu')
        """
        if not data:
            raise ValueError("data must contain observations.")
        points = Points.from_cells(space, [(cell, t) for cell, t, _ in data])
        outputs = []
        for _, t, y in data:
            outputs.append(cls(t, y).values.cpu().double())
        return cls(points, torch.cat(outputs), weights=weights)


@dataclass(frozen=True)
class QuadraticPenalty:
    """Weights of squared coefficients and adjacent coefficient differences."""

    ridge: float = 0.0
    smoothness: float = 0.0

    def __post_init__(self) -> None:
        import math

        if any(not math.isfinite(v) or v < 0 for v in (self.ridge, self.smoothness)):
            raise ValueError("penalty weights must be finite and nonnegative.")


@dataclass(frozen=True, init=False)
class FixedControls:
    """Explicit fixed values keyed by globally shared coefficient row.

    Values are copied to CPU float64 tensors. Non-tensor floating inputs
    are read directly as float64, independently of PyTorch's default dtype;
    tensor inputs retain their existing precision before conversion.
    Complex inputs are rejected.

    Examples
    --------
    >>> from torch_bsf import FixedControls
    >>> FixedControls({0: [0.2]}).values[0].item()
    0.2
    """

    values: Mapping[int, torch.Tensor]

    def __init__(self, values: Mapping[int, Any]):
        checked = {}
        for index, value in values.items():
            _integer(index, "control index")
            tensor = torch.as_tensor(value)
            if tensor.is_complex():
                raise ValueError("fixed control values must be real.")
            if not isinstance(value, torch.Tensor) and tensor.is_floating_point():
                tensor = torch.as_tensor(value, dtype=torch.float64)
            tensor = tensor.to(dtype=torch.float64, device="cpu")
            if tensor.ndim != 1 or not torch.isfinite(tensor).all():
                raise ValueError("fixed control values must be finite output vectors.")
            checked[index] = tensor.clone()
        object.__setattr__(self, "values", MappingProxyType(checked))

    def __deepcopy__(self, memo: dict[int, Any]) -> FixedControls:
        """Copy constraint tensors for independent estimator cloning."""
        copied = FixedControls({row: value.detach() for row, value in self.values.items()})
        memo[id(self)] = copied
        return copied

    @classmethod
    def from_cells(cls, space: BezierSpace, values: Sequence[tuple[int | Simploid, Sequence[Any], Any]]) -> FixedControls:
        """Resolve local references and reject contradictory shared values."""
        checked: dict[int, torch.Tensor] = {}
        for cell, index, value in values:
            row = space.control_index(index, cell=cell)
            normalized = cls({row: value}).values[row]
            if row in checked and not torch.equal(checked[row], normalized):
                raise ValueError("conflicting fixed values for the same shared control.")
            checked[row] = normalized
        return cls(checked)


def _point_groups(ids: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
    """Group row indices once, keeping observation order within each active cell."""
    if not len(ids):
        return []
    order = torch.argsort(ids, stable=True)
    cells, counts = torch.unique_consecutive(ids[order], return_counts=True)
    return list(zip(cells.tolist(), torch.split(order, counts.tolist()), strict=True))


class BasisOperator:
    """Cell-local basis supporting products, transpose products and explicit export."""

    def __init__(self, space: BezierSpace, points: Points, *, _groups: list[tuple[int, torch.Tensor]] | None = None):
        self.space, self.points = space, points
        self.shape = (len(points), space.n_controls)
        self.blocks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        t = points.parameters
        for cid, rows in _point_groups(points.cell_ids) if _groups is None else _groups:
            info = space.cells[cid]
            columns = torch.tensor(info.controls, dtype=torch.long, device=t.device)
            basis = bernstein_basis(info.dimensions, info.degree, t[rows, : info.width])
            self.blocks.append((rows, columns, basis))

    def matmat(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Evaluate a coefficient matrix, accumulating only local contributions."""
        out = coefficients.new_zeros(self.shape[0], coefficients.shape[1])
        for rows, columns, basis in self.blocks:
            out.index_copy_(0, rows, basis @ coefficients[columns])
        return out

    def rmatmat(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the transpose, summing repeated shared coefficient columns."""
        out = values.new_zeros(self.shape[1], values.shape[1])
        for rows, columns, basis in self.blocks:
            out.index_add_(0, columns, basis.T @ values[rows])
        return out

    def to_dense(self) -> torch.Tensor:
        """Explicitly materialize the full design matrix, for a dense solve."""
        out = self.points.parameters.new_zeros(self.shape)
        for rows, columns, basis in self.blocks:
            out.index_put_((rows[:, None], columns[None, :]), basis, accumulate=True)
        return out


class BezierModel(nn.Module):
    """A vector-valued differentiable map on a compiled :class:`BezierSpace`.

    Parameters
    ----------
    space
        Geometry, degrees and coefficient ordering.
    coefficients
        Matrix of shape ``(space.n_controls, n_outputs)``. Copied on construction.
        Floating tensors and NumPy arrays retain their dtype. Python float
        lists use ``torch.get_default_dtype()``; integer inputs become
        float64. Pass float64 tensors or arrays for double precision.

    Examples
    --------
    >>> import torch
    >>> from torch_bsf import BezierSpace, BezierModel
    >>> model = BezierModel(BezierSpace.simplex(1, degree=1), torch.tensor([[1.], [3.]]))
    >>> model(torch.tensor([[0.5, 0.5]])).item()
    2.0
    """

    def __init__(self, space: BezierSpace, coefficients: Any):
        super().__init__()
        p = torch.as_tensor(coefficients)
        if p.ndim != 2 or p.shape[0] != space.n_controls or p.shape[1] < 1 or not torch.isfinite(p).all():
            raise ValueError("coefficients must be finite with shape (space.n_controls, n_outputs).")
        if p.is_complex():
            raise ValueError("coefficients must be real.")
        if not p.is_floating_point():
            p = p.to(torch.float64)
        self.space = space
        self.coefficients = nn.Parameter(p.detach().clone())

    @property
    def n_outputs(self) -> int:
        """Dimension of each predicted value."""
        return self.coefficients.shape[1]

    def forward(self, points: Points | torch.Tensor) -> torch.Tensor:
        """Evaluate points in their original row order.

        Parameters
        ----------
        points : Points or torch.Tensor
            Barycentric coordinates. A raw tensor of shape
            ``(n_samples, space.width)`` is accepted only when the space has
            one addressable cell; all rows then belong to cell zero.
            Multi-cell spaces require :class:`Points` with explicit IDs,
            including when every requested row belongs to the same cell.
            Active coordinates must be finite barycentric blocks; extra
            padding in mixed-cell Points is ignored.

        Returns
        -------
        torch.Tensor, shape (n_samples, n_outputs)
            Predictions in input row order, on the coefficient dtype and
            device. Input coordinates are copied to that dtype and device.
            Evaluation is differentiable in coordinates and coefficients.

        Raises
        ------
        ValueError
            If raw coordinates are used with a multi-cell space, Points
            construction fails, IDs are outside the space, coordinate width
            is invalid, or active coordinates fail barycentric validation.

        Examples
        --------
        >>> import torch
        >>> from torch_bsf import BezierModel, BezierSpace, Points, SimploidalSet
        >>> model = BezierModel(BezierSpace.simplex(1, degree=1), [[1.0], [3.0]])
        >>> tuple(model(torch.tensor([[0.5, 0.5], [0.25, 0.75]])).shape)
        (2, 1)
        >>> domain = SimploidalSet()
        >>> edge = domain.add_simplex(1)
        >>> space = BezierSpace.on_set(domain, degree=1)
        >>> model = BezierModel(space, torch.ones(space.n_controls, 2))
        >>> points = Points.from_cells(space, [(edge, [[0.5, 0.5]])])
        >>> model(points).tolist()
        [[1.0, 1.0]]
        >>> model(torch.tensor([[0.5, 0.5]]))
        Traceback (most recent call last):
            ...
        ValueError: multi-cell models require explicit Points with cell IDs.
        """
        if not isinstance(points, Points) and self.space.n_cells != 1:
            raise ValueError("multi-cell models require explicit Points with cell IDs.")
        batch = points if isinstance(points, Points) else Points(points)
        batch = batch.to(device=self.coefficients.device, dtype=self.coefficients.dtype)
        return self.space.basis(batch).matmat(self.coefficients)

    def copy(self) -> BezierModel:
        """Copy coefficients without mutating or duplicating immutable geometry."""
        return BezierModel(self.space, self.coefficients)
