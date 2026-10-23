"""Spaces, observations and differentiable models shared by every fitting backend."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
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
        _integer(dimension, "dimension")
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
            tuple(map(tuple, compiled.adjacency().tolist())),
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
        if not isinstance(degree, int):
            raise ValueError("a multi-cell degree candidate must be a scalar degree.")
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
        for cid in torch.unique(points.cell_ids).tolist():
            info = self.cells[cid]
            if points.parameters.shape[1] < info.width:
                raise ValueError(f"cell {cid} requires {info.width} barycentric coordinates.")
            t = points.parameters[points.cell_ids == cid, : info.width]
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
        self.validate(batch)
        return BasisOperator(self, batch)


@dataclass(frozen=True, init=False)
class Points:
    """Barycentric coordinate rows and their cell IDs, in prediction order."""

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
        """Pack ordered cell-local coordinates without manual padding."""
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
    """Points with finite observed outputs and optional nonnegative weights."""

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
        """Pack ordered cell-local observations with different coordinate widths."""
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
    """Explicit fixed values keyed by globally shared coefficient row."""

    values: Mapping[int, torch.Tensor]

    def __init__(self, values: Mapping[int, Any]):
        checked = {}
        for index, value in values.items():
            _integer(index, "control index")
            tensor = torch.as_tensor(value)
            if tensor.is_complex():
                raise ValueError("fixed control values must be real.")
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


class BasisOperator:
    """Cell-local basis supporting products, transpose products and explicit export."""

    def __init__(self, space: BezierSpace, points: Points):
        self.space, self.points = space, points
        self.shape = (len(points), space.n_controls)
        self.blocks: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        t, ids = points.parameters, points.cell_ids
        for cid in torch.unique(ids).tolist():
            info = space.cells[cid]
            rows = torch.nonzero(ids == cid).flatten()
            columns = torch.tensor(info.controls, dtype=torch.long, device=t.device)
            basis = bernstein_basis(info.dimensions, info.degree, t[rows, : info.width])
            self.blocks.append((rows, columns, basis))

    def matmat(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Evaluate a coefficient matrix, accumulating only local contributions."""
        out = coefficients.new_zeros(self.shape[0], coefficients.shape[1])
        for rows, columns, basis in self.blocks:
            out = out.index_copy(0, rows, basis @ coefficients[columns])
        return out

    def rmatmat(self, values: torch.Tensor) -> torch.Tensor:
        """Apply the transpose, summing repeated shared coefficient columns."""
        out = values.new_zeros(self.shape[1], values.shape[1])
        for rows, columns, basis in self.blocks:
            out = out.index_add(0, columns, basis.T @ values[rows])
        return out

    def to_dense(self) -> torch.Tensor:
        """Explicitly materialize the full design matrix, for a dense solve."""
        out = self.points.parameters.new_zeros(self.shape)
        for rows, columns, basis in self.blocks:
            block = out.new_zeros(len(rows), self.shape[1]).index_add(1, columns, basis)
            out = out.index_copy(0, rows, block)
        return out


class BezierModel(nn.Module):
    """A vector-valued differentiable map on a compiled :class:`BezierSpace`.

    Parameters
    ----------
    space
        Geometry, degrees and coefficient ordering.
    coefficients
        Matrix of shape ``(space.n_controls, n_outputs)``. Copied on construction.

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
        """Evaluate points in their original row order, differentiable in both inputs and coefficients."""
        if not isinstance(points, Points) and self.space.n_cells != 1:
            raise ValueError("multi-cell models require explicit Points with cell IDs.")
        batch = points if isinstance(points, Points) else Points(points)
        batch = batch.to(device=self.coefficients.device, dtype=self.coefficients.dtype)
        return self.space.basis(batch).matmat(self.coefficients)

    def copy(self) -> BezierModel:
        """Copy coefficients without mutating or duplicating immutable geometry."""
        return BezierModel(self.space, self.coefficients)
