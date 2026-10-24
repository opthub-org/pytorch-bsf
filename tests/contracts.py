"""Rejected inputs and explicit stopping states of the numerical public API."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from torch_bsf import BezierModel, BezierSpace, Ensemble, FixedControls, Points, QuadraticPenalty, Samples, SimploidalSet, fit
from torch_bsf.active_learning import farthest_points, query_by_committee
from torch_bsf.core import Cell
from torch_bsf.model_selection import select_degree
from torch_bsf.optimization import PreconditionedSGD, gram_matrix, linear_scalarization, minimize
from torch_bsf.results import ConvergenceWarning
from torch_bsf.sampling import grid, sample
from torch_bsf.solvers import DenseLeastSquares, SparseLeastSquares
from torch_bsf.splitting import EdgeSplit, LongestEdge, ResidualSplit, subdivide


@pytest.mark.parametrize(
    "config",
    [
        {"cells": ()},
        {"main_cells": ()},
        {"main_cells": (True,)},
        {"main_cells": (3,)},
        {"_edges": ((1, 0),)},
        {"cells": (Cell((1,), (2,), (0, 1)),)},
        {"cells": (Cell((1,), (1,), (0, 7)),)},
    ],
)
def test_inconsistent_space_layout_rejected(config):
    with pytest.raises(ValueError):
        replace(BezierSpace.simplex(1, degree=1), **config)


def test_topology_and_reference_errors():
    with pytest.raises(ValueError):
        BezierSpace.on_set(SimploidalSet(), degree=2)
    domain = SimploidalSet()
    edge = domain.add_simplex(1)
    space = BezierSpace.on_set(domain, degree=2)
    foreign = SimploidalSet().add_simplex(1)
    with pytest.raises(ValueError):
        space.cell_id(foreign)
    for cid in (-1, True, space.n_cells):
        with pytest.raises(ValueError):
            space.cell_id(cid)
    with pytest.raises(ValueError):
        space.control_index((7, 0), cell=edge)
    with pytest.raises(ValueError):
        space.with_degree((2,))
    with pytest.raises(ValueError):
        space.basis(torch.tensor([[0.4, 0.6]]))


@pytest.mark.parametrize(
    "coordinates,ids",
    [
        ([1.0, 0.0], None),
        ([[1.0, 0.0]], [0.5]),
        ([[1.0, 0.0]], [[0]]),
        ([[1.0, 0.0]], [True]),
        ([[1.0, 0.0]], [1j]),
    ],
)
def test_invalid_points_shape_and_ids(coordinates, ids):
    with pytest.raises(ValueError):
        Points(coordinates, ids)


@pytest.mark.parametrize(
    "coordinates,ids",
    [
        ([[1.0, 0.0]], [-1]),
        ([[1.0, 0.0]], [3]),
        ([[1.0, 0.0, 0.0]], [0]),
        ([[float("nan"), 0.0]], [0]),
        ([[-1.0, 2.0]], [0]),
        ([[0.2, 0.2]], [0]),
    ],
)
def test_points_must_belong_to_space(coordinates, ids):
    space = BezierSpace.simplex(1, degree=2)
    with pytest.raises(ValueError):
        space.basis(Points(coordinates, ids))


def test_sample_and_fixed_validation():
    p = Points([[1.0, 0.0]])
    with pytest.raises(ValueError):
        Samples(p, [[1.0]], cell_ids=[0])
    for values in ([], [[float("inf")]], [[1.0, 2.0], [3.0, 4.0]]):
        with pytest.raises(ValueError):
            Samples(p, values)
    for weights in ([float("inf")], [1e309], [-1.0]):
        with pytest.raises(ValueError):
            Samples(p, [[1.0]], weights=weights)
    for fixed in ({0: [[1.0]]}, {0: [float("nan")]}, {-1: [1.0]}):
        with pytest.raises(ValueError):
            FixedControls(fixed)
    for penalty in (dict(ridge=-1), dict(smoothness=float("inf"))):
        with pytest.raises(ValueError):
            QuadraticPenalty(**penalty)
    space = BezierSpace.simplex(1, degree=1)
    for fixed_values in ({3: [1.0]}, {0: [1.0, 2.0]}):
        with pytest.raises(ValueError):
            fit(space, Samples(torch.eye(2), [[1.0], [2.0]]), fixed=FixedControls(fixed_values))
    with pytest.raises(ValueError):
        Samples.from_cells(space, [])
    with pytest.raises(ValueError):
        Samples.from_cells(space, [(0, [[1.0, 0.0, 0.0]], [[1.0]])])
    assert len(Points.from_cells(space, [])) == 0
    assert Points([[1, 0]]).parameters.dtype == torch.float64
    with pytest.raises(ValueError):
        BezierModel(space, [[float("inf")], [0.0]])
    assert BezierModel(space, [[1], [2]]).coefficients.dtype == torch.float64


@pytest.mark.parametrize(
    "config",
    [
        {"max_iter": 0},
        {"max_iter": True},
        {"batch_size": 0},
        {"step_size": 0},
        {"preconditioner": "unknown"},
        {"step_decay": "unknown"},
        {"gradient_tolerance": -1},
    ],
)
def test_invalid_optimization_config(config):
    with pytest.raises(ValueError):
        PreconditionedSGD(**config)


@pytest.mark.parametrize("config", [{"rcond": -1}, {"diagnostics": "invalid"}])
def test_invalid_dense_config(config):
    with pytest.raises(ValueError):
        DenseLeastSquares(**config)


@pytest.mark.parametrize("config", [{"atol": -1}, {"btol": float("inf")}, {"conlim": -1}, {"max_iter": 0}])
def test_invalid_sparse_config(config):
    with pytest.raises(ValueError):
        SparseLeastSquares(**config)


def test_explicit_failure_guards():
    space = BezierSpace.simplex(1, degree=1)
    observations = Samples(torch.eye(2), [[1.0], [2.0]])
    with pytest.raises(TypeError):
        fit(space, torch.eye(2))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        fit(space, observations, solver="dense")  # type: ignore[arg-type]
    with patch("torch.linalg.lstsq") as solve:
        solve.return_value.solution = torch.full((2, 1), float("nan"), dtype=torch.float64)
        solve.return_value.singular_values = torch.ones(2, dtype=torch.float64)
        solve.return_value.rank = torch.tensor(2)
        with pytest.warns(ConvergenceWarning):
            result = fit(space, observations)
        assert not result.success and result.status == "numerical_failure"
        assert not result.diagnostics.solution_available


def test_minimization_stopping_and_validation():
    space = BezierSpace.simplex(1, degree=1)
    model = BezierModel(space, [[0.0], [0.0]])
    loss = lambda y, t: (y[:, 0] - 1).square()  # noqa: E731
    all_fixed = minimize(model, loss, fixed=FixedControls({0: [1.0], 1: [1.0]}), seed=7)
    assert all_fixed.success and all_fixed.require_success() is all_fixed and all_fixed.objective == 0
    stopped = minimize(model, loss, callback=lambda iteration, current: True, seed=7)
    assert stopped.status == "callback_stop" and not stopped.success
    with pytest.raises(ValueError):
        minimize(model, loss, n_samples=0)
    with pytest.raises(ValueError):
        minimize(model, lambda y, t: y)
    with pytest.raises(ValueError):
        minimize(model, loss, fixed=FixedControls({3: [0.0]}))
    with pytest.warns(ConvergenceWarning):
        result = minimize(
            model, loss, method=PreconditionedSGD(max_iter=2, batch_size=3, step_decay="sqrt", preconditioner="none"), seed=7
        )
    assert not result.success
    with pytest.raises(ArithmeticError):
        minimize(model, lambda y, t: y[:, 0].sqrt())
    domain = SimploidalSet()
    a, b = domain.add_simplex(1), domain.add_simplex(1)
    multi = BezierSpace.on_set(domain, degree=2)
    with pytest.raises(ValueError):
        gram_matrix(multi, cell_weights={a: 1, b: 0})


def test_scalarization_contract():
    space = BezierSpace.simplex(1, degree=1)
    points = grid(space, 2)
    loss = linear_scalarization(lambda y: torch.cat([y.square(), (y - 2).square()], 1))
    assert loss(torch.ones(3, 1), points).shape == (3,)
    with pytest.raises(ValueError):
        loss(torch.ones(3, 2), points)


@pytest.mark.parametrize("weights", [{}, {0: -1}, {0: float("inf")}])
def test_invalid_cell_distributions(weights):
    with pytest.raises(ValueError):
        sample(BezierSpace.simplex(1, degree=1), 5, cell_weights=weights)


def test_constant_sampling_and_grid_contracts():
    space = BezierSpace.simplex(0, degree=3)
    assert sample(space, 5, seed=7).parameters.shape == (5, 0)
    assert grid(space, 0).parameters.shape == (1, 0)
    assert len(grid(space, cells=[])) == 0
    assert len(sample(space, 0)) == 0
    with pytest.raises(ValueError):
        sample(space, 5, method="invalid")  # type: ignore[arg-type]
    domain = SimploidalSet()
    edge = domain.add_simplex(1)
    compiled = BezierSpace.on_set(domain, degree=2)
    chosen = sample(compiled, 3, cell_weights={edge: 1}, seed=0)
    assert chosen.cell_ids.unique().tolist() == [compiled.cell_id(edge)]
    assert grid(BezierSpace.simplex(1, degree=2), 0).parameters.tolist() == [[0.5, 0.5]]


def test_ensemble_and_query_preconditions():
    space = BezierSpace.simplex(1, degree=1)
    a = BezierModel(space, [[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError):
        Ensemble([])
    with pytest.raises(ValueError):
        Ensemble([a, BezierModel(BezierSpace.simplex(1, degree=2), torch.zeros(3, 2))])
    with pytest.raises(ValueError):
        Ensemble([a, a.double().copy().float()])
    candidates = grid(space, 3)
    with pytest.raises(ValueError):
        query_by_committee(Ensemble([a, a]), candidates, 9)
    with pytest.raises(ValueError):
        farthest_points(candidates, candidates, 9)
    with pytest.raises(ValueError):
        select_degree(space, Samples(candidates, torch.ones(4, 1)), degrees=[])


def test_subdivision_policy_and_weight_contracts():
    space = BezierSpace.simplex(1, degree=2)
    p = grid(space, 8)
    observations = Samples(p, p.parameters[:, :1] ** 3)
    model = fit(space, observations).model
    for policy in (LongestEdge(), ResidualSplit(observations, 3), EdgeSplit(0, 1, 0.4)):
        result = subdivide(model, policy=policy)
        assert len(result.models) == 2
    with pytest.raises(TypeError):
        subdivide(model, policy="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        subdivide(model, policy=ResidualSplit(Samples(p, observations.values, weights=torch.arange(9)), 3))
    with pytest.raises(ValueError):
        subdivide(model, policy=EdgeSplit(0, 1, 0))
    with pytest.raises(ValueError):
        from torch_bsf.plotting import plot_model

        plot_model(model, coordinates=(4, 5))


def test_multi_cell_coordinates_cannot_be_shorter_than_the_local_width():
    domain = SimploidalSet()
    triangle = domain.add_simplex(2)
    space = BezierSpace.on_set(domain, degree=2)
    with pytest.raises(ValueError, match="requires 3"):
        space.basis(Points([[0.4, 0.6]], [space.cell_id(triangle)]))
