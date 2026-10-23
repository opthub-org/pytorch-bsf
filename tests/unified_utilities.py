"""Sampling and exact local subdivision of the unified model."""

import pytest
import torch

from torch_bsf import BezierModel, BezierSpace
from torch_bsf.sampling import grid, sample
from torch_bsf.splitting import EdgeSplit, subdivide


@pytest.mark.parametrize("dimensions", [(1,), (2,), (1, 1), (2, 1)])
def test_subdivision_exact_maps(dimensions):
    space = BezierSpace.simploid(dimensions, degree=3)
    model = BezierModel(space, torch.randn(space.n_controls, 2, dtype=torch.float64))
    result = subdivide(model, policy=EdgeSplit(0, 1, 0.3, factor=len(dimensions) - 1))
    points = sample(result.models[0].space, 40, seed=6)
    for child, transform in zip(result.models, result.maps, strict=True):
        parent = transform.to_parent(points)
        assert torch.allclose(child(points), model(parent), atol=1e-12)
        restored, mask = transform.from_parent(parent)
        assert mask.all()
        assert torch.allclose(restored.parameters, points.parameters, atol=1e-12)


@pytest.mark.parametrize("method", ["random", "sobol"])
def test_sampling_local_seed(method):
    space = BezierSpace.simploid((2, 1), degree=2)
    state = torch.random.get_rng_state().clone()
    a, b = sample(space, 12, seed=9, method=method), sample(space, 12, seed=9, method=method)
    assert torch.equal(a.parameters, b.parameters)
    assert torch.equal(state, torch.random.get_rng_state())
    space.validate(a)
    space.validate(grid(space, 4))


def test_unified_plot_and_active_queries():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from torch_bsf import Ensemble
    from torch_bsf.active_learning import farthest_points, query_by_committee
    from torch_bsf.plotting import plot_model

    space = BezierSpace.simplex(1, degree=1)
    a, b = BezierModel(space, [[0.0, 0.0], [1.0, 1.0]]), BezierModel(space, [[0.0, 0.0], [2.0, 2.0]])
    candidates = grid(space, 10)
    committee = Ensemble([a, b])
    queried = query_by_committee(committee, candidates, 2)
    assert len(queried) == 2
    spread = farthest_points(candidates, queried, 3)
    assert len(spread) == 3
    ax = plot_model(a, show_controls=False)
    assert len(ax.lines) == 1
    plt.close(ax.figure)


def test_inverse_subdivision_on_padded_mixed_cell_batch():
    from torch_bsf import Points, SimploidalSet

    domain = SimploidalSet()
    triangle, square = domain.add_simplex(2), domain.add_simploid((1, 1))
    space = BezierSpace.on_set(domain, degree=2)
    model = BezierModel(space, torch.randn(space.n_controls, 2, dtype=torch.float64))
    pieces = subdivide(model, cell=space.cell_id(triangle), policy=EdgeSplit(0, 1, 0.4))
    points = Points.from_cells(space, [(square, [[0.2, 0.8, 0.5, 0.5]]), (triangle, [[0.7, 0.1, 0.2], [0.1, 0.7, 0.2]])])
    for child, mapping in zip(pieces.models, pieces.maps, strict=True):
        local, mask = mapping.from_parent(points)
        assert mask.shape == (3,) and not mask[0]
        assert torch.allclose(child(local), model(points.take(torch.nonzero(mask).flatten())), atol=1e-12)
