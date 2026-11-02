"""Exact BSF export of finite-scale curve topology candidates."""

import math

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from torch_bsf._adaptive_curve.export import _elevate_controls, export_curve
from torch_bsf._adaptive_curve.polynomial import CurveGap, CurveLearningResult, _Spline
from torch_bsf.core import Points


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _spline(knots, degree, controls, center=None):
    values = torch.tensor(controls, dtype=torch.float64)
    offset = torch.zeros(values.shape[1], dtype=torch.float64) if center is None else torch.tensor(center, dtype=torch.float64)
    return _Spline(torch.tensor(knots, dtype=torch.float64), degree, values, offset)


def _result(charts, bounds=(-1.0, 1.0)):
    inputs = torch.cat([chart.knots for chart in charts])[:, None]
    values = torch.cat([chart.predict(chart.knots) for chart in charts])
    gaps = tuple(
        CurveGap(float(a.knots[-1]), float(b.knots[0]), 1.0, 0.1, 0.0) for a, b in zip(charts[:-1], charts[1:], strict=True)
    )
    return CurveLearningResult(inputs, values, gaps, "budget_exhausted", {}, charts, bounds)


def _mixed():
    return _result(
        [
            _spline([-0.9, -0.55, -0.2], 1, [[0, 0], [1, 2], [2, 0]], [0.25, -0.5]),
            _spline([0.1, 0.6], 3, [[2.25, -0.5], [4, 1], [-1, 2], [3, 0]]),
            _spline([0.85], 0, [[0, 0]], [3, 4]),
        ]
    )


def test_mixed_degrees_components_vertices_and_gap_predictions_match_without_refitting():
    original = _mixed()
    exported = export_curve(original)
    x = torch.cat((torch.linspace(-1, 1, 301, dtype=torch.float64), original.inputs[:, 0]))[:, None]
    x = x[torch.randperm(len(x), generator=torch.Generator().manual_seed(11))]
    torch.testing.assert_close(exported.predict(x), original.predict(x), rtol=1e-12, atol=1e-12, equal_nan=True)
    assert exported.degree == 3
    assert exported.domain.betti_numbers() == [3, 0]
    assert exported.model.space.n_controls == 3 * 3 + 3
    assert len(exported.intervals) == 4
    assert torch.isnan(exported.predict([[-0.1], [0.7]])).all()
    assert exported.predict(torch.empty(0, 1)).shape == (0, 2)
    assert "output-image topology is not certified" in exported.topology_scope


def test_known_piece_incidence_is_glued_but_equal_output_endpoints_are_not():
    original = _mixed()
    exported = export_curve(original)
    first, second, separate, vertex = exported.intervals
    space, degree = exported.model.space, exported.degree
    shared_end = space.control_index((0, degree), cell=first.handle)
    shared_start = space.control_index((degree, 0), cell=second.handle)
    assert shared_end == shared_start
    end = space.control_index((0, degree), cell=second.handle)
    other_start = space.control_index((degree, 0), cell=separate.handle)
    assert end != other_start
    torch.testing.assert_close(exported.model.coefficients[end], exported.model.coefficients[other_start], rtol=0, atol=0)
    assert space.cells[vertex.cell_id].dimensions == ()
    points = Points.from_cells(space, [(first.handle, [[0.0, 1.0]]), (second.handle, [[1.0, 0.0]])])
    values = exported.model(points).detach()
    torch.testing.assert_close(values[0], values[1], rtol=0, atol=0)
    torch.testing.assert_close(exported.predict([[-0.55]])[0], values[0], rtol=0, atol=0)


def test_degree_elevation_and_cell_orientation_keep_start_and_end_values():
    source = _result(
        [
            _spline([-1.0, -0.25], 1, [[10, 20], [30, 40]], [1, -1]),
            _spline([0.25, 1.0], 4, [[0, 0], [1, 1], [2, 1], [3, 1], [4, 0]]),
        ]
    )
    exported = export_curve(source)
    cell = exported.intervals[0]
    for k in range(5):
        row = exported.model.space.control_index((4 - k, k), cell=cell.handle)
        expected = torch.tensor([11 + 5 * k, 19 + 5 * k], dtype=torch.float64)
        torch.testing.assert_close(exported.model.coefficients[row], expected, rtol=0, atol=1e-14)
    torch.testing.assert_close(
        exported.predict([[-1.0], [-0.25]]),
        torch.tensor([[11.0, 19.0], [31.0, 39.0]], dtype=torch.float64),
        rtol=0,
        atol=0,  # Endpoint evaluation selects the stored endpoint controls exactly.
    )


def test_isolated_vertices_remain_separate_even_when_their_images_coincide():
    source = _result([_spline([-0.5], 0, [[1, 2]]), _spline([0.5], 0, [[1, 2]])])
    exported = export_curve(source)
    assert exported.domain.betti_numbers() == [2]
    assert exported.model.space.n_controls == 2 and exported.model.space.width == 0
    assert exported.intervals[0].cell_id != exported.intervals[1].cell_id
    x = torch.tensor([[-1.0], [-0.5], [0.0], [0.5], [1.0]], dtype=torch.float64)
    torch.testing.assert_close(exported.predict(x), source.predict(x), rtol=0, atol=0, equal_nan=True)


def test_degree_zero_interval_and_grid_endpoint_clamping_are_preserved():
    source = _result([_spline([-0.9, -0.1, 0.9], 0, [[1, 3]], [2, -1])], bounds=(-0.91, 0.94))
    exported = export_curve(source)
    assert exported.degree == 1 and exported.domain.betti_numbers() == [1, 0]
    assert exported.model.space.n_controls == 3
    x = torch.tensor([[-0.91], [-0.9], [0.0], [0.9], [0.94]], dtype=torch.float64)
    torch.testing.assert_close(exported.predict(x), source.predict(x), rtol=0, atol=0)


def test_export_is_independent_of_later_source_mutations_and_does_not_change_inputs():
    source = _mixed()
    exported = export_curve(source)
    x = torch.tensor([[-0.9], [-0.4], [0.3], [0.85]], dtype=torch.float64, requires_grad=True)
    saved, before = x.detach().clone(), exported.predict(x)
    source._models[0].controls.add_(10)
    source._models[1].center.add_(20)
    source._models[-1].knots.add_(0.01)
    torch.testing.assert_close(exported.predict(x), before, rtol=0, atol=0)
    torch.testing.assert_close(x, saved, rtol=0, atol=0)


@st.composite
def _coefficients_and_degree(draw):
    degree = draw(st.integers(0, 5))
    outputs = draw(st.integers(1, 3))
    target = degree + draw(st.integers(0, 4))
    values = draw(
        st.lists(
            st.floats(-3, 3, allow_nan=False, allow_infinity=False),
            min_size=(degree + 1) * outputs,
            max_size=(degree + 1) * outputs,
        )
    )
    t = draw(st.floats(0, 1, allow_nan=False, allow_infinity=False))
    return torch.tensor(values, dtype=torch.float64).reshape(degree + 1, outputs), target, t


def _casteljau(controls, t):
    values = controls.clone()
    while len(values) > 1:
        values = (1 - t) * values[:-1] + t * values[1:]
    return values[0]


@given(_coefficients_and_degree())
@settings(deadline=None)
def test_random_degree_elevation_preserves_polynomials_against_de_casteljau(case):
    controls, target, t = case
    raised = _elevate_controls(controls, target)
    torch.testing.assert_close(_casteljau(raised, t), _casteljau(controls, t), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(raised[0], controls[0], rtol=0, atol=0)
    torch.testing.assert_close(raised[-1], controls[-1], rtol=0, atol=0)


@pytest.mark.parametrize("points", [[[2.0]], [[math.nan]], [[0.0, 0.0]], [[1j]]])
def test_invalid_prediction_inputs_are_rejected(points):
    with pytest.raises(ValueError):
        export_curve(_mixed()).predict(points)


def test_invalid_source_geometry_or_controls_are_not_silently_exported():
    with pytest.raises(TypeError):
        export_curve(object())  # type: ignore[arg-type]
    source = _mixed()
    source._models[0].knots[1] = source._models[0].knots[0]
    with pytest.raises(ValueError, match="knots"):
        export_curve(source)
    source = _mixed()
    source._models[0].controls[0, 0] = math.nan
    with pytest.raises(ValueError, match="coefficients"):
        export_curve(source)
