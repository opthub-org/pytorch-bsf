"""Contract tests for the unified v0 API, independent of the private geometry kernels."""

import pytest
import torch

import torch_bsf as bsf
from torch_bsf.solvers import DenseLeastSquares


def test_single_cell_fit_returns_an_explicit_result():
    space = bsf.BezierSpace.simplex(dimension=1, degree=2)
    t = torch.tensor([[1.0, 0.0], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0.0, 1.0]])
    result = bsf.fit(space, bsf.Samples(parameters=t, values=t[:, :1] ** 2))
    assert isinstance(result, bsf.FitResult)
    assert isinstance(result.model, bsf.BezierModel)
    assert result.success
    assert not callable(result)
    assert result.model.coefficients.dtype == torch.float64
    assert torch.allclose(result.model(t), (t[:, :1] ** 2).double(), atol=1e-8)
    assert result.metrics["data_mse"] < 1e-20


def test_fixed_values_do_not_require_an_initial_model():
    space = bsf.BezierSpace.simplex(dimension=1, degree=1)
    t = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    fixed = bsf.FixedControls({space.control_index((1, 0)): [3.0]})
    result = bsf.fit(space, bsf.Samples(t, torch.zeros(2, 1)), fixed=fixed)
    assert result.model(t)[0, 0] == 3.0
    assert result.model(t)[1, 0] == 0.0
    assert result.diagnostics.n_free_controls == 1


def test_multi_cell_and_single_cell_use_the_same_model_and_result():
    domain = bsf.SimploidalSet()
    tri, square = domain.add_simplex(2), domain.add_simploid((1, 1))
    domain.identify(tri.face(0, 2), square.face(1, 1))
    space = bsf.BezierSpace.on_set(domain, degree=2)
    t1 = torch.tensor([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float64)
    t2 = torch.tensor([[0.3, 0.7, 0.4, 0.6], [1.0, 0.0, 1.0, 0.0]], dtype=torch.float64)
    observations = bsf.Samples.from_cells(space, [(tri, t1, t1[:, :1]), (square, t2, t2[:, :1])])
    result = bsf.fit(space, observations)
    assert result.model(observations.points).shape == (5, 1)
    edge1 = bsf.Points(torch.tensor([[0.4, 0.6, 0.0]]), [space.cell_id(tri)])
    edge2 = bsf.Points(torch.tensor([[0.4, 0.6, 1.0, 0.0]]), [space.cell_id(square)])
    assert torch.allclose(result.model(edge1), result.model(edge2), atol=1e-12)


def test_space_snapshots_mutable_topology():
    domain = bsf.SimploidalSet()
    domain.add_simplex(1)
    space = bsf.BezierSpace.on_set(domain, degree=2)
    before = (space.n_cells, space.n_controls)
    domain.add_simplex(2)
    assert (space.n_cells, space.n_controls) == before
    assert not isinstance(space, torch.nn.Module)


def test_both_model_parameters_and_coordinates_are_differentiable():
    space = bsf.BezierSpace.simplex(1, degree=2)
    model = bsf.BezierModel(space, torch.arange(3.0).reshape(3, 1))
    t = torch.tensor([[0.4, 0.6]], requires_grad=True)
    model(t).sum().backward()
    assert t.grad is not None and torch.isfinite(t.grad).all()
    assert model.coefficients.grad is not None


def test_sample_replication_does_not_change_regularization():
    space = bsf.BezierSpace.simplex(1, degree=2)
    t = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=torch.float64)
    y = torch.tensor([[1.0], [0.0], [2.0]])
    penalty = bsf.QuadraticPenalty(ridge=0.1, smoothness=0.2)
    a = bsf.fit(space, bsf.Samples(t, y), penalty=penalty)
    b = bsf.fit(space, bsf.Samples(t.repeat(3, 1), y.repeat(3, 1)), penalty=penalty)
    assert torch.allclose(a.model.coefficients, b.model.coefficients, atol=1e-12)


def test_diagnostics_do_not_change_return_type():
    space = bsf.BezierSpace.simplex(1, degree=1)
    samples = bsf.Samples(torch.eye(2), torch.ones(2, 1))
    for level in ("basic", "full"):
        result = bsf.fit(space, samples, solver=DenseLeastSquares(diagnostics=level))
        assert isinstance(result, bsf.FitResult)
        assert result.require_success() is result


def test_old_public_entry_points_are_removed():
    for name in ("BezierSimploid", "SimploidalBezierSpace", "fit_simploidal_set", "fit_kfold", "fit_sgd"):
        assert not hasattr(bsf, name)


@pytest.mark.parametrize("weights", [[0.0, 0.0], [-1.0, 2.0], [float("nan"), 1.0]])
def test_invalid_sample_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="weights"):
        bsf.Samples(torch.eye(2), torch.ones(2, 1), weights=weights)


def test_unsupported_options_are_not_silently_ignored():
    with pytest.raises(TypeError):
        DenseLeastSquares(max_epochs=2)  # type: ignore[call-arg]


def test_mixed_degrees_keep_shared_face_coefficients():
    from torch_bsf.sampling import sample

    domain = bsf.SimploidalSet()
    prism, cube = domain.add_simploid((2, 1)), domain.add_simploid((1, 1, 1))
    domain.identify(prism.face(0, 0), cube.face(0, 0))
    space = bsf.BezierSpace.on_set(domain, degree={prism: (2, 3), cube: (4, 2, 3)})
    points = sample(space, 120, seed=7)
    # A constant field is consistent across all incidences and degree assignments.
    observations = bsf.Samples(points, torch.full((120, 2), 3.0))
    result = bsf.fit(space, observations)
    assert result.success
    assert torch.allclose(result.model(points), observations.values.double(), atol=1e-10)
    assert space.cells[space.cell_id(prism)].degree == (2, 3)
    assert space.cells[space.cell_id(cube)].degree == (4, 2, 3)


@pytest.mark.parametrize("field", ["coordinates", "values", "coefficients", "fixed"])
def test_complex_inputs_are_rejected(field):
    t = torch.eye(2, dtype=torch.complex128)
    with pytest.raises(ValueError, match="real"):
        if field == "coordinates":
            bsf.Points(t)
        elif field == "values":
            bsf.Samples(t.real, t)
        elif field == "coefficients":
            bsf.BezierModel(bsf.BezierSpace.simplex(1, degree=1), t)
        else:
            bsf.FixedControls({0: [1j]})


def test_multi_cell_shortcut_cannot_silently_predict_on_cell_zero():
    domain = bsf.SimploidalSet()
    domain.add_simplex(2)
    space = bsf.BezierSpace.on_set(domain, degree=2)
    model = bsf.BezierModel(space, torch.zeros(space.n_controls, 1))
    with pytest.raises(ValueError, match="cell IDs"):
        model(torch.tensor([[0.2, 0.3, 0.5]]))


def test_shared_fixed_references_reject_conflicts():
    domain = bsf.SimploidalSet()
    a, b = domain.add_simplex(1), domain.add_simplex(1)
    domain.identify(a.face(0, 0), b.face(0, 1))
    space = bsf.BezierSpace.on_set(domain, degree=2)
    left = (a, (0, 2), [1.0])
    right = (b, (2, 0), [2.0])
    with pytest.raises(ValueError, match="conflicting"):
        bsf.FixedControls.from_cells(space, [left, right])
    fixed = bsf.FixedControls.from_cells(space, [left, (b, (2, 0), [1.0])])
    assert len(fixed.values) == 1
    points = bsf.Points.from_cells(space, [(a, [[0.0, 1.0]]), (b, [[1.0, 0.0]])])
    assert points.cell_ids.tolist() == [space.cell_id(a), space.cell_id(b)]


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["list", "tuple", "numpy32", "numpy64", "tensor32", "tensor64", "integer"])
def test_fixed_controls_parse_non_tensor_floats_without_default_dtype_rounding(default_dtype, kind):
    import numpy as np

    inputs = {
        "list": [0.2],
        "tuple": (0.2,),
        "numpy32": np.array([0.2], dtype=np.float32),
        "numpy64": np.array([0.2], dtype=np.float64),
        "tensor32": torch.tensor([0.2], dtype=torch.float32),
        "tensor64": torch.tensor([0.2], dtype=torch.float64),
        "integer": [2],
    }
    value = inputs[kind]
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        actual = bsf.FixedControls({0: value}).values[0]
    finally:
        torch.set_default_dtype(previous)
    expected = torch.as_tensor(value, dtype=torch.float64)
    assert actual.dtype == torch.float64
    assert actual.device.type == "cpu"
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("value", [[0.2j], (0.2j,), torch.tensor([0.2j])])
def test_fixed_controls_reject_complex_inputs_before_conversion(value):
    with pytest.raises(ValueError, match="real"):
        bsf.FixedControls({0: value})


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["list", "numpy32", "numpy64", "tensor32", "tensor64", "integer"])
def test_core_containers_preserve_existing_dtype_inference(default_dtype, kind):
    import numpy as np

    inputs = {
        "list": [[0.2, 0.8]],
        "numpy32": np.array([[0.2, 0.8]], dtype=np.float32),
        "numpy64": np.array([[0.2, 0.8]], dtype=np.float64),
        "tensor32": torch.tensor([[0.2, 0.8]], dtype=torch.float32),
        "tensor64": torch.tensor([[0.2, 0.8]], dtype=torch.float64),
        "integer": [[0, 1]],
    }
    value = inputs[kind]
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(default_dtype)
        inferred = torch.as_tensor(value)
        expected = inferred if inferred.is_floating_point() else inferred.double()
        points = bsf.Points(value)
        samples = bsf.Samples(value, value)
        model = bsf.BezierModel(bsf.BezierSpace.simplex(0, degree=0), value)
    finally:
        torch.set_default_dtype(previous)
    for actual in (points.parameters, samples.points.parameters, model.coefficients):
        assert actual.dtype == expected.dtype
        assert torch.equal(actual, expected)
    assert samples.values.dtype == inferred.dtype
    assert torch.equal(samples.values, inferred)
