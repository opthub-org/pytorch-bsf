"""Accelerator inputs must move to CPU before conversion to float64 (MPS has no float64)."""

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from torch_bsf import BezierModel, BezierSpace, FixedControls, Points, Samples, fit, fit_point_cloud
from torch_bsf._kernels._splitting import longest_edge_criterion
from torch_bsf._kernels.api import BezierSimploid
from torch_bsf.interpolation import SimplicialMesh, hermite_interpolant, lattice_mesh


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")),
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    ]
)
def input_device(request):
    return request.param


def test_mesh_copies_vertices_before_float64_conversion(input_device):
    vertices = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], device=input_device)
    expected = vertices.cpu().double().clone()
    mesh = SimplicialMesh(vertices, [[0, 1, 2]])
    vertices.zero_()
    torch.testing.assert_close(mesh.vertices, expected)


def test_locate_accepts_device_points(input_device):
    mesh = lattice_mesh(2, 2)
    points = torch.tensor([[0.125, 0.25], [0.5, 0.125]], device=input_device)
    actual, expected = mesh.locate(points), mesh.locate(points.cpu())
    torch.testing.assert_close(actual.parameters, expected.parameters)
    torch.testing.assert_close(actual.cell_ids, expected.cell_ids)


@pytest.mark.parametrize("supplied_gradients", [False, True])
def test_hermite_accepts_device_values_and_gradients(input_device, supplied_gradients):
    mesh = lattice_mesh(2, 2)
    values = mesh.vertices.float().sum(dim=1, keepdim=True).to(input_device)
    gradients = torch.ones(mesh.n_vertices, mesh.dimension, 1, device=input_device) if supplied_gradients else None
    actual = hermite_interpolant(mesh, values, gradients=gradients)
    expected = hermite_interpolant(mesh, values.cpu(), gradients=gradients.cpu() if gradients is not None else None)
    torch.testing.assert_close(actual.coefficients, expected.coefficients)
    torch.testing.assert_close(actual(mesh.locate(mesh.vertices)), values.cpu().double())


def test_fixed_controls_copy_device_values_and_preserve_autograd(input_device):
    values = torch.tensor([0.25, 1e30], device=input_device, requires_grad=True)
    actual = FixedControls({0: values}).values[0]
    torch.testing.assert_close(actual, values.detach().cpu().double())
    actual.sum().backward()
    torch.testing.assert_close(values.grad, torch.ones_like(values))
    with torch.no_grad():
        values.zero_()
    assert actual[0] == 0.25


def test_point_cloud_accepts_device_initial_model(input_device):
    space = BezierSpace.simplex(1, degree=1)
    coefficients = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    values = torch.tensor([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    actual = fit_point_cloud(space, values, initial=BezierModel(space, coefficients.to(input_device))).require_success()
    expected = fit_point_cloud(space, values, initial=BezierModel(space, coefficients)).require_success()
    torch.testing.assert_close(actual.model.coefficients, expected.model.coefficients)


def test_longest_edge_accepts_device_control_points(input_device):
    coefficients = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    actual = longest_edge_criterion(BezierSimploid((2,), 1, coefficients.to(input_device)))
    expected = longest_edge_criterion(BezierSimploid((2,), 1, coefficients))
    assert actual == expected == (0, 1, 2, 0.5)


def test_points_to_cpu_preserves_values_ids_and_autograd(input_device):
    coordinates = torch.tensor([[0.25, 0.75], [1.0, 0.0]], device=input_device, requires_grad=True)
    ids = torch.tensor([2, 7], device=input_device)
    points = Points(coordinates, ids)
    actual = points.to()
    torch.testing.assert_close(actual.parameters, coordinates.detach().cpu().double(), rtol=0, atol=0)
    torch.testing.assert_close(actual.cell_ids, ids.cpu())
    assert actual.parameters.device.type == "cpu"
    assert actual.parameters.dtype == torch.float64
    actual.parameters.square().sum().backward()
    torch.testing.assert_close(coordinates.grad, 2 * coordinates.detach())
    with torch.no_grad():
        coordinates.zero_()
        ids.zero_()
    torch.testing.assert_close(points.parameters.cpu(), torch.tensor([[0.25, 0.75], [1.0, 0.0]]))
    torch.testing.assert_close(actual.cell_ids, torch.tensor([2, 7]))
    with torch.no_grad():
        points.parameters.zero_()
        points.cell_ids.zero_()
    torch.testing.assert_close(
        actual.parameters, torch.tensor([[0.25, 0.75], [1.0, 0.0]], dtype=torch.float64), rtol=0, atol=0
    )
    torch.testing.assert_close(actual.cell_ids, torch.tensor([2, 7]))


def test_fit_and_predict_accept_device_coordinates_and_weights(input_device):
    space = BezierSpace.simplex(1, degree=1)
    coordinates = torch.tensor([[1.0, 0.0], [0.25, 0.75], [0.0, 1.0]], device=input_device)
    values = torch.tensor([[1.0], [1.75], [2.0]], device=input_device)
    weights = torch.tensor([1.0, 0.5, 2.0], device=input_device, requires_grad=True)
    samples = Samples(coordinates, values, weights=weights)
    torch.testing.assert_close(samples.weights, weights.detach().cpu().double())
    samples.weights.sum().backward()
    torch.testing.assert_close(weights.grad, torch.ones_like(weights))
    actual = fit(space, samples).require_success().model
    expected = fit(space, Samples(coordinates.cpu(), values.cpu(), weights=weights.detach().cpu())).require_success().model
    torch.testing.assert_close(actual.coefficients, expected.coefficients)
    predictions = actual(coordinates)
    torch.testing.assert_close(predictions, values.cpu().double(), rtol=32 * torch.finfo(torch.float64).eps, atol=0)
    assert predictions.device.type == "cpu"
    assert predictions.dtype == torch.float64
    with torch.no_grad():
        weights.zero_()
    torch.testing.assert_close(samples.weights, torch.tensor([1.0, 0.5, 2.0], dtype=torch.float64))


def test_point_cloud_accepts_device_observations(input_device):
    space = BezierSpace.simplex(1, degree=1)
    values = torch.tensor([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], device=input_device)
    actual = fit_point_cloud(space, values).require_success()
    expected = fit_point_cloud(space, values.cpu()).require_success()
    torch.testing.assert_close(actual.model.coefficients, expected.model.coefficients)
    torch.testing.assert_close(actual.model(actual.parameters), expected.model(expected.parameters))
    torch.testing.assert_close(values.cpu(), torch.tensor([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]))


def test_integer_device_coordinates_and_coefficients_promote_to_float64(input_device):
    coordinates = torch.tensor([[1, 0], [0, 1]], device=input_device)
    coefficients = torch.tensor([[3], [7]], device=input_device)
    points = Points(coordinates)
    model = BezierModel(BezierSpace.simplex(1, degree=1), coefficients)
    expected_device = "cpu" if input_device == "mps" else input_device
    assert points.parameters.device.type == model.coefficients.device.type == expected_device
    assert points.parameters.dtype == model.coefficients.dtype == torch.float64
    torch.testing.assert_close(points.parameters.cpu(), coordinates.cpu().double())
    torch.testing.assert_close(model(points).cpu(), coefficients.cpu().double())
    torch.testing.assert_close(coordinates.cpu(), torch.eye(2, dtype=torch.long))
    torch.testing.assert_close(coefficients.cpu(), torch.tensor([[3], [7]]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_points_to_accepts_explicit_compute_device(input_device, dtype):
    if input_device == "mps" and dtype == torch.float64:
        pytest.skip("MPS cannot compute in float64")
    points = Points(torch.tensor([[0.25, 0.75]], dtype=torch.float64))
    actual = points.to(device=input_device, dtype=dtype)
    assert actual.parameters.dtype == dtype
    assert actual.parameters.device.type == input_device
    assert actual.cell_ids.device.type == input_device
    torch.testing.assert_close(actual.parameters.cpu(), torch.tensor([[0.25, 0.75]], dtype=dtype))


def test_python_weights_and_observations_keep_float64_precision():
    value = 1 + 2**-40
    samples = Samples([[1.0]], [[0.0]], weights=[value])
    assert samples.weights.item() == value
    fitted = fit_point_cloud(BezierSpace.simplex(0, degree=0), [[value]]).require_success()
    assert fitted.model.coefficients.item() == value


@given(st.lists(st.floats(width=32, allow_nan=False, allow_infinity=False), min_size=0, max_size=30))
def test_points_cpu_float64_conversion_preserves_float32_values(values):
    source = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
    points = Points(source)
    converted = points.to()
    assert converted.parameters.dtype == torch.float64
    assert converted.parameters.device.type == "cpu"
    torch.testing.assert_close(converted.parameters, source.double(), rtol=0, atol=0)
    torch.testing.assert_close(points.parameters, source, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_points_to_meta_preserves_metadata_and_source(dtype):
    """Device-independent transfers must retain the public Points metadata contract."""
    coordinates = torch.tensor([[0.25, 0.75], [1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    points = Points(coordinates, [2, 7])
    actual = points.to(device="meta", dtype=dtype)
    assert actual.parameters.device.type == actual.cell_ids.device.type == "meta"
    assert actual.parameters.dtype == dtype
    assert actual.cell_ids.dtype == torch.long
    assert actual.parameters.shape == points.parameters.shape == (2, 2)
    assert actual.cell_ids.shape == points.cell_ids.shape == (2,)
    assert len(actual) == len(points) == 2
    assert actual.parameters.requires_grad
    assert actual.parameters.grad_fn is not None
    assert actual.parameters is not points.parameters
    assert actual.cell_ids is not points.cell_ids
    assert points.parameters.device.type == points.cell_ids.device.type == "cpu"
    assert points.parameters.dtype == torch.float64
    torch.testing.assert_close(points.parameters, coordinates, rtol=0, atol=0)
    torch.testing.assert_close(points.cell_ids, torch.tensor([2, 7]))
