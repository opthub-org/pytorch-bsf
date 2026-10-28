"""Accelerator inputs must move to CPU before conversion to float64 (MPS has no float64)."""

import pytest
import torch

from torch_bsf import BezierModel, BezierSpace, FixedControls, fit_point_cloud
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
