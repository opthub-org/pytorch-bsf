import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # Non-interactive backend for tests
pytest.importorskip("scipy")  # Required for triangle plotting via Delaunay

import torch_bsf.bezier_simplex as tbbs
from torch_bsf.plotting import plot_bezier_simplex


@pytest.fixture
def bezier_curve_2d():
    return tbbs.randn(n_params=2, n_values=2, degree=2)


@pytest.fixture
def bezier_curve_3d():
    return tbbs.randn(n_params=2, n_values=3, degree=2)


@pytest.fixture
def bezier_triangle_2d():
    return tbbs.randn(n_params=3, n_values=2, degree=2)


@pytest.fixture
def bezier_triangle_3d():
    return tbbs.randn(n_params=3, n_values=3, degree=2)


def test_plot_bezier_curve_2d_returns_axes(bezier_curve_2d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_curve_2d, num=10)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_bezier_curve_3d_returns_axes(bezier_curve_3d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_curve_3d, num=10)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_bezier_triangle_2d_returns_axes(bezier_triangle_2d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_triangle_2d, num=5)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_bezier_triangle_3d_returns_axes(bezier_triangle_3d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_triangle_3d, num=5)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_bezier_simplex_not_implemented():
    model = tbbs.randn(n_params=4, n_values=2, degree=1)
    with pytest.raises(NotImplementedError):
        plot_bezier_simplex(model)


def test_plot_curve_with_existing_axes(bezier_curve_2d):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result = plot_bezier_simplex(bezier_curve_2d, num=10, ax=ax)
    assert result is ax
    plt.close(fig)


def test_plot_triangle_with_existing_axes(bezier_triangle_2d):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    result = plot_bezier_simplex(bezier_triangle_2d, num=5, ax=ax)
    assert result is ax
    plt.close(fig)


def test_plot_curve_without_control_points(bezier_curve_2d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_curve_2d, num=10, show_control_points=False)
    assert ax is not None
    plt.close(ax.figure)


def test_plot_triangle_without_control_points(bezier_triangle_2d):
    import matplotlib.pyplot as plt

    ax = plot_bezier_simplex(bezier_triangle_2d, num=5, show_control_points=False)
    assert ax is not None
    plt.close(ax.figure)
