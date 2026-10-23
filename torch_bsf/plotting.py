"""A single plotting entry point for single- and multiple-cell models."""

from collections.abc import Sequence
from typing import Any, cast

import torch

from torch_bsf._kernels._plotting import _check_simploid_plot_args, _draw_simploid, _prepare_axes
from torch_bsf._kernels.combinatorics import _indices
from torch_bsf.core import BezierModel, Points


def plot_model(
    model: BezierModel,
    *,
    resolution: int = 20,
    coordinates: Sequence[int] | None = None,
    ax: Any = None,
    show_controls: bool = True,
    **kwargs: Any,
) -> Any:
    """Draw maximal cells and their control nets on shared Matplotlib axes.

    Select two or three output coordinates for higher-dimensional models.
    Domains above dimension two are shown through their two-dimensional faces.
    Returns the axes. Matplotlib is imported only when plotting.
    """
    selected = tuple(range(model.n_outputs)) if coordinates is None else tuple(coordinates)
    if len(set(selected)) != len(selected) or any(i < 0 or i >= model.n_outputs for i in selected):
        raise ValueError("coordinates must contain distinct valid output indices.")
    _check_simploid_plot_args(len(selected), resolution, ax, "plot_model")
    _, ax = _prepare_axes(ax, len(selected))
    options = dict(kwargs)
    color = options.pop("color", "C0")
    for cid in model.space.main_cells:
        cell = model.space.cells[cid]

        def evaluate(t: torch.Tensor, cell_id: int = cid) -> torch.Tensor:
            ids = torch.full((len(t),), cell_id, dtype=torch.long)
            return cast(torch.Tensor, model(Points(t, ids))[:, selected])

        _draw_simploid(
            ax,
            cell.dimensions,
            evaluate,
            list(_indices(cell.dimensions, cell.degree)),
            model.coefficients[list(cell.controls)][:, selected],
            resolution,
            show_controls,
            color,
            options,
        )
    ax.autoscale_view()
    return ax
