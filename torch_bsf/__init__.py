"""torch_bsf: PyTorch implementation of Bézier simplex fitting.

"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pytorch-bsf")
except PackageNotFoundError:
    # Package is not installed (e.g., running from source without pip install)
    __version__ = "unknown"

from torch_bsf.bezier_simplex import BezierSimplex, BezierSimplexDataModule, fit, fit_kfold, validate_control_points
from torch_bsf.simploidal import (
    BezierSimploid,
    Simploid,
    SimploidalBezierSpace,
    SimploidalSet,
    fit_simploid,
    fit_simploidal_set,
)
from torch_bsf.splitting import (
    SplitCriterion,
    longest_edge_criterion,
    max_error_criterion,
    reparametrize,
    split,
    split_by_criterion,
)

__all__ = [
    "BezierSimplex",
    "BezierSimplexDataModule",
    "fit",
    "fit_kfold",
    "validate_control_points",
    "SplitCriterion",
    "split",
    "reparametrize",
    "longest_edge_criterion",
    "max_error_criterion",
    "split_by_criterion",
    "BezierSimploid",
    "Simploid",
    "SimploidalSet",
    "SimploidalBezierSpace",
    "fit_simploid",
    "fit_simploidal_set",
]
