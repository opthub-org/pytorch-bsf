"""Bézier fitting with explicit spaces, observations, models and numerical results."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pytorch-bsf")
except PackageNotFoundError:
    # Package is not installed (e.g., running from source without pip install)
    __version__ = "unknown"

from torch_bsf.core import BezierModel, BezierSpace, FixedControls, Points, QuadraticPenalty, Samples
from torch_bsf.ensemble import Ensemble
from torch_bsf.fitting import fit
from torch_bsf.geometry import Simploid, SimploidalSet
from torch_bsf.optimization import OptimizationResult, PreconditionedSGD, linear_scalarization, minimize
from torch_bsf.point_cloud import PointCloudFitResult, SkeletonFitResult, fit_point_cloud, fit_skeleton
from torch_bsf.results import ConvergenceError, ConvergenceWarning, FitDiagnostics, FitResult, MatrixDiagnostics

__all__ = [
    "BezierSpace",
    "BezierModel",
    "Points",
    "Samples",
    "FixedControls",
    "QuadraticPenalty",
    "Simploid",
    "SimploidalSet",
    "fit",
    "fit_point_cloud",
    "fit_skeleton",
    "PointCloudFitResult",
    "SkeletonFitResult",
    "FitResult",
    "FitDiagnostics",
    "MatrixDiagnostics",
    "ConvergenceError",
    "ConvergenceWarning",
    "Ensemble",
    "OptimizationResult",
    "PreconditionedSGD",
    "linear_scalarization",
    "minimize",
]
