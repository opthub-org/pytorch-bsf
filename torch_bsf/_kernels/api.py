"""torch_bsf: PyTorch implementation of Bézier simploid fitting.

BSF stands for Bézier Simploid Fitting (formerly Bézier Simplex Fitting).
It fits Bézier simplices and, more generally, Bézier simploids, i.e., tensor
products of Bézier simplices, and simploidal sets glued from them.
A Bézier simplex is the Bézier simploid of a single factor, so both are
:class:`BezierSimploid` models fitted by :func:`fit`.
"""

from torch_bsf._kernels import BezierSimploid, Simploid, SimploidalBezierSpace, SimploidalSet, fit_simploidal_set
from torch_bsf._kernels._splitting import (
    SplitCriterion,
    longest_edge_criterion,
    max_error_criterion,
    reparametrize,
    split,
    split_by_criterion,
)
from torch_bsf._kernels.fitting import fit, fit_kfold
from torch_bsf._kernels.sgd import fit_sgd

__all__ = [
    "BezierSimploid",
    "fit",
    "fit_kfold",
    "fit_sgd",
    "SplitCriterion",
    "split",
    "reparametrize",
    "longest_edge_criterion",
    "max_error_criterion",
    "split_by_criterion",
    "Simploid",
    "SimploidalSet",
    "SimploidalBezierSpace",
    "fit_simploidal_set",
]
