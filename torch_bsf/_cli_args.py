"""Shared parsing for the explicit command-line operations."""

import importlib
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torch_bsf.core import BezierSpace, QuadraticPenalty, Samples
from torch_bsf.solvers import DenseLeastSquares, LeastSquaresSolver, SparseLeastSquares


def integers(text: str) -> tuple[int, ...]:
    """Parse comma-separated nonnegative integers."""
    try:
        result = tuple(int(part) for part in text.split(","))
        if not result or min(result) < 0:
            raise ValueError
        return result
    except ValueError:
        raise ArgumentTypeError("expected comma-separated nonnegative integers") from None


def load_table(path: Path, header: int = 0) -> torch.Tensor:
    """Read a numeric CSV, preserving singleton rows and columns."""
    return torch.as_tensor(np.loadtxt(path, delimiter=",", skiprows=header, ndmin=2), dtype=torch.float64)


def import_callable(spec: str) -> Any:
    """Import an explicit module:function objective."""
    try:
        module, name = spec.split(":")
        objective = getattr(importlib.import_module(module), name)
    except (ValueError, ImportError, AttributeError) as error:
        raise ValueError(f"cannot import objective {spec!r}: {error}") from error
    if not callable(objective):
        raise ValueError("objective must be callable")
    return objective


def add_data(parser: ArgumentParser) -> None:
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--values", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--header", type=int, default=0)
    parser.add_argument("--dimensions", type=integers, help="Simplex dimensions of product factors, e.g. 2,1")
    parser.add_argument("--degree", type=int, required=True)
    parser.add_argument("--solver", choices=("dense", "sparse"), default="dense")
    parser.add_argument("--diagnostics", choices=("basic", "full"), default="basic")
    parser.add_argument("--ridge", type=float, default=0)
    parser.add_argument("--smoothness", type=float, default=0)


def regression_options(args: Namespace) -> tuple[BezierSpace, Samples, LeastSquaresSolver, QuadraticPenalty]:
    """Build the same problem and typed settings used by the Python API."""
    t, y = load_table(args.params, args.header), load_table(args.values, args.header)
    dimensions = args.dimensions if args.dimensions is not None else ((t.shape[1] - 1,) if t.shape[1] > 1 else ())
    space = BezierSpace.simploid(dimensions, degree=args.degree)
    if space.width == 0 and t.shape[1] == 1:
        t = t[:, :0]
    if t.shape[1] != space.width:
        raise ValueError("parameter CSV columns do not match the domain.")
    weights = load_table(args.weights, args.header).flatten() if args.weights else None
    solver: LeastSquaresSolver = (
        DenseLeastSquares(diagnostics=args.diagnostics) if args.solver == "dense" else SparseLeastSquares()
    )
    if args.solver == "sparse" and args.diagnostics != "basic":
        raise ValueError("full SVD diagnostics require --solver=dense")
    return space, Samples(t, y, weights=weights), solver, QuadraticPenalty(args.ridge, args.smoothness)
