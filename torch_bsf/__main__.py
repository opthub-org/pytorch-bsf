"""Explicit CLI operations, all backed by the public Python API."""

import json
from argparse import ArgumentParser, Namespace
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from torch_bsf import Points, fit
from torch_bsf._cli_args import add_data, import_callable, load_table, regression_options
from torch_bsf.io import load_model, save_model
from torch_bsf.model_selection import KFold, cross_validate, select_degree
from torch_bsf.optimization import PreconditionedSGD, linear_scalarization, minimize


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="python -m torch_bsf", description="Bézier fitting and objective minimization")
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("fit", help="Solve weighted least squares")
    add_data(fitting)
    fitting.add_argument("--save", type=Path, required=True)
    fitting.add_argument("--track", action="store_true", help="Record results with the optional MLflow adapter")
    fitting.add_argument("--tracking-uri")
    for name in ("cross-validate", "select-degree"):
        validation = commands.add_parser(name)
        add_data(validation)
        validation.add_argument("--folds", type=int, default=5)
        validation.add_argument("--seed", type=int, default=0)
        if name == "select-degree":
            validation.add_argument("--degrees", type=int, nargs="+", required=True)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--model", type=Path, required=True)
    prediction.add_argument("--params", type=Path, required=True)
    prediction.add_argument("--cell-ids", type=Path, help="One integer cell ID per CSV row for a multi-cell model")
    prediction.add_argument("--save", type=Path, required=True)
    optimization = commands.add_parser("minimize")
    optimization.add_argument("--init", type=Path, required=True, help="Explicit initial model in .bsf format")
    optimization.add_argument("--objective", required=True, help="Differentiable vector objective as module:function")
    optimization.add_argument("--save", type=Path, required=True)
    optimization.add_argument("--max-iter", type=int, default=100)
    optimization.add_argument("--step-size", type=float, default=0.05)
    optimization.add_argument("--n-samples", type=int, default=100)
    optimization.add_argument("--seed", type=int, default=0)
    return parser


def _regression(args: Namespace) -> tuple[dict[str, Any], bool]:
    space, samples, solver, penalty = regression_options(args)
    if args.command == "fit":
        if args.tracking_uri and not args.track:
            raise ValueError("--tracking-uri requires --track")
        result = fit(space, samples, solver=solver, penalty=penalty)
        save_model(args.save, result.model)
        if args.track:
            from torch_bsf.integrations.mlflow import log_result

            log_result(result, tracking_uri=args.tracking_uri)
        return {
            "success": result.success,
            "status": result.status,
            "metrics": result.metrics,
            "diagnostics": asdict(result.diagnostics),
        }, result.success
    cv = KFold(args.folds, seed=args.seed)
    if args.command == "cross-validate":
        validated = cross_validate(space, samples, cv=cv, solver=solver, penalty=penalty)
        return {
            "success": validated.success,
            "metrics": validated.metrics,
            "folds": [
                {
                    "train": f.train_indices.tolist(),
                    "test": f.test_indices.tolist(),
                    "metrics": f.metrics,
                    "status": f.result.status,
                }
                for f in validated.folds
            ],
        }, validated.success
    selected = select_degree(space, samples, degrees=args.degrees, cv=cv, solver=solver, penalty=penalty)
    return {
        "success": selected.success,
        "best_degree": selected.best_degree,
        "candidates": [{"degree": c.degree, "metrics": c.validation.metrics} for c in selected.candidates],
    }, selected.success


def main(argv: Sequence[str] | None = None) -> None:
    """Run an explicit operation; failed convergence produces exit status one."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "predict":
            model = load_model(args.model)
            parameters = load_table(args.params)
            if model.space.width == 0 and parameters.shape[1] == 1:
                parameters = parameters[:, :0]
            points: Points | torch.Tensor = parameters
            if args.cell_ids is not None:
                identifiers = load_table(args.cell_ids)
                if (
                    identifiers.shape != (len(parameters), 1)
                    or not torch.isfinite(identifiers).all()
                    or not torch.equal(identifiers, identifiers.round())
                ):
                    raise ValueError("--cell-ids must have one integer per parameter row")
                points = Points(parameters, identifiers[:, 0].long())
            with torch.no_grad():
                np.savetxt(args.save, model(points).cpu().numpy(), delimiter=",")
            return
        if args.command == "minimize":
            result = minimize(
                load_model(args.init),
                linear_scalarization(import_callable(args.objective)),
                method=PreconditionedSGD(max_iter=args.max_iter, step_size=args.step_size),
                n_samples=args.n_samples,
                seed=args.seed,
            )
            save_model(args.save, result.model)
            output, success = (
                {"success": result.success, "status": result.status, "objective": result.objective},
                result.success,
            )
        else:
            output, success = _regression(args)
    except (ValueError, OSError, ImportError) as error:
        parser.error(str(error))
    print(json.dumps(output, allow_nan=True, default=lambda value: value.tolist()))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
