from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.nn import functional as F

from torch_bsf.__main__ import _LOGLEVEL_HELP, _autolog, _check_loglevel, _mlflow_run
from torch_bsf._cli_args import (
    add_common_training_args,
    add_model_args,
    check_columns,
    file_tag,
    fit_options,
    load_model_data,
    load_table,
)
from torch_bsf.simploidal import BezierSimploid
from torch_bsf.simploidal.fitting import _fit_folds, _fold_indices


def _held_out_stats(
    models: torch.nn.ModuleList, folds: list[torch.Tensor], params: torch.Tensor, values: torch.Tensor
) -> list[dict[str, float]]:
    """The errors of each model on the fold it was not fitted to."""
    stats = []
    with torch.no_grad():
        for model, test in zip(models, folds, strict=True):
            predicted = model(params[test])
            expected = values[test].to(predicted)
            stats.append({"val_mse": F.mse_loss(predicted, expected).item(), "val_mae": F.l1_loss(predicted, expected).item()})
    return stats


def main() -> None:
    """Entry point for ``python -m torch_bsf.model_selection.kfold``."""
    parser = ArgumentParser(
        prog="python -m torch_bsf.model_selection.kfold",
        description="Bezier simplex (or simploid) fitting with k-fold cross validation",
    )
    add_model_args(parser)
    parser.add_argument(
        "--num_folds",
        type=int,
        default=5,
        metavar="K",
        help="Number of folds for k-fold cross-validation (default: 5)",
    )
    parser.add_argument(
        "--shuffle",
        action=BooleanOptionalAction,
        default=True,
        help="Shuffle the dataset before splitting into folds (default: True); use --no-shuffle to disable",
    )
    add_common_training_args(parser)
    parser.add_argument("--loglevel", type=int, choices=(0, 1, 2), default=2, help=_LOGLEVEL_HELP)

    args = parser.parse_args()
    if args.split_ratio != 1.0:
        parser.error("--split_ratio is not supported: every fold holds out its own validation samples")
    params, values, simploid_type, init = load_model_data(parser, args)
    meshgrid: Path = args.params if (args.meshgrid is None or args.meshgrid.is_dir()) else args.meshgrid
    ts = load_table(meshgrid, args.header)
    check_columns(parser, "--meshgrid", ts, simploid_type)

    args.loglevel = _check_loglevel(args.loglevel)
    _autolog(args.loglevel)

    try:
        folds = _fold_indices(len(params), args.num_folds, args.shuffle, args.seed)
        models = _fit_folds(params, values, args.degree, folds, **fit_options(args, simploid_type, init))
    except ValueError as e:
        parser.error(str(e))

    stats = _held_out_stats(models, folds, params, values)
    for k, fold_stats in enumerate(stats):
        print(f"Fold {k}: " + ", ".join(f"{name}={value:.6g}" for name, value in fold_stats.items()))
    means = {f"mean_{name}": sum(s[name] for s in stats) / len(stats) for name in stats[0]}
    print("Mean: " + ", ".join(f"{name}={value:.6g}" for name, value in means.items()))
    if args.loglevel >= 1:
        import mlflow

        with _mlflow_run(resume=args.solver == "trainer"):
            for k, fold_stats in enumerate(stats):
                mlflow.log_metrics(fold_stats, step=k)
            mlflow.log_metrics(means)

    degree: tuple[int, ...] = cast(BezierSimploid, models[0]).degree
    tag = file_tag(simploid_type, degree)
    prefix = f"{args.params.name},{args.values.name},{len(models)}fold,meshgrid,{tag}"
    with torch.no_grad():
        xs = torch.stack([model(ts) for model in models])
    for k, x in enumerate(xs):
        np.savetxt(f"{prefix},{k}.csv", x.cpu().numpy(), delimiter=",")
    fn = f"{prefix}.csv"
    np.savetxt(fn, xs.mean(dim=0).cpu().numpy(), delimiter=",")  # mean over folds
    print(f"Meshgrid saved: {fn}")


if __name__ == "__main__":
    main()
