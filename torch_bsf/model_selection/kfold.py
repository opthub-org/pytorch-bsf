from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path
from typing import cast

import numpy as np
import torch
from mlflow import autolog

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
from torch_bsf.simploidal.fitting import fit_kfold


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
    parser.add_argument(
        "--loglevel", type=int, choices=(0, 1, 2), default=2, help="0: nothing, 1: metrics, 2: metrics & models"
    )

    args = parser.parse_args()
    if args.split_ratio != 1.0:
        parser.error("--split_ratio is not supported: every fold holds out its own validation samples")
    params, values, simploid_type, init = load_model_data(parser, args)
    meshgrid: Path = args.params if (args.meshgrid is None or args.meshgrid.is_dir()) else args.meshgrid
    ts = load_table(meshgrid, args.header)
    check_columns(parser, "--meshgrid", ts, simploid_type)

    autolog(
        log_input_examples=(args.loglevel >= 2),
        log_model_signatures=False,
        log_models=False,
        disable=(args.loglevel <= 0),
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=(args.loglevel <= 0),
    )

    try:
        models = fit_kfold(
            params,
            values,
            args.degree,
            n_folds=args.num_folds,
            shuffle=args.shuffle,
            **fit_options(args, simploid_type, init),
        )
    except ValueError as e:
        parser.error(str(e))

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
