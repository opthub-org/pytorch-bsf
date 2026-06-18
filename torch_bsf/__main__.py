from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import mlflow
from mlflow import autolog

from torch_bsf import BezierSimplexDataModule
from torch_bsf._cli_args import add_common_training_args
from torch_bsf.bezier_simplex import BezierSimplex, load, randn
from torch_bsf.validator import index_list, validate_simplex_indices


def main() -> None:
    """Entry point for ``python -m torch_bsf``."""
    parser = ArgumentParser(
        prog="python -m torch_bsf", description="Bezier simplex fitting"
    )
    parser.add_argument(
        "--params", type=Path, required=True, metavar="CSV",
        help="Path to the input parameters CSV file",
    )
    parser.add_argument(
        "--values", type=Path, required=True, metavar="CSV",
        help="Path to the output values CSV file",
    )
    parser.add_argument(
        "--meshgrid", type=Path, metavar="CSV|DIR",
        help=(
            "Path to the meshgrid CSV file used for prediction output; if omitted or if an "
            "existing directory path is given, the parameters CSV specified by --params is used instead"
        ),
    )
    mutual_group = parser.add_mutually_exclusive_group(required=True)
    mutual_group.add_argument(
        "--degree", type=int, metavar="N",
        help="Degree of the Bezier simplex (required when --init is not given)",
    )
    mutual_group.add_argument(
        "--init", type=Path, metavar="PT",
        help="Path to a pretrained model file to initialize from (required when --degree is not given)",
    )
    parser.add_argument(
        "--freeze", type=index_list, metavar="INDICES",
        help=(
            "JSON-style list-of-lists of simplex indices of control points to freeze during training; "
            "each simplex index list must have length n_params and its elements must sum to the Bezier "
            "simplex degree (or the model's degree when using --init) "
            "(e.g. '[[2,0],[0,2]]' for n_params=2 and degree=2)"
        ),
    )
    parser.add_argument(
        "--header", type=int, default=0, metavar="N",
        help="Number of header rows to skip in CSV files (default: 0)",
    )
    parser.add_argument(
        "--normalize", type=str, choices=("none", "max", "std", "quantile"), default="none",
        metavar="{none,max,std,quantile}",
        help="Normalization applied to values before training (default: none)",
    )
    add_common_training_args(parser)
    parser.add_argument(
        "--loglevel", type=int, choices=(0, 1, 2), default=2, help="0: nothing, 1: metrics, 2: metrics & models"
    )

    args = parser.parse_args()

    meshgrid: Path = args.params if (args.meshgrid is None or args.meshgrid.is_dir()) else args.meshgrid

    autolog(
        log_input_examples=(args.loglevel >= 2),
        log_model_signatures=False,
        log_models=False,
        disable=(args.loglevel <= 0),
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=(args.loglevel <= 0),
    )

    dm = BezierSimplexDataModule(
        params=args.params,
        values=args.values,
        header=args.header,
        batch_size=args.batch_size,
        split_ratio=args.split_ratio,
        normalize=args.normalize,
    )

    if args.init:
        _loaded = load(args.init)
        _same_weight = getattr(_loaded, "smoothness_weight", None) == args.smoothness_weight
        _has_adjacency = hasattr(_loaded, "adjacency_indices_")
        if _same_weight and _has_adjacency:
            bs = _loaded
        else:
            bs = BezierSimplex(_loaded.control_points, smoothness_weight=args.smoothness_weight)
    else:
        bs = randn(
            n_params=dm.n_params,
            n_values=dm.n_values,
            degree=args.degree,
            smoothness_weight=args.smoothness_weight,
        )

    freeze: list[list[int]] = args.freeze or []
    validate_simplex_indices(freeze, bs.n_params, bs.degree)

    for index in freeze:
        bs.freeze_row(index)

    trainer = Trainer(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        precision=args.precision,
        num_nodes=args.num_nodes,
        max_epochs=args.max_epochs,
        enable_checkpointing=args.enable_checkpointing,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[EarlyStopping(monitor="val_mse")],
    )
    trainer.fit(bs, dm)

    if args.loglevel >= 2:
        from mlflow.models import ModelSignature
        from mlflow.types import Schema, TensorSpec
        import mlflow.pytorch

        # Use float64 in signature to match CSV/JSON input (float32/float16 models auto-convert
        # via torch.as_tensor in forward). Shape (-1, n) validates column count without dtype lock-in.
        signature = ModelSignature(
            inputs=Schema([TensorSpec(np.dtype("float64"), (-1, bs.n_params))]),
            outputs=Schema([TensorSpec(np.dtype("float64"), (-1, bs.n_values))]),
        )
        model_log_kwargs = {"signature": signature, "serialization_format": "pickle"}
        last_run = mlflow.last_active_run()
        if mlflow.active_run() is not None or last_run is None:
            mlflow.pytorch.log_model(bs, "model", **model_log_kwargs)
        else:
            with mlflow.start_run(run_id=last_run.info.run_id):
                mlflow.pytorch.log_model(bs, "model", **model_log_kwargs)

    # search for filename
    fn = f"{args.params.name},{args.values.name},meshgrid,d_{bs.degree},r_{args.split_ratio}.csv"

    ts = dm.load_data(meshgrid)
    xs = bs.forward(ts)
    xs = dm.inverse_transform(xs)

    # save meshgrid
    x = xs.to("cpu").detach().numpy()
    np.savetxt(fn, x, delimiter=",")
    print(f"Meshgrid saved: {fn}")


if __name__ == "__main__":
    main()
