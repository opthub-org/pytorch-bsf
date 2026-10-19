from argparse import ArgumentParser, Namespace
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow
import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from mlflow import autolog
from torch.nn import functional as F

from torch_bsf import BezierSimplexDataModule
from torch_bsf._cli_args import add_common_training_args, add_simploid_args
from torch_bsf.bezier_simplex import BezierSimplex, load, randn
from torch_bsf.simploidal import BezierSimploid, fit_simploid
from torch_bsf.simploidal import save as save_simploid
from torch_bsf.simploidal import simploid_n_params
from torch_bsf.validator import index_list, validate_simplex_indices


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="python -m torch_bsf", description="Bezier simplex (or simploid) fitting")
    parser.add_argument(
        "--params",
        type=Path,
        required=True,
        metavar="CSV",
        help="Path to the input parameters CSV file",
    )
    parser.add_argument(
        "--values",
        type=Path,
        required=True,
        metavar="CSV",
        help="Path to the output values CSV file",
    )
    parser.add_argument(
        "--meshgrid",
        type=Path,
        metavar="CSV|DIR",
        help=(
            "Path to the meshgrid CSV file used for prediction output; if omitted or if an "
            "existing directory path is given, the parameters CSV specified by --params is used instead"
        ),
    )
    mutual_group = parser.add_mutually_exclusive_group(required=True)
    mutual_group.add_argument(
        "--degree",
        type=int,
        metavar="N",
        help=(
            "Degree of the Bezier simplex, or of every factor of the Bezier simploid with --simploid-type "
            "(required when --init is not given)"
        ),
    )
    mutual_group.add_argument(
        "--init",
        type=Path,
        metavar="PT",
        help="Path to a pretrained model file to initialize from (required when --degree is not given)",
    )
    parser.add_argument(
        "--freeze",
        type=index_list,
        metavar="INDICES",
        help=(
            "JSON-style list-of-lists of simplex indices of control points to freeze during training; "
            "each simplex index list must have length n_params and its elements must sum to the Bezier "
            "simplex degree (or the model's degree when using --init) "
            "(e.g. '[[2,0],[0,2]]' for n_params=2 and degree=2)"
        ),
    )
    parser.add_argument(
        "--header",
        type=int,
        default=0,
        metavar="N",
        help="Number of header rows to skip in CSV files (default: 0)",
    )
    parser.add_argument(
        "--normalize",
        type=str,
        choices=("none", "max", "std", "quantile"),
        default="none",
        metavar="{none,max,std,quantile}",
        help="Normalization applied to values before training (default: none)",
    )
    add_common_training_args(parser)
    add_simploid_args(parser)
    parser.add_argument(
        "--save",
        type=Path,
        metavar="FILE",
        help=(
            "Path to save the control points of the fitted Bezier simploid to; the format is chosen by the "
            "extension (.csv, .tsv, .json, .yml, .yaml or .pt) "
            "(default: '<params>,<values>,control_points,t_<A1-...-An>,d_<N>.csv'; requires --simploid-type)"
        ),
    )
    parser.add_argument(
        "--loglevel", type=int, choices=(0, 1, 2), default=2, help="0: nothing, 1: metrics, 2: metrics & models"
    )
    return parser


def _check_simploid_args(parser: ArgumentParser, args: Namespace) -> None:
    """Validates the options that depend on whether ``--simploid-type`` is given."""
    if args.simploid_type is None:
        given = [f"--{name}" for name in ("solver", "ridge", "save") if getattr(args, name) is not None]
        if given:
            parser.error(f"{', '.join(given)} requires --simploid-type")
        return
    unsupported = {
        "--init": args.init is not None,
        "--freeze": args.freeze is not None,
        "--smoothness_weight": args.smoothness_weight != 0.0,
        "--split_ratio": args.split_ratio != 1.0,
    }
    for name, given in unsupported.items():
        if given:
            parser.error(f"{name} is not supported with --simploid-type (a Bezier simploid is fitted on all data)")
    if args.degree < 0:
        parser.error(f"--degree must be non-negative, but {args.degree} is given")
    args.solver = args.solver or "lstsq"
    args.ridge = 0.0 if args.ridge is None else args.ridge
    if args.ridge < 0:
        parser.error(f"--ridge must be non-negative, but {args.ridge} is given")
    if args.ridge and args.solver != "lstsq":
        parser.error("--ridge is supported only with --solver=lstsq")


def _trainer_kwargs(args: Namespace) -> dict[str, Any]:
    return dict(
        accelerator=args.accelerator,
        strategy=args.strategy,
        devices=args.devices,
        precision=args.precision,
        num_nodes=args.num_nodes,
        max_epochs=args.max_epochs,
        enable_checkpointing=args.enable_checkpointing,
        log_every_n_steps=args.log_every_n_steps,
    )


@contextmanager
def _mlflow_run(resume: bool = True) -> Iterator[None]:
    """Logs to the active run, else (if ``resume``) to the run autolog last finished, else to a run started on demand.

    ``resume`` must be false when no Lightning trainer ran in this invocation, since autolog did not start a run then.
    """
    last_run = mlflow.last_active_run() if resume else None
    if mlflow.active_run() is not None or last_run is None:
        yield
    else:
        with mlflow.start_run(run_id=last_run.info.run_id):
            yield


def _log_model(model: torch.nn.Module, n_params: int, n_values: int) -> None:
    import mlflow.pytorch
    from mlflow.models import ModelSignature
    from mlflow.types import Schema, TensorSpec

    # Use float64 in signature to match CSV/JSON input (float32/float16 models auto-convert
    # via torch.as_tensor in forward). Shape (-1, n) validates column count without dtype lock-in.
    signature = ModelSignature(
        inputs=Schema([TensorSpec(np.dtype("float64"), (-1, n_params))]),
        outputs=Schema([TensorSpec(np.dtype("float64"), (-1, n_values))]),
    )
    with _mlflow_run():
        mlflow.pytorch.log_model(model, "model", signature=signature, serialization_format="pickle")


def _fit_simplex(args: Namespace, dm: BezierSimplexDataModule) -> BezierSimplex:
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

    trainer = Trainer(**_trainer_kwargs(args), callbacks=[EarlyStopping(monitor="val_mse")])
    trainer.fit(bs, dm)
    return bs


def _run_simplex(args: Namespace, dm: BezierSimplexDataModule, meshgrid: Path) -> None:
    bs = _fit_simplex(args, dm)

    if args.loglevel >= 2:
        _log_model(bs, bs.n_params, bs.n_values)

    # search for filename
    fn = f"{args.params.name},{args.values.name},meshgrid,d_{bs.degree},r_{args.split_ratio}.csv"

    ts = dm.load_data(meshgrid)
    xs = bs.forward(ts)
    xs = dm.inverse_transform(xs)

    # save meshgrid
    x = xs.to("cpu").detach().numpy()
    np.savetxt(fn, x, delimiter=",")
    print(f"Meshgrid saved: {fn}")


def _check_columns(parser: ArgumentParser, option: str, data: torch.Tensor | np.ndarray, args: Namespace) -> None:
    n_params = simploid_n_params(args.simploid_type)
    if data.shape[1] != n_params:
        type_str = ",".join(map(str, args.simploid_type))
        parser.error(
            f"{option} has {data.shape[1]} columns, but --simploid-type={type_str} requires sum(Ai + 1) = {n_params} columns"
        )


def _load_table(path: Path, header: int) -> np.ndarray:
    """Reads a data file as :meth:`BezierSimplexDataModule.load_data` does."""
    return np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None, skiprows=header, ndmin=2)


def _check_simploid_data(parser: ArgumentParser, args: Namespace) -> None:
    """Checks the shapes of the data files before :class:`BezierSimplexDataModule` reads them.

    The data module pairs the rows of both files on construction, which fails with an unhelpful error
    (an ``AssertionError`` in some PyTorch versions) if their row counts differ.
    """
    params = _load_table(args.params, args.header)
    values = _load_table(args.values, args.header)
    _check_columns(parser, "--params", params, args)
    if params.shape[0] != values.shape[0]:
        parser.error(f"--params has {params.shape[0]} rows, but --values has {values.shape[0]} rows")


def _fit_simploid(args: Namespace, dm: BezierSimplexDataModule) -> BezierSimploid:
    params = dm.load_params()
    values = dm.fit_transform(dm.load_values())
    kwargs = _trainer_kwargs(args) if args.solver == "trainer" else {}
    model = fit_simploid(
        params,
        values,
        args.simploid_type,
        args.degree,
        solver=args.solver,
        ridge=args.ridge,
        batch_size=args.batch_size,
        **kwargs,
    )
    # Undo the normalization on the control points: the scalers are affine in each column and the
    # Bernstein basis is a partition of unity, so the resulting model predicts values in the original scale.
    control_points = dm.inverse_transform(model.control_points.detach())
    return BezierSimploid(model.simploid_type, model.degree, control_points)


def _log_simploid(args: Namespace, dm: BezierSimplexDataModule, model: BezierSimploid, save_path: Path) -> None:
    """Logs the parameters, the training error, the control points file and (loglevel 2) the model to MLflow."""
    with torch.no_grad():
        mse = F.mse_loss(model(dm.load_params()), dm.load_values().to(model.control_points.dtype)).item()
    trained = args.solver == "trainer"  # whether autolog started (and finished) a run
    with _mlflow_run(resume=trained):
        mlflow.log_metric("train_mse", mse)  # starts a run if none is active
        run = mlflow.active_run()
        # An MLflow project run has already logged its own parameters, which cannot be changed.
        logged = mlflow.get_run(run.info.run_id).data.params if run is not None else {}
        params = {
            "simploid_type": ",".join(map(str, model.simploid_type)),
            "degree": args.degree,
            "solver": args.solver,
            "ridge": args.ridge,
        }
        mlflow.log_params({k: v for k, v in params.items() if k not in logged})
        if args.loglevel >= 2:
            mlflow.log_artifact(str(save_path))
            _log_model(model, model.n_params, model.n_values)


def _run_simploid(parser: ArgumentParser, args: Namespace, dm: BezierSimplexDataModule, meshgrid: Path) -> None:
    model = _fit_simploid(args, dm)
    ts = dm.load_data(meshgrid)
    _check_columns(parser, "--meshgrid", ts, args)

    tag = f"t_{'-'.join(map(str, model.simploid_type))},d_{args.degree}"
    save_path: Path = args.save or Path(f"{args.params.name},{args.values.name},control_points,{tag}.csv")
    save_simploid(save_path, model)
    print(f"Control points saved: {save_path}")

    if args.loglevel >= 1:
        _log_simploid(args, dm, model, save_path)

    fn = f"{args.params.name},{args.values.name},meshgrid,{tag}.csv"
    np.savetxt(fn, model(ts).to("cpu").detach().numpy(), delimiter=",")
    print(f"Meshgrid saved: {fn}")


def main() -> None:
    """Entry point for ``python -m torch_bsf``."""
    parser = _build_parser()
    args = parser.parse_args()
    _check_simploid_args(parser, args)
    if args.simploid_type is not None:
        _check_simploid_data(parser, args)

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

    if args.simploid_type is None:
        _run_simplex(args, dm, meshgrid)
    else:
        _run_simploid(parser, args, dm, meshgrid)


if __name__ == "__main__":
    main()
