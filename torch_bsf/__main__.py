from argparse import ArgumentParser, Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mlflow
import numpy as np
import torch
from mlflow import autolog
from torch.nn import functional as F

from torch_bsf._cli_args import (
    add_common_training_args,
    add_model_args,
    check_columns,
    file_tag,
    fit_options,
    load_model_data,
    load_table,
)
from torch_bsf.simploidal import BezierSimploid, save
from torch_bsf.simploidal.fitting import fit


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(prog="python -m torch_bsf", description="Bezier simplex (or simploid) fitting")
    add_model_args(parser)
    add_common_training_args(parser)
    parser.add_argument(
        "--save",
        type=Path,
        metavar="FILE",
        help=(
            "Path to save the control points of the fitted model to; the format is chosen by the "
            "extension (.csv, .tsv, .json, .yml, .yaml or .pt) "
            "(default: '<params>,<values>,control_points,t_<A1-...-An>,d_<N>.csv', "
            "where N lists the degree of every factor, e.g. '2-1', if they differ)"
        ),
    )
    parser.add_argument(
        "--loglevel", type=int, choices=(0, 1, 2), default=2, help="0: nothing, 1: metrics, 2: metrics & models"
    )
    return parser


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


def _split(
    params: torch.Tensor, values: torch.Tensor, split_ratio: float, seed: int | None
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor] | None]:
    """Splits the samples at random into the training set and the validation set (``None`` if empty)."""
    n_train = int(len(params) * split_ratio)
    if n_train == len(params):
        return (params, values), None
    generator = torch.Generator()
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(seed)
    order = torch.randperm(len(params), generator=generator)
    train, val = order[:n_train], order[n_train:]
    return (params[train], values[train]), (params[val], values[val])


def _metrics(model: BezierSimploid, prefix: str, params: torch.Tensor, values: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        predicted = model(params)
        values = values.to(predicted)
        return {
            f"{prefix}_mse": F.mse_loss(predicted, values).item(),
            f"{prefix}_mae": F.l1_loss(predicted, values).item(),
        }


def _log_run(args: Namespace, model: BezierSimploid, metrics: dict[str, float], save_path: Path) -> None:
    """Logs the parameters, the errors, the control points file and (loglevel 2) the model to MLflow."""
    trained = args.solver == "trainer"  # whether autolog started (and finished) a run
    with _mlflow_run(resume=trained):
        mlflow.log_metrics(metrics)  # starts a run if none is active
        run = mlflow.active_run()
        # An MLflow project run has already logged its own parameters, which cannot be changed.
        logged = mlflow.get_run(run.info.run_id).data.params if run is not None else {}
        params = {
            "simploid_type": ",".join(map(str, model.simploid_type)),
            "degree": ",".join(map(str, model.degree)),
            "solver": args.solver,
            "ridge": args.ridge,
            "smoothness_weight": args.smoothness_weight,
            "normalize": args.normalize,
            "split_ratio": args.split_ratio,
        }
        mlflow.log_params({k: v for k, v in params.items() if k not in logged})
        if args.loglevel >= 2:
            mlflow.log_artifact(str(save_path))
            _log_model(model, model.n_params, model.n_values)


def main() -> None:
    """Entry point for ``python -m torch_bsf``."""
    parser = _build_parser()
    args = parser.parse_args()
    if not 0.0 < args.split_ratio <= 1.0:
        parser.error(f"--split_ratio must satisfy 0 < R <= 1, but {args.split_ratio} is given")
    params, values, simploid_type, init = load_model_data(parser, args)
    if int(len(params) * args.split_ratio) == 0:
        parser.error(f"--split_ratio {args.split_ratio} leaves no training samples out of {len(params)}")
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

    (train_params, train_values), val = _split(params, values, args.split_ratio, args.seed)
    try:
        model = fit(train_params, train_values, args.degree, **fit_options(args, simploid_type, init))
    except ValueError as e:
        parser.error(str(e))

    tag = file_tag(model.simploid_type, model.degree)
    save_path: Path = args.save or Path(f"{args.params.name},{args.values.name},control_points,{tag}.csv")
    save(save_path, model)
    print(f"Control points saved: {save_path}")

    if args.loglevel >= 1:
        metrics = _metrics(model, "train", train_params, train_values)
        if val is not None:
            metrics.update(_metrics(model, "val", *val))
        _log_run(args, model, metrics, save_path)

    fn = f"{args.params.name},{args.values.name},meshgrid,{tag}.csv"
    with torch.no_grad():
        np.savetxt(fn, model(ts).cpu().numpy(), delimiter=",")
    print(f"Meshgrid saved: {fn}")


if __name__ == "__main__":
    main()
