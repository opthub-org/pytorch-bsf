import sys
import time
from argparse import ArgumentParser, Namespace
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.nn import functional as F

from torch_bsf._cli_args import (
    add_common_training_args,
    add_model_args,
    add_sgd_args,
    check_columns,
    check_sgd_args,
    file_tag,
    fit_options,
    import_callable,
    load_model_data,
    load_sgd_init,
    load_table,
)
from torch_bsf.simploidal import BezierSimploid, save
from torch_bsf.simploidal.fitting import fit
from torch_bsf.simploidal.sgd import fit_sgd, linear_scalarization, sample_domain

if TYPE_CHECKING:
    from mlflow.entities import Metric

# MLflow is an optional dependency (the extra ``pytorch-bsf[mlflow]``), imported only when the CLI logs to it.
_LOGLEVEL_HELP = "Logging to MLflow, which requires pytorch-bsf[mlflow]; 0: nothing, 1: metrics, 2: metrics & models"


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="python -m torch_bsf",
        description="Bezier simplex (or simploid) fitting",
        epilog=(
            "To fit a Bezier simplex to the Pareto set of objective functions instead of data, "
            "run 'python -m torch_bsf sgd' (see 'python -m torch_bsf sgd --help')."
        ),
    )
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
    parser.add_argument("--loglevel", type=int, choices=(0, 1, 2), default=2, help=_LOGLEVEL_HELP)
    return parser


def _check_loglevel(loglevel: int) -> int:
    """The level to log to MLflow at: ``loglevel``, or 0 with a warning if MLflow is not installed.

    Without MLflow, the commands still fit the model and save their files, but log nothing.
    """
    if loglevel <= 0:
        return loglevel
    try:
        import mlflow  # noqa: F401
    except ImportError:
        print(
            f"warning: MLflow is not installed, so nothing is logged (--loglevel={loglevel} is ignored); "
            'install it with: pip install "pytorch-bsf[mlflow]"',
            file=sys.stderr,
        )
        return 0
    return loglevel


def _autolog(loglevel: int) -> None:
    """Enables (or, at loglevel 0, disables) MLflow autologging of Lightning, if MLflow is installed."""
    try:
        from mlflow import autolog
    except ImportError:  # nothing to log to, as _check_loglevel has warned
        return
    autolog(
        log_input_examples=(loglevel >= 2),
        log_model_signatures=False,
        log_models=False,
        disable=(loglevel <= 0),
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=(loglevel <= 0),
    )


@contextmanager
def _mlflow_run(resume: bool = True) -> Iterator[None]:
    """Logs to the active run, else (if ``resume``) to the run autolog last finished, else to a run started on demand.

    ``resume`` must be false when no Lightning trainer ran in this invocation, since autolog did not start a run then.
    """
    import mlflow

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


def _log_to_mlflow(
    model: BezierSimploid,
    metrics: dict[str, float],
    params: dict[str, Any],
    save_path: Path,
    loglevel: int,
    resume: bool,
    history: Sequence["Metric"] = (),
) -> None:
    """Logs the parameters, the metrics, the control points file and (loglevel 2) the model to MLflow.

    ``resume`` is passed to :func:`_mlflow_run`.  ``history`` gives metrics with their steps, which
    :meth:`mlflow.MlflowClient.log_batch` splits into requests within the limits of MLflow.
    """
    import mlflow

    with _mlflow_run(resume=resume):
        mlflow.log_metrics(metrics)  # starts a run if none is active
        run = mlflow.active_run()
        if run is not None and history:
            mlflow.MlflowClient().log_batch(run.info.run_id, metrics=history)
        # An MLflow project run has already logged its own parameters, which cannot be changed.
        logged = mlflow.get_run(run.info.run_id).data.params if run is not None else {}
        mlflow.log_params({k: v for k, v in params.items() if k not in logged})
        if loglevel >= 2:
            mlflow.log_artifact(str(save_path))
            _log_model(model, model.n_params, model.n_values)


def _log_run(args: Namespace, model: BezierSimploid, metrics: dict[str, float], save_path: Path) -> None:
    """Logs the parameters, the errors, the control points file and (loglevel 2) the model to MLflow."""
    params = {
        "simploid_type": ",".join(map(str, model.simploid_type)),
        "degree": ",".join(map(str, model.degree)),
        "solver": args.solver,
        "ridge": args.ridge,
        "smoothness_weight": args.smoothness_weight,
        "normalize": args.normalize,
        "split_ratio": args.split_ratio,
    }
    trained = args.solver == "trainer"  # whether autolog started (and finished) a run
    _log_to_mlflow(model, metrics, params, save_path, args.loglevel, resume=trained)


def _build_sgd_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="python -m torch_bsf sgd",
        description=(
            "Bezier simplex fitting to the Pareto set of objective functions by the preconditioned stochastic "
            "gradient descent of Hikima et al. (AISTATS 2025), minimizing the linear scalarization t^T f(x) "
            "averaged over the weights t on the simplex"
        ),
    )
    add_sgd_args(parser)
    parser.add_argument(
        "--save",
        type=Path,
        metavar="FILE",
        help=(
            "Path to save the control points of the fitted model to; the format is chosen by the "
            "extension (.csv, .tsv, .json, .yml, .yaml or .pt) "
            "(default: '<module>.<function>,control_points,t_<M-1>,d_<N>.csv' named after --objective)"
        ),
    )
    parser.add_argument("--loglevel", type=int, choices=(0, 1, 2), default=2, help=_LOGLEVEL_HELP)
    return parser


def _load_objective(parser: ArgumentParser, spec: str) -> Callable[[torch.Tensor], torch.Tensor]:
    """Imports the objective functions given by ``--objective``, reporting an error if they cannot be imported."""
    try:
        return import_callable(spec)
    except ValueError as e:
        parser.error(f"cannot load --objective: {e}")


def _check_objective(
    parser: ArgumentParser, objective: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, n_objectives: int
) -> None:
    """Reports an error unless ``objective`` maps the points ``x`` to differentiable values of ``n_objectives`` columns."""
    try:
        fs = objective(x)
    except Exception as e:  # the objective is user code
        parser.error(f"--objective failed on points of shape {tuple(x.shape)}: {type(e).__name__}: {e}")
    expected = (len(x), n_objectives)
    if not isinstance(fs, torch.Tensor) or fs.shape != expected:
        got = f"shape {tuple(fs.shape)}" if isinstance(fs, torch.Tensor) else f"a {type(fs).__name__}"
        parser.error(
            f"--objective must map points of shape {tuple(x.shape)} (--n-variables) to a tensor of shape "
            f"{expected} (--n-objectives), but it returned {got}"
        )
    # The values must be connected to x in the autograd graph, not only require grad (e.g. through parameters of their own).
    (grad,) = torch.autograd.grad(fs.sum(), x, allow_unused=True) if fs.requires_grad else (None,)
    if grad is None:
        parser.error(
            "--objective must be differentiable with PyTorch, but its values do not depend on the points through autograd"
        )


def _sgd_params(args: Namespace) -> dict[str, Any]:
    """The parameters of ``python -m torch_bsf sgd`` logged to MLflow."""
    names = ("objective", "n_objectives", "n_variables", "degree", "n_samples", "batch_size", "max_iter", "step_size")
    options = ("step_decay", "preconditioner", "seed", "init", "freeze", "smoothness_weight", "log_every")
    return {name: getattr(args, name) for name in (*names, *options)}


def _loss_recorder(log_every: int) -> tuple[list["Metric"], Callable[[int, BezierSimploid, float], None]]:
    """A callback of :func:`fit_sgd` recording ``batch_loss`` at every ``log_every``-th iteration, and its records."""
    from mlflow.entities import Metric

    history: list[Metric] = []

    def record(k: int, model: BezierSimploid, batch_loss: float) -> None:  # reads only the loss, not the model
        if k % log_every == 0:
            history.append(Metric("batch_loss", batch_loss, int(time.time() * 1000), k))

    return history, record


def sgd_main(argv: Sequence[str] | None = None) -> None:
    """Entry point for ``python -m torch_bsf sgd``, which fits a Bézier simplex with :func:`torch_bsf.fit_sgd`.

    Parameters
    ----------
    argv
        The arguments after ``sgd``; ``sys.argv[2:]`` if ``None``.
    """
    parser = _build_sgd_parser()
    args = parser.parse_args(sys.argv[2:] if argv is None else argv)
    check_sgd_args(parser, args)
    args.loglevel = _check_loglevel(args.loglevel)
    objective = _load_objective(parser, args.objective)
    simploid_type = (args.n_objectives - 1,)
    ts = None
    if args.meshgrid is not None and not args.meshgrid.is_dir():
        ts = load_table(args.meshgrid, args.header)
        check_columns(parser, "--meshgrid", ts, simploid_type)

    init = load_sgd_init(parser, args)
    # The samples fit_sgd would draw itself, kept to evaluate the loss after fitting.
    (t,) = sample_domain(init, args.n_samples, seed=args.seed)
    _check_objective(parser, objective, init(t[:2]), args.n_objectives)
    loss = linear_scalarization(objective)
    names = ("batch_size", "max_iter", "step_size", "step_decay", "preconditioner", "freeze", "smoothness_weight")
    options = {name: getattr(args, name) for name in names}
    history, record = _loss_recorder(args.log_every) if args.loglevel >= 1 and args.log_every > 0 else ([], None)
    model = fit_sgd(init, loss, inputs=[t], seed=args.seed, callback=record, **options)  # the options are checked above

    tag = file_tag(model.simploid_type, model.degree)
    name = args.objective.replace(":", ".")
    save_path: Path = args.save or Path(f"{name},control_points,{tag}.csv")
    save(save_path, model)
    print(f"Control points saved: {save_path}")

    if args.loglevel >= 1:
        with torch.no_grad():
            metrics = {"train_loss": loss(model(t), t).mean().item()}
        _log_to_mlflow(model, metrics, _sgd_params(args), save_path, args.loglevel, resume=False, history=history)

    if ts is not None:
        fn = f"{name},meshgrid,{tag}.csv"
        with torch.no_grad():
            np.savetxt(fn, model(ts).cpu().numpy(), delimiter=",")
        print(f"Meshgrid saved: {fn}")


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for ``python -m torch_bsf``; ``python -m torch_bsf sgd ...`` runs :func:`sgd_main`.

    Parameters
    ----------
    argv
        The command-line arguments; ``sys.argv[1:]`` if ``None``.
    """
    args = sys.argv[1:] if argv is None else list(argv)
    if args[:1] == ["sgd"]:
        sgd_main(args[1:])
    else:
        _fit_main(args)


def _fit_main(argv: Sequence[str]) -> None:
    """Fits a model to the data given by the command-line arguments ``argv``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not 0.0 < args.split_ratio <= 1.0:
        parser.error(f"--split_ratio must satisfy 0 < R <= 1, but {args.split_ratio} is given")
    params, values, simploid_type, init = load_model_data(parser, args)
    if int(len(params) * args.split_ratio) == 0:
        parser.error(f"--split_ratio {args.split_ratio} leaves no training samples out of {len(params)}")
    meshgrid: Path = args.params if (args.meshgrid is None or args.meshgrid.is_dir()) else args.meshgrid
    ts = load_table(meshgrid, args.header)
    check_columns(parser, "--meshgrid", ts, simploid_type)

    args.loglevel = _check_loglevel(args.loglevel)
    _autolog(args.loglevel)

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
