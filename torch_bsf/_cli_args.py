"""Shared argparse argument registration and data loading for CLI entry points."""

import importlib
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from torch_bsf.simploidal import BezierSimploid, load, simploid_n_params
from torch_bsf.validator import index_list, int_or_str


def add_common_training_args(parser: ArgumentParser) -> None:
    """Register training-related arguments that are shared across entry points."""
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=1.0,
        metavar="R",
        help=("Fraction of data used for training; the remainder becomes the validation set (default: 1.0)"),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        metavar="N",
        help="Mini-batch size for training; omit to use full-batch loading",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=2,
        metavar="N",
        help="Maximum number of training epochs (default: 2)",
    )
    parser.add_argument(
        "--smoothness_weight",
        type=float,
        default=0.0,
        metavar="W",
        help="Weight of the smoothness regularization term (default: 0.0)",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        metavar="TYPE",
        help=("Hardware accelerator for the Lightning trainer (e.g., 'cpu', 'gpu', 'tpu', 'mps', or 'auto'; default: auto)"),
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        metavar="NAME",
        help=(
            "Distributed training strategy for the Lightning trainer (e.g., 'ddp', "
            "'fsdp', 'deepspeed', or 'auto'; default: auto)"
        ),
    )
    parser.add_argument(
        "--devices",
        type=int_or_str,
        default="auto",
        metavar="N|auto",
        help=("Number of devices to use, or 'auto' to let the Lightning trainer decide (default: auto)"),
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        metavar="N",
        help="Number of compute nodes for distributed training (default: 1)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        metavar="PRECISION",
        help=(
            "Floating-point precision for the Lightning trainer: '32-true', '16-mixed', 'bf16-mixed', etc. (default: 32-true)"
        ),
    )
    parser.add_argument(
        "--enable_checkpointing",
        action="store_true",
        help="Enable Lightning model checkpointing during training (disabled by default)",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=1,
        metavar="N",
        help="Log training metrics every N optimizer steps (default: 1)",
    )


def simploid_type(text: str) -> tuple[int, ...]:
    """Parse a simploid type given as comma-separated positive integers.

    Parameters
    ----------
    text
        The type, e.g. ``"2,1"`` for a prism (triangle x segment).
        Surrounding parentheses or brackets are allowed (``"(2, 1)"``).

    Returns
    -------
    The simploid type as a tuple of positive integers.

    Raises
    ------
    ArgumentTypeError
        If ``text`` is not a non-empty list of positive integers.

    Examples
    --------
    >>> from torch_bsf._cli_args import simploid_type
    >>> simploid_type("2,1")
    (2, 1)
    >>> simploid_type("(1, 1, 1)")
    (1, 1, 1)
    """
    message = (
        f"invalid simploid type {text!r}: expected comma-separated positive integers "
        "(the simplex dimensions a_1,...,a_n of the factors), e.g. '2,1'"
    )
    try:
        parsed = tuple(int(part) for part in text.strip().strip("()[]").split(","))
    except ValueError:
        raise ArgumentTypeError(message) from None
    if any(a < 1 for a in parsed):
        raise ArgumentTypeError(message)
    return parsed


def _optional_simploid_type(text: str) -> tuple[int, ...] | None:
    """Parse a simploid type, or ``None`` for an empty string (the default of MLflow projects)."""
    return simploid_type(text) if text.strip() else None


def add_model_args(parser: ArgumentParser) -> None:
    """Register the data and model arguments shared by the fitting entry points."""
    parser.add_argument("--params", type=Path, required=True, metavar="CSV", help="Path to the input parameters CSV file")
    parser.add_argument("--values", type=Path, required=True, metavar="CSV", help="Path to the output values CSV file")
    parser.add_argument(
        "--meshgrid",
        type=Path,
        metavar="CSV|DIR",
        help=(
            "Path to the meshgrid CSV file used for prediction output; if omitted or if an "
            "existing directory path is given, the parameters CSV specified by --params is used instead"
        ),
    )
    parser.add_argument(
        "--header",
        type=int,
        default=0,
        metavar="N",
        help="Number of header rows to skip in CSV files (default: 0)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--degree",
        type=int,
        metavar="N",
        help="Degree of the Bezier simplex, or of every factor of the Bezier simploid (required when --init is not given)",
    )
    group.add_argument(
        "--init",
        type=Path,
        metavar="FILE",
        help=(
            "Path to a model file saved by torch_bsf.simploidal.save (e.g. by --save) to initialize from; "
            "its type and degree are used (required when --degree is not given)"
        ),
    )
    parser.add_argument(
        "--simploid-type",
        "--simploid_type",
        dest="simploid_type",
        type=_optional_simploid_type,
        metavar="A1,...,An",
        help=(
            "Fit a Bezier simploid, i.e. a tensor product of Bezier simplices of dimensions A1,...,An "
            "(e.g. '2,1' for a triangle x segment prism); the parameters file must then have sum(Ai + 1) columns, "
            "one block of barycentric coordinates per factor (default: the type of --init, "
            "else a Bezier simplex of dimension n_params - 1)"
        ),
    )
    parser.add_argument(
        "--freeze",
        type=index_list,
        metavar="INDICES",
        help=(
            "JSON-style list of the indices of control points to keep at their values in --init; "
            "an index of a Bezier simplex is a list of n_params integers summing to the degree "
            "(e.g. '[[2,0],[0,2]]' for n_params=2 and degree=2), and that of a Bezier simploid "
            "is a list of such lists, one per factor (e.g. '[[[2,0,0],[2,0]]]' for the type 2,1)"
        ),
    )
    parser.add_argument(
        "--normalize",
        type=str,
        choices=("none", "max", "std", "quantile"),
        default="none",
        metavar="{none,max,std,quantile}",
        help=(
            "Normalization applied to values while fitting; the fitted model predicts in the original scale (default: none)"
        ),
    )
    parser.add_argument(
        "--solver",
        choices=("lstsq", "trainer"),
        default="lstsq",
        metavar="{lstsq,trainer}",
        help=(
            "How to fit: 'lstsq' solves the least-squares problem exactly, "
            "'trainer' trains with the Lightning trainer options (default: lstsq)"
        ),
    )
    parser.add_argument(
        "--ridge",
        type=float,
        default=0.0,
        metavar="W",
        help="Weight of the ridge (Tikhonov) penalty for --solver=lstsq (default: 0.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        metavar="N",
        help="Random seed for splitting the data and for training (default: none)",
    )


def load_table(path: Path, header: int) -> torch.Tensor:
    """Reads a data file: comma-separated if its suffix is ``.csv``, whitespace-separated otherwise."""
    data = np.loadtxt(path, delimiter="," if path.suffix.lower() == ".csv" else None, skiprows=header, ndmin=2)
    return torch.from_numpy(data).to(torch.get_default_dtype())


def load_model_data(
    parser: ArgumentParser, args: Namespace
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], BezierSimploid | None]:
    """Reads the data and the initial model, and resolves the simploid type, reporting errors with ``parser``.

    Returns
    -------
    ``(params, values, simploid_type, init)``.
    """
    _check_fit_args(parser, args)
    params = load_table(args.params, args.header)
    values = load_table(args.values, args.header)
    if params.shape[0] != values.shape[0]:
        parser.error(f"--params has {params.shape[0]} rows, but --values has {values.shape[0]} rows")
    init = _load_init(parser, args.init)
    st = _resolve_simploid_type(parser, args.simploid_type, init, params)
    check_columns(parser, "--params", params, st)
    return params, values, st, init


def _check_fit_args(parser: ArgumentParser, args: Namespace) -> None:
    """Reports an error if the fitting options ``--degree``, ``--ridge``, ``--solver`` and ``--freeze`` are inconsistent."""
    if args.degree is not None and args.degree < 0:
        parser.error(f"--degree must be non-negative, but {args.degree} is given")
    if args.ridge < 0:
        parser.error(f"--ridge must be non-negative, but {args.ridge} is given")
    if args.ridge and args.solver != "lstsq":
        parser.error("--ridge is supported only with --solver=lstsq")
    if args.freeze is not None and args.init is None:
        parser.error("--freeze requires --init")


def _load_init(parser: ArgumentParser, path: Path | None) -> BezierSimploid | None:
    """Reads the initial model from ``path`` if it is given, reporting an error if it cannot be read."""
    if path is None:
        return None
    try:
        return load(path)
    except (OSError, ValueError) as e:
        parser.error(f"cannot load --init: {e}")


def _resolve_simploid_type(
    parser: ArgumentParser, simploid_type: tuple[int, ...] | None, init: BezierSimploid | None, params: torch.Tensor
) -> tuple[int, ...]:
    """The simploid type given by ``--simploid-type``, by the initial model, or of the simplex of the columns of ``params``."""
    if simploid_type is not None:
        return simploid_type
    if init is not None:
        return init.simploid_type
    if params.shape[1] >= 2:
        return (params.shape[1] - 1,)
    parser.error(f"--params has {params.shape[1]} column; a Bezier simplex needs at least 2")


def check_columns(parser: ArgumentParser, option: str, data: torch.Tensor, st: tuple[int, ...]) -> None:
    """Reports an error if ``data`` does not have the columns of the parameters of a simploid of type ``st``."""
    n_params = simploid_n_params(st)
    if data.shape[1] != n_params:
        type_str = ",".join(map(str, st))
        parser.error(
            f"{option} has {data.shape[1]} columns, but the type {type_str} requires sum(Ai + 1) = {n_params} columns"
        )


def file_tag(simploid_type: tuple[int, ...], degree: tuple[int, ...]) -> str:
    """The part ``t_<A1-...-An>,d_<N>`` of output file names; the degree is a single number if all factors share it.

    Examples
    --------
    >>> from torch_bsf._cli_args import file_tag
    >>> file_tag((2,), (3,)), file_tag((2, 1), (2, 2)), file_tag((1, 1), (2, 1))
    ('t_2,d_3', 't_2-1,d_2', 't_1-1,d_2-1')
    """
    degrees = degree[:1] if len(set(degree)) == 1 else degree
    return f"t_{'-'.join(map(str, simploid_type))},d_{'-'.join(map(str, degrees))}"


def trainer_kwargs(args: Namespace) -> dict[str, Any]:
    """The arguments for :class:`lightning.pytorch.Trainer` registered by :func:`add_common_training_args`."""
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


def fit_options(args: Namespace, simploid_type: tuple[int, ...], init: BezierSimploid | None) -> dict[str, Any]:
    """The keyword arguments for :func:`torch_bsf.fit` given by the command line.

    The trainer options apply to ``--solver=trainer`` only, and are ignored otherwise.
    """
    options: dict[str, Any] = dict(
        simploid_type=simploid_type,
        init=init,
        freeze=args.freeze,
        smoothness_weight=args.smoothness_weight,
        normalize=args.normalize,
        solver=args.solver,
        ridge=args.ridge,
        seed=args.seed,
    )
    if args.solver == "trainer":
        options.update(batch_size=args.batch_size, **trainer_kwargs(args))
    return options


def _optional_int(text: str) -> int | None:
    """Parse an integer, or ``None`` for an empty string (the default of MLflow projects)."""
    return int(text) if text.strip() else None


def _optional_path(text: str) -> Path | None:
    """Parse a path, or ``None`` for an empty string (the default of MLflow projects)."""
    return Path(text) if text.strip() else None


def _optional_index_list(text: str) -> list[list[int]] | None:
    """Parse a list of indices as :func:`~torch_bsf.validator.index_list`, or ``None`` for an empty string."""
    return index_list(text) if text.strip() else None


def add_sgd_args(parser: ArgumentParser) -> None:
    """Register the arguments of ``python -m torch_bsf sgd``, which fits a Bézier simplex with :func:`torch_bsf.fit_sgd`."""
    parser.add_argument(
        "--objective",
        required=True,
        metavar="MODULE:FUNCTION",
        help=(
            "Objective functions to minimize: a differentiable PyTorch function mapping points of shape (batch, L) "
            "to objective values of shape (batch, M), e.g. 'mypkg.problems:objective'; the module is imported from "
            "the current directory or the installed packages"
        ),
    )
    parser.add_argument(
        "--n-objectives",
        "--n_objectives",
        dest="n_objectives",
        type=int,
        required=True,
        metavar="M",
        help="Number M >= 2 of objectives; a Bezier simplex of dimension M - 1 is fitted",
    )
    parser.add_argument(
        "--n-variables",
        "--n_variables",
        dest="n_variables",
        type=int,
        required=True,
        metavar="L",
        help="Number L >= 1 of variables of the objectives, i.e. the dimension of the space the Bezier simplex lies in",
    )
    parser.add_argument("--degree", type=int, required=True, metavar="N", help="Degree of the Bezier simplex")
    parser.add_argument(
        "--meshgrid",
        type=Path,
        metavar="CSV|DIR",
        help=(
            "Path to a CSV file of parameters (M columns) at which to evaluate the fitted model; "
            "nothing is evaluated if omitted or if an existing directory path is given"
        ),
    )
    parser.add_argument(
        "--header",
        type=int,
        default=0,
        metavar="N",
        help="Number of header rows to skip in the meshgrid CSV file (default: 0)",
    )
    parser.add_argument(
        "--n-samples",
        "--n_samples",
        dest="n_samples",
        type=int,
        default=100,
        metavar="N",
        help="Number of parameters sampled uniformly on the simplex to average the loss over (default: 100)",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=_optional_int,
        metavar="N",
        help="Minibatch size; omit to use the full batch",
    )
    parser.add_argument(
        "--max-iter",
        "--max_iter",
        dest="max_iter",
        type=int,
        default=100,
        metavar="N",
        help="Number of iterations (default: 100)",
    )
    parser.add_argument(
        "--step-size",
        "--step_size",
        dest="step_size",
        type=float,
        default=0.05,
        metavar="ALPHA",
        help="Step size (default: 0.05)",
    )
    parser.add_argument(
        "--step-decay",
        "--step_decay",
        dest="step_decay",
        choices=("constant", "sqrt"),
        default="constant",
        help="'constant' keeps the step size, 'sqrt' divides it by sqrt(k + 1) at iteration k (default: constant)",
    )
    parser.add_argument(
        "--preconditioner",
        choices=("gram", "none"),
        default="gram",
        help=(
            "'gram' preconditions the gradient by the inverse Gram matrix of the Bernstein basis (Hikima et al., 2025); "
            "'none' is plain stochastic gradient descent (default: gram)"
        ),
    )
    parser.add_argument(
        "--seed", type=_optional_int, metavar="N", help="Random seed for the samples and the minibatches (default: none)"
    )
    parser.add_argument(
        "--init",
        type=_optional_path,
        metavar="FILE",
        help=(
            "Path to a model file saved by torch_bsf.simploidal.save (e.g. by --save) to start from, which must be "
            "a Bezier simplex of dimension M - 1, degree N and L values (default: the zero control points)"
        ),
    )
    parser.add_argument(
        "--freeze",
        type=_optional_index_list,
        metavar="INDICES",
        help=(
            "JSON-style list of the multi-indices of control points to keep at their values in --init, "
            "each a list of M integers summing to the degree (e.g. '[[2,0,0],[0,2,0],[0,0,2]]' for the vertices "
            "of a quadratic Bezier triangle)"
        ),
    )
    parser.add_argument(
        "--smoothness-weight",
        "--smoothness_weight",
        dest="smoothness_weight",
        type=float,
        default=0.0,
        metavar="W",
        help="Weight of the smoothness penalty on adjacent control points added to the loss (default: 0.0)",
    )
    parser.add_argument(
        "--log-every",
        "--log_every",
        dest="log_every",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Log the loss of the minibatch 'batch_loss' to MLflow at every N-th iteration, "
            "with the iteration as the step; 0 logs only the final loss (default: 1)"
        ),
    )


def check_sgd_args(parser: ArgumentParser, args: Namespace) -> None:
    """Reports an error if an argument registered by :func:`add_sgd_args` is out of range."""
    minimums = {
        "--n-objectives": (args.n_objectives, 2),
        "--n-variables": (args.n_variables, 1),
        "--degree": (args.degree, 0),
        "--n-samples": (args.n_samples, 1),
        "--batch-size": (1 if args.batch_size is None else args.batch_size, 1),
        "--max-iter": (args.max_iter, 0),
        "--log-every": (args.log_every, 0),
    }
    for option, (value, minimum) in minimums.items():
        if value < minimum:
            parser.error(f"{option} must be at least {minimum}, but {value} is given")
    if not args.step_size > 0:
        parser.error(f"--step-size must be positive, but {args.step_size} is given")
    if not args.smoothness_weight >= 0:
        parser.error(f"--smoothness-weight must be non-negative, but {args.smoothness_weight} is given")
    if args.freeze is not None and args.init is None:
        parser.error("--freeze requires --init")


def load_sgd_init(parser: ArgumentParser, args: Namespace) -> BezierSimploid:
    """The initial model of ``python -m torch_bsf sgd``: ``--init`` if given, else zeros.

    Reports an error with ``parser`` if ``--init`` cannot be read or does not match ``--n-objectives``,
    ``--n-variables`` and ``--degree``, or if ``--freeze`` has an invalid index.
    """
    simploid_type, degree = (args.n_objectives - 1,), (args.degree,)
    init = _load_init(parser, args.init)
    if init is None:
        return BezierSimploid.zeros(simploid_type, degree, n_values=args.n_variables)
    if init.simploid_type != simploid_type:
        type_str = ",".join(map(str, init.simploid_type))
        parser.error(
            f"--init is a Bezier simploid of type {type_str}, but --n-objectives={args.n_objectives} requires "
            f"a Bezier simplex of type {simploid_type[0]}"
        )
    if init.degree != degree:
        parser.error(f"--init has degree {init.degree[0]}, but --degree={args.degree} is given")
    if init.n_values != args.n_variables:
        parser.error(f"--init has {init.n_values} values, but --n-variables={args.n_variables} is given")
    _check_freeze(parser, init, args.freeze)
    return init


def _check_freeze(parser: ArgumentParser, model: BezierSimploid, freeze: Any) -> None:
    """Reports an error unless ``freeze`` is ``None`` or a list of control point indices of ``model``."""
    if freeze is None:
        return
    if not isinstance(freeze, list):
        parser.error(f"--freeze must be a list of indices, but {freeze!r} is given")
    for index in freeze:
        try:
            model.row(index)
        except ValueError:
            parser.error(
                f"--freeze has {index!r}, which is not a control point index of a Bezier simplex of type "
                f"{model.simploid_type[0]} and degree {model.degree[0]}"
            )


def import_callable(spec: str) -> Callable[..., Any]:
    """Import the callable given as ``module:name``, where ``name`` may be a dotted path of attributes.

    Raises
    ------
    ValueError
        If ``spec`` is malformed, the module cannot be imported, it has no such attribute, or it is not callable.

    Examples
    --------
    >>> from torch_bsf._cli_args import import_callable
    >>> import_callable("torch_bsf.simploidal:BezierSimploid.zeros")((1,), 1, n_values=1).degree
    (1,)
    """
    module_name, _, name = spec.partition(":")
    if not module_name or not name:
        raise ValueError(f"expected MODULE:FUNCTION, e.g. 'mypkg.problems:objective', but {spec!r} is given")
    try:
        obj = importlib.import_module(module_name)
    except Exception as e:  # importing a module runs arbitrary code
        raise ValueError(f"cannot import module {module_name!r}: {type(e).__name__}: {e}") from e
    for attr in name.split("."):
        if not hasattr(obj, attr):
            raise ValueError(f"module {module_name!r} has no attribute {name!r}")
        obj = getattr(obj, attr)
    if not callable(obj):
        raise ValueError(f"{spec!r} is not callable")
    return cast(Callable[..., Any], obj)
