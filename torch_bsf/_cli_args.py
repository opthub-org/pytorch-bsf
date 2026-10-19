"""Shared argparse argument registration for CLI entry points."""

from argparse import ArgumentParser, ArgumentTypeError

from torch_bsf.validator import int_or_str


def add_common_training_args(parser: ArgumentParser) -> None:
    """Register training-related arguments that are shared across entry points."""
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=1.0,
        metavar="R",
        help=("Fraction of data used for training; the remainder becomes the validation " "set (default: 1.0)"),
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
        help=(
            "Hardware accelerator for the Lightning trainer (e.g., 'cpu', 'gpu', " "'tpu', 'mps', or 'auto'; default: auto)"
        ),
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
        help=("Number of devices to use, or 'auto' to let the Lightning trainer " "decide (default: auto)"),
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
            "Floating-point precision for the Lightning trainer: '32-true', "
            "'16-mixed', 'bf16-mixed', etc. (default: 32-true)"
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


def add_simploid_args(parser: ArgumentParser) -> None:
    """Register the arguments for fitting a Bezier simploid instead of a Bezier simplex."""
    parser.add_argument(
        "--simploid-type",
        "--simploid_type",
        dest="simploid_type",
        type=simploid_type,
        metavar="A1,...,An",
        help=(
            "Fit a Bezier simploid, i.e. a tensor product of Bezier simplices of dimensions A1,...,An "
            "(e.g. '2,1' for a triangle x segment prism), instead of a Bezier simplex; "
            "the parameters file must then have sum(Ai + 1) columns, one block of barycentric coordinates per factor"
        ),
    )
    parser.add_argument(
        "--solver",
        choices=("lstsq", "trainer"),
        metavar="{lstsq,trainer}",
        help=(
            "How to fit a Bezier simploid: 'lstsq' solves the least-squares problem exactly, "
            "'trainer' trains with the Lightning trainer options below (default: lstsq; requires --simploid-type)"
        ),
    )
    parser.add_argument(
        "--ridge",
        type=float,
        metavar="W",
        help="Weight of the ridge (Tikhonov) penalty for --solver=lstsq (default: 0.0; requires --simploid-type)",
    )
