import logging
from typing import Any

import torch
from torch.nn import functional as F

from torch_bsf.simploidal.fitting import _check_n_samples, _fold_indices, fit

_logger = logging.getLogger(__name__)


def select_degree(
    params: torch.Tensor,
    values: torch.Tensor,
    min_degree: int = 1,
    max_degree: int = 5,
    num_folds: int = 5,
    patience: int = 1,
    *,
    shuffle: bool = True,
    seed: int | None = None,
    **kwargs: Any,
) -> int:
    """Select the best degree for a Bézier simplex or simploid using cross-validation.

    For each degree, the samples are split into ``num_folds`` folds, a model is
    fitted with :func:`torch_bsf.fit` to all folds but one, and its mean squared
    error is measured on the held-out fold.  The degree with the least mean of
    these errors wins.

    Parameters
    ----------
    params
        The input parameters.
    values
        The output values.
    min_degree
        Starting degree to check.
    max_degree
        Ending degree to check.
    num_folds
        Number of folds for cross-validation (capped at the number of samples).
    patience
        Number of consecutive degrees without improvement before early
        stopping.  Defaults to ``1``, which stops as soon as the mean CV
        MSE increases for the first time (after ``min_degree + 1``).
        Increase this value to tolerate transient upswings caused by
        numerical noise.
    shuffle
        Whether to shuffle the samples before splitting them into folds.
    seed
        Random seed for shuffling, also passed to :func:`torch_bsf.fit`.
    kwargs
        Other arguments for :func:`torch_bsf.fit`, e.g. ``simploid_type``, ``solver``,
        ``smoothness_weight`` and :class:`lightning.pytorch.Trainer` arguments.

    Returns
    -------
    int
        The best degree found.

    Raises
    ------
    ValueError
        If ``patience < 1``, there are fewer than 2 samples, ``params`` and ``values`` have different numbers
        of samples, or from :func:`torch_bsf.fit`.

    Examples
    --------
    >>> from torch_bsf.model_selection.degree_selection import select_degree
    >>> from torch_bsf.sampling import simplex_grid
    >>> params = simplex_grid(n_params=3, degree=6)
    >>> values = params[:, :1] ** 2
    >>> select_degree(params, values, max_degree=4, seed=0)
    2
    """
    if patience < 1:
        raise ValueError(f"patience must be a positive integer (>= 1), got {patience!r}.")
    params, values = torch.as_tensor(params), torch.as_tensor(values)
    _check_n_samples(params, values)
    if kwargs.get("freeze") is not None:
        kwargs["freeze"] = list(kwargs["freeze"])  # an iterator would be consumed by the first fold
    folds = _fold_indices(len(params), num_folds, shuffle, seed)

    best_degree = min_degree
    best_mse = float("inf")
    no_improve_count = 0

    for d in range(min_degree, max_degree + 1):
        _logger.info("Checking degree %d...", d)
        mses = []
        for k, test in enumerate(folds):
            train = torch.cat([fold for i, fold in enumerate(folds) if i != k])
            model = fit(params[train], values[train], d, seed=seed, **kwargs)
            with torch.no_grad():
                predicted = model(params[test])
                mses.append(F.mse_loss(predicted, values[test].to(predicted)).item())
        mean_mse = sum(mses) / len(mses)
        _logger.info("Degree %d: Mean MSE = %.6f", d, mean_mse)

        if mean_mse < best_mse:
            best_mse = mean_mse
            best_degree = d
            no_improve_count = 0
        else:
            no_improve_count += 1
            if d > min_degree and no_improve_count >= patience:
                _logger.info(
                    "No improvement for %d consecutive degree(s) at degree %d, stopping.",
                    no_improve_count,
                    d,
                )
                break

    return best_degree


def _cli_main() -> None:
    """Entry point for ``python -m torch_bsf.model_selection.degree_selection``."""
    from argparse import ArgumentParser
    from pathlib import Path

    from torch_bsf._cli_args import _optional_simploid_type, check_columns, load_table
    from torch_bsf.validator import int_or_str

    parser = ArgumentParser(
        prog="python -m torch_bsf.model_selection.degree_selection",
        description="Automatic degree selection for a Bezier simplex (or simploid) via k-fold cross-validation",
    )
    parser.add_argument("--params", type=Path, required=True, help="Path to the input parameters CSV file")
    parser.add_argument("--values", type=Path, required=True, help="Path to the output values CSV file")
    parser.add_argument("--header", type=int, default=0, help="Number of header rows to skip (default: 0)")
    parser.add_argument(
        "--simploid-type",
        "--simploid_type",
        dest="simploid_type",
        type=_optional_simploid_type,
        metavar="A1,...,An",
        help="Type of the Bezier simploid (default: a Bezier simplex of dimension n_params - 1)",
    )
    parser.add_argument("--min_degree", type=int, default=1, help="Minimum degree to search (default: 1)")
    parser.add_argument("--max_degree", type=int, default=5, help="Maximum degree to search (default: 5)")
    parser.add_argument("--num_folds", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    parser.add_argument(
        "--patience",
        type=int,
        default=1,
        help="Consecutive non-improving degrees before stopping (default: 1)",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for the folds and training (default: none)")
    parser.add_argument(
        "--solver",
        choices=("lstsq", "trainer"),
        default="lstsq",
        help="'lstsq' solves the least-squares problem exactly, 'trainer' trains with Lightning (default: lstsq)",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for --solver=trainer (default: full-batch)")
    parser.add_argument("--max_epochs", type=int, default=2, help="Training epochs per fold for --solver=trainer (default: 2)")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator for --solver=trainer (default: auto)")
    parser.add_argument(
        "--devices", type=int_or_str, default="auto", help="Devices for --solver=trainer, integer or 'auto' (default: auto)"
    )
    parser.add_argument(
        "--loglevel",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Python logging level for degree selection progress (default: INFO)",
    )
    args = parser.parse_args()

    if args.min_degree > args.max_degree:
        parser.error(f"--min_degree ({args.min_degree}) must be <= --max_degree ({args.max_degree})")

    logging.basicConfig(
        level=getattr(logging, args.loglevel),
        format="%(levelname)s:%(name)s:%(message)s",
        force=True,
    )

    params_tensor = load_table(args.params, args.header)
    values_tensor = load_table(args.values, args.header)
    if args.simploid_type is not None:
        check_columns(parser, "--params", params_tensor, args.simploid_type)

    fit_kwargs: dict[str, Any] = {"simploid_type": args.simploid_type, "solver": args.solver}
    if args.solver == "trainer":
        fit_kwargs.update(
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            accelerator=args.accelerator,
            devices=args.devices,
        )

    try:
        best_degree = select_degree(
            params=params_tensor,
            values=values_tensor,
            min_degree=args.min_degree,
            max_degree=args.max_degree,
            num_folds=args.num_folds,
            patience=args.patience,
            seed=args.seed,
            **fit_kwargs,
        )
    except ValueError as e:
        parser.error(str(e))

    print(f"Best degree: {best_degree}")


if __name__ == "__main__":
    _cli_main()
