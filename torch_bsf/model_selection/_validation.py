"""Cross-validation and explicit, fully recorded degree search."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import torch

from torch_bsf.core import BezierModel, BezierSpace, FixedControls, QuadraticPenalty, Samples, _integer
from torch_bsf.fitting import fit
from torch_bsf.results import FitResult
from torch_bsf.sampling import _generator
from torch_bsf.solvers import DenseLeastSquares, LeastSquaresSolver


@dataclass(frozen=True)
class KFold:
    """Local-generator k-fold splits; an impossible requested split count is an error."""

    n_splits: int = 5
    shuffle: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        _integer(self.n_splits, "n_splits", 2)

    def split(self, n_samples: int) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Return disjoint train/test indices, covering every test sample once."""
        if self.n_splits > n_samples:
            raise ValueError("n_splits cannot exceed n_samples.")
        order = torch.randperm(n_samples, generator=_generator(self.seed)) if self.shuffle else torch.arange(n_samples)
        folds = torch.tensor_split(order, self.n_splits)
        return tuple((torch.cat([fold for j, fold in enumerate(folds) if j != k]), test) for k, test in enumerate(folds))


@dataclass(frozen=True)
class FoldResult:
    """A fitted training fold with its untouched held-out indices and metrics."""

    train_indices: torch.Tensor
    test_indices: torch.Tensor
    result: FitResult
    metrics: dict[str, float]
    test_weight: float


@dataclass(frozen=True)
class CVResult:
    """Validation evidence; never an implicitly refitted model or ensemble."""

    folds: tuple[FoldResult, ...]
    metrics: dict[str, float]
    success: bool

    @property
    def models(self) -> tuple[BezierModel, ...]:
        """Models available for explicitly constructing an Ensemble."""
        return tuple(fold.result.model for fold in self.folds)


def cross_validate(
    space: BezierSpace,
    samples: Samples,
    *,
    cv: KFold = KFold(),
    solver: LeastSquaresSolver = DenseLeastSquares(),
    penalty: QuadraticPenalty = QuadraticPenalty(),
    fixed: FixedControls | None = None,
) -> CVResult:
    """Fit each training fold and retain split, solver and held-out evidence.

    Every fold must have positive observation weight in both its train and test
    portions. Failed solves remain in the result and make the whole CV fail;
    they are never silently dropped from model selection. Non-finite held-out
    or aggregate metrics also make CV fail, retaining every fold for inspection.
    Zero-weight held-out rows are excluded from metric evaluation.
    """
    return _cross_validate(space, samples, cv.split(len(samples)), solver, penalty, fixed)


def _cross_validate(
    space: BezierSpace,
    samples: Samples,
    splits: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    solver: LeastSquaresSolver,
    penalty: QuadraticPenalty,
    fixed: FixedControls | None,
) -> CVResult:
    folds = []
    for train, test in splits:
        training, held_out = samples.take(train), samples.take(test)
        fitted = fit(space, training, solver=solver, penalty=penalty, fixed=fixed)
        with torch.no_grad():
            metrics = _held_out_metrics(fitted.model, held_out)
        folds.append(FoldResult(train.clone(), test.clone(), fitted, metrics, held_out.weights.sum().item()))
    mse = torch.tensor([fold.metrics["mse"] for fold in folds], dtype=torch.float64)
    scale = mse.abs().max()
    scaled = mse / scale if scale > 0 else mse
    metrics = {
        "mean_mse": (scaled.mean() * scale).item(),
        "std_mse": (scaled.std(unbiased=False) * scale).item(),
        "pooled_mse": _pooled_mse(folds),
    }
    success = all(f.result.success and all(math.isfinite(v) for v in f.metrics.values()) for f in folds) and all(
        math.isfinite(v) for v in metrics.values()
    )
    return CVResult(tuple(folds), metrics, success)


def _pooled_mse(folds: Sequence[FoldResult]) -> float:
    errors = [fold.metrics["mse"] for fold in folds]
    if not all(math.isfinite(error) for error in errors):
        return sum(errors)
    # These are already CPU scalar fold summaries. A single exact reduction
    # avoids overflowing either the weight sum or independently rounded terms,
    # and preserves tiny weights whose product with a large error is meaningful.
    weights = [Fraction.from_float(fold.test_weight) for fold in folds]
    numerator = sum(Fraction.from_float(error) * weight for error, weight in zip(errors, weights, strict=True))
    return float(numerator / sum(weights))


def _weighted_error(
    difference: torch.Tensor, weights: torch.Tensor, total: float, outputs: int, power: int, exponent_shift: int | torch.Tensor
) -> float:
    # Combine mantissas before scaling exponents: weight/total and difference**2
    # may each be unrepresentable even though their product is finite.
    value, exponent = torch.frexp(difference.abs())
    weight_value, weight_exponent = torch.frexp(weights[:, None])
    total_value, total_exponent = math.frexp(total)
    mantissa = value.pow(power) * weight_value / total_value / outputs
    mantissa, adjustment = torch.frexp(mantissa)
    exponent = power * (exponent + exponent_shift) + weight_exponent - total_exponent + adjustment
    exponent = torch.where(mantissa == 0, 0, exponent)
    if torch.isfinite(difference).all() and (((exponent > 1000) | (exponent < -1021)) & (mantissa != 0)).any():
        # Boundary contributions must be combined before rounding: separately
        # rounded terms can overflow on addition or all underflow to zero.
        shifts = torch.zeros_like(difference, dtype=torch.int32) + exponent_shift
        numerator = sum(
            Fraction(weight)
            * sum((Fraction(abs(value)) * 2 ** int(shift)) ** power for value, shift in zip(row, row_shifts, strict=True))
            for row, row_shifts, weight in zip(difference.tolist(), shifts.tolist(), weights.tolist(), strict=True)
        )
        exact = numerator / (sum(Fraction(weight) for weight in weights.tolist()) * outputs)
        try:
            return float(exact)
        except OverflowError:
            return math.inf
    # PyTorch 2.3 ldexp forms its intermediate power in the default float32
    # dtype even for float64 inputs. Use an explicit float64 base, and split
    # the exponent so a power itself does not overflow at a finite boundary.
    first = exponent.clamp(min=-512, max=512)
    base = mantissa.new_tensor(2.0)
    terms = mantissa * torch.pow(base, first) * torch.pow(base, exponent - first)
    return terms.sum().item()


def _held_out_metrics(model: BezierModel, held_out: Samples) -> dict[str, float]:
    # Zero-weight observations do not participate in prediction or arithmetic.
    selected = torch.nonzero(held_out.weights > 0).flatten()
    positive = held_out.take(selected)
    predicted = model(positive.points)
    observed = positive.values.cpu().double()
    difference = predicted - observed
    shift: int | torch.Tensor = 0
    finite = torch.isfinite(difference)
    if not finite.all():
        # Only overflowing entries need halving. Applying it to every row
        # would erase otherwise representable subnormal residuals.
        difference = torch.where(finite, difference, predicted / 2 - observed / 2)
        shift = (~finite).to(torch.int32)
    weights, total = positive.weights, positive.weights.sum().item()
    return {
        "mse": _weighted_error(difference, weights, total, model.n_outputs, 2, shift),
        "mae": _weighted_error(difference, weights, total, model.n_outputs, 1, shift),
    }


@dataclass(frozen=True)
class DegreeCandidate:
    """A candidate degree and all its validation evidence."""

    degree: int | tuple[int, ...]
    space: BezierSpace
    validation: CVResult


@dataclass(frozen=True)
class DegreeSearchResult:
    """All evaluated candidates and a selection, without an implicit full-data fit."""

    candidates: tuple[DegreeCandidate, ...]
    best_index: int | None
    searched_all: bool = True

    @property
    def success(self) -> bool:
        return self.best_index is not None

    @property
    def best_degree(self) -> int | tuple[int, ...] | None:
        return self.candidates[self.best_index].degree if self.best_index is not None else None

    @property
    def best_space(self) -> BezierSpace | None:
        return self.candidates[self.best_index].space if self.best_index is not None else None


def select_degree(
    space: BezierSpace,
    samples: Samples,
    *,
    degrees: Sequence[int | Sequence[int]],
    cv: KFold = KFold(),
    solver: LeastSquaresSolver = DenseLeastSquares(),
    penalty: QuadraticPenalty = QuadraticPenalty(),
) -> DegreeSearchResult:
    """Evaluate every specified degree on identical folds and record the result.

    Invalid degrees raise; failed numerical solves remain visible and cannot
    win. Non-finite validation metrics likewise make candidates ineligible.
    If none is eligible, best_index is None and success is False.
    To refit, explicitly call ``fit(result.best_space, samples, ...)``.
    """
    if not degrees:
        raise ValueError("degrees must contain at least one candidate.")
    candidates = []
    splits = cv.split(len(samples))
    for degree in degrees:
        compiled = space.with_degree(degree)
        evaluated = _cross_validate(compiled, samples, splits, solver, penalty, None)
        candidates.append(DegreeCandidate(degree if isinstance(degree, int) else tuple(degree), compiled, evaluated))
    eligible = [k for k, c in enumerate(candidates) if c.validation.success]
    best = min(eligible, key=lambda k: candidates[k].validation.metrics["pooled_mse"]) if eligible else None
    return DegreeSearchResult(tuple(candidates), best)
