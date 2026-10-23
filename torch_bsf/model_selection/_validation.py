"""Cross-validation and explicit, fully recorded degree search."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
    they are never silently dropped from model selection.
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
            difference = fitted.model(held_out.points) - held_out.values.cpu().double()
            squared = difference.square().mean(1)
            absolute = difference.abs().mean(1)
            weight = held_out.weights / held_out.weights.sum()
            metrics = {"mse": (squared * weight).sum().item(), "mae": (absolute * weight).sum().item()}
        folds.append(FoldResult(train.clone(), test.clone(), fitted, metrics, held_out.weights.sum().item()))
    mse = torch.tensor([fold.metrics["mse"] for fold in folds], dtype=torch.float64)
    total = sum(fold.test_weight for fold in folds)
    metrics = {
        "mean_mse": mse.mean().item(),
        "std_mse": mse.std(unbiased=False).item(),
        "pooled_mse": sum(fold.metrics["mse"] * fold.test_weight for fold in folds) / total,
    }
    return CVResult(tuple(folds), metrics, all(f.result.success for f in folds))


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
    win. To refit, explicitly call ``fit(result.best_space, samples, ...)``.
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
