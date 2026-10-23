"""Scikit-learn integration using the same numerical problem and results."""

from collections.abc import Sequence
from typing import Any, cast

import numpy.typing as npt
import torch

try:
    from sklearn.base import BaseEstimator, RegressorMixin

    _sklearn_available = True
except ModuleNotFoundError as error:
    if error.name != "sklearn" and not (error.name == "sklearn.base" and __import__("sys").modules.get("sklearn") is None):
        raise

    class BaseEstimator:  # type: ignore[no-redef]
        pass

    class RegressorMixin:  # type: ignore[no-redef]
        pass

    _sklearn_available = False

from torch_bsf.core import BezierSpace, FixedControls, QuadraticPenalty, Samples
from torch_bsf.fitting import fit
from torch_bsf.solvers import DenseLeastSquares, LeastSquaresSolver


def _check_sklearn() -> None:
    if not _sklearn_available:
        raise ImportError('scikit-learn is required; install "pytorch-bsf[sklearn]".')


class BezierRegressor(RegressorMixin, BaseEstimator):
    """Fit a single simplex or product using explicit numerical settings.

    Parameters
    ----------
    degree
        Polynomial degree, or one degree per simplex factor.
    dimensions
        Factor dimensions; inferred as a single simplex from X when omitted.
    solver, penalty, fixed
        The same typed settings accepted by :func:`torch_bsf.fit`.

    Attributes
    ----------
    result_, model_, diagnostics_
        Complete fit evidence, fitted model, and numerical diagnostics.
    """

    def __init__(
        self,
        degree: int | Sequence[int] = 3,
        *,
        dimensions: Sequence[int] | None = None,
        solver: LeastSquaresSolver = DenseLeastSquares(),
        penalty: QuadraticPenalty = QuadraticPenalty(),
        fixed: FixedControls | None = None,
    ):
        _check_sklearn()
        self.degree = degree
        self.dimensions = dimensions
        self.solver = solver
        self.penalty = penalty
        self.fixed = fixed

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> "BezierRegressor":
        """Fit weighted observations and expose their complete FitResult."""
        from sklearn.utils.validation import check_X_y

        x, target = check_X_y(X, y, multi_output=True, y_numeric=True)
        self._single_output = target.ndim == 1
        dimensions = ((x.shape[1] - 1,) if x.shape[1] > 1 else ()) if self.dimensions is None else self.dimensions
        space = BezierSpace.simploid(dimensions, degree=self.degree)
        self._constant = not space.cells[0].dimensions
        if self._constant and x.shape[1] != 1:
            raise ValueError("constant regression requires one dummy feature column.")
        coordinates = x[:, :0] if self._constant else x
        if coordinates.shape[1] != space.width:
            raise ValueError("X columns must match the space's barycentric coordinate width.")
        self.result_ = fit(
            space,
            Samples(coordinates, target, weights=sample_weight),
            solver=self.solver,
            penalty=self.penalty,
            fixed=self.fixed,
        ).require_success()
        self.model_ = self.result_.model
        self.diagnostics_ = self.result_.diagnostics
        self.n_features_in_ = x.shape[1]
        return self

    def predict(self, X: Any) -> npt.NDArray[Any]:
        """Predict with the fitted model; preserve the target's original rank."""
        from sklearn.utils.validation import check_array, check_is_fitted

        check_is_fitted(self, "result_")
        x = check_array(X)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("X has a different number of features from training.")
        with torch.no_grad():
            values = cast(npt.NDArray[Any], self.model_(torch.as_tensor(x[:, :0] if self._constant else x)).cpu().numpy())
        return values.ravel() if self._single_output else values
