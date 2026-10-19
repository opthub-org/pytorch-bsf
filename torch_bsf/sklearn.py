from typing import Any, Iterable, cast

import numpy as np
import torch

try:
    from sklearn.base import BaseEstimator, RegressorMixin
    _sklearn_available = True
except ImportError:
    BaseEstimator = object  # type: ignore[assignment,misc]
    RegressorMixin = object  # type: ignore[assignment,misc]
    _sklearn_available = False

from torch_bsf.bezier_simplex import BezierSimplex, ControlPointsData, Index, fit


def _check_sklearn() -> None:
    """Raise a clear ImportError when scikit-learn is not installed."""
    if not _sklearn_available:
        raise ImportError(
            "scikit-learn is required for torch_bsf.sklearn. "
            "Install it with: pip install scikit-learn"
        )


class BezierSimplexRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn wrapper for Bézier Simplex Fitting.

    Parameters
    ----------
    degree : int, default=3
        The degree of the Bézier simplex.
    smoothness_weight : float, default=0.0
        The weight of smoothness penalty.
    init : BezierSimplex | ControlPointsData | None, default=None
        Initial control points or model.
    freeze : Iterable[Index] | None, default=None
        Indices of control points to freeze during training.
    batch_size : int | None, default=None
        Size of minibatches.
    max_epochs : int, default=1000
        Maximum number of epochs to train.
    accelerator : str, default="auto"
        Hardware accelerator to use ("cpu", "gpu", "auto", etc.).
    devices : int | str, default="auto"
        Number of devices to use.
    precision : str, default="32-true"
        Floating point precision.
    trainer_kwargs : dict | None, default=None
        Additional keyword arguments for lightning.pytorch.Trainer.
    """

    def __init__(
        self,
        degree: int = 3,
        smoothness_weight: float = 0.0,
        init: BezierSimplex | ControlPointsData | None = None,
        freeze: Iterable[Index] | None = None,
        batch_size: int | None = None,
        max_epochs: int = 1000,
        accelerator: str = "auto",
        devices: int | str = "auto",
        precision: str = "32-true",
        trainer_kwargs: dict[str, Any] | None = None,
    ):
        _check_sklearn()
        self.degree = degree
        self.smoothness_weight = smoothness_weight
        self.init = init
        self.freeze = freeze
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        self.precision = precision
        self.trainer_kwargs = trainer_kwargs

    def fit(self, X: Any, y: Any):
        """Fit the Bézier simplex model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_params)
            Training data (parameters on a simplex).
        y : array-like of shape (n_samples, n_values)
            Target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate data
        from sklearn.utils.validation import check_X_y

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        # Convert to torch tensors
        ts = torch.from_numpy(X).float()
        ys = torch.from_numpy(y).float()

        # Fit using the core library's fit function
        self.model_ = fit(
            params=ts,
            values=ys,
            degree=self.degree,
            init=self.init,
            smoothness_weight=self.smoothness_weight,
            freeze=self.freeze,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=self.devices,
            precision=self.precision,
            **(self.trainer_kwargs or {}),
        )

        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = y.shape[1]

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict using the Bézier simplex model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_params)
            Input parameters.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, n_values)
            Predicted values.
        """
        from sklearn.utils.validation import check_array, check_is_fitted

        check_is_fitted(self)
        X = check_array(X)

        ts = torch.from_numpy(X).float().to(self.model_.device)
        self.model_.eval()
        with torch.no_grad():
            ys = self.model_(ts)

        return cast(np.ndarray, ys.cpu().numpy())

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_params)
            Test samples.
        y : array-like of shape (n_samples, n_values)
            True values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            R^2 of self.predict(X) wrt. y.
        """
        from sklearn.metrics import r2_score

        return float(r2_score(y, self.predict(X), sample_weight=sample_weight))
