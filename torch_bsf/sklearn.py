from collections.abc import Iterable, Sequence
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import torch

try:
    from sklearn.base import BaseEstimator, RegressorMixin

    _sklearn_available = True
except ImportError:
    BaseEstimator = object
    RegressorMixin = object
    _sklearn_available = False

from torch_bsf.preprocessing import NormalizeType
from torch_bsf.simploidal.bezier import BezierSimploid, ControlPointData
from torch_bsf.simploidal.combinatorics import SimploidIndex
from torch_bsf.simploidal.fitting import fit


def _check_sklearn() -> None:
    """Raise a clear ImportError when scikit-learn is not installed."""
    if not _sklearn_available:
        raise ImportError("scikit-learn is required for torch_bsf.sklearn. Install it with: pip install scikit-learn")


class BezierSimploidRegressor(RegressorMixin, BaseEstimator):
    """Scikit-learn wrapper for Bézier simplex and Bézier simploid fitting.

    Each sample of ``X`` is the parameter of a point on a simplex, or on a simploid
    of type ``simploid_type``, and the model is fitted with :func:`torch_bsf.fit`.

    Parameters
    ----------
    degree : int | Sequence[int], default=3
        The degree of every factor, or one degree per factor.
    simploid_type : Sequence[int] | None, default=None
        The type of the simploid.  If ``None``, the samples are on a simplex:
        the type is ``(n_features - 1,)``, or ``()`` (a constant) for a single feature.
    solver : {"lstsq", "trainer"}, default="lstsq"
        ``"lstsq"`` solves the least-squares problem exactly in double precision, and the fitted
        model keeps the floating dtype of ``y``; ``"trainer"`` trains the control points with
        :class:`lightning.pytorch.Trainer` in single precision.
    ridge : float, default=0.0
        Weight of the Tikhonov penalty on the control points (``solver="lstsq"`` only).
    smoothness_weight : float, default=0.0
        The weight of the smoothness penalty.
    init : BezierSimploid | tensor | mapping | None, default=None
        Initial control points; required by ``freeze``.
    freeze : Iterable of indices | None, default=None
        Indices of control points that keep their values in ``init``.
    normalize : {"none", "max", "std", "quantile"}, default="none"
        How ``y`` is normalized while fitting; the predictions are in the scale of ``y`` either way.
    batch_size : int | None, default=None
        Size of minibatches (``solver="trainer"`` only).
    max_epochs : int, default=1000
        Maximum number of epochs to train (``solver="trainer"`` only).
    accelerator : str, default="auto"
        Hardware accelerator to use ("cpu", "gpu", "auto", etc.; ``solver="trainer"`` only).
    devices : int | str, default="auto"
        Number of devices to use (``solver="trainer"`` only).
    precision : str, default="32-true"
        Floating point precision (``solver="trainer"`` only).
    trainer_kwargs : dict | None, default=None
        Additional keyword arguments for :class:`lightning.pytorch.Trainer` (``solver="trainer"`` only).

    The options marked ``solver="trainer"`` only are ignored by ``solver="lstsq"``.

    Attributes
    ----------
    model_ : BezierSimploid
        The fitted model.
    simploid_type_ : tuple[int, ...]
        The type of the fitted model.

    Examples
    --------
    >>> import numpy as np
    >>> from torch_bsf.sklearn import BezierSimploidRegressor
    >>> X = np.array([[1.0, 0.0], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0.0, 1.0]])
    >>> y = X[:, :1] ** 2
    >>> reg = BezierSimploidRegressor(degree=2).fit(X, y)
    >>> np.allclose(reg.predict(X), y)
    True

    On a square, the product of two segments:

    >>> from torch_bsf.simploidal import sample_simploid
    >>> X = sample_simploid((1, 1), 20, seed=0).numpy()
    >>> y = X[:, 0] * X[:, 2]
    >>> reg = BezierSimploidRegressor(degree=1, simploid_type=(1, 1)).fit(X, y)
    >>> np.allclose(reg.predict(X), y, atol=1e-6)
    True
    """

    def __init__(
        self,
        degree: int | Sequence[int] = 3,
        simploid_type: Sequence[int] | None = None,
        solver: Literal["lstsq", "trainer"] = "lstsq",
        ridge: float = 0.0,
        smoothness_weight: float = 0.0,
        init: BezierSimploid | ControlPointData | None = None,
        freeze: Iterable[SimploidIndex | Sequence[int]] | None = None,
        normalize: NormalizeType = "none",
        batch_size: int | None = None,
        max_epochs: int = 1000,
        accelerator: str = "auto",
        devices: int | str = "auto",
        precision: str = "32-true",
        trainer_kwargs: dict[str, Any] | None = None,
    ):
        _check_sklearn()
        self.degree = degree
        self.simploid_type = simploid_type
        self.solver = solver
        self.ridge = ridge
        self.smoothness_weight = smoothness_weight
        self.init = init
        self.freeze = freeze
        self.normalize = normalize
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.accelerator = accelerator
        self.devices = devices
        self.precision = precision
        self.trainer_kwargs = trainer_kwargs

    def _params(self, X: npt.NDArray[Any]) -> torch.Tensor:
        """The parameters of the fitted type: a single feature on a vertex carries no parameter."""
        if self.simploid_type_ == ():
            X = X[:, :0]
        return torch.from_numpy(np.ascontiguousarray(X))

    def fit(self, X: Any, y: Any) -> "BezierSimploidRegressor":
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training parameters on the simplex or simploid.
        y : array-like of shape (n_samples,) or (n_samples, n_values)
            Target values.  A one-dimensional ``y`` is treated as a single output,
            and :meth:`predict` then returns a one-dimensional array.

        Returns
        -------
        self : object
            Fitted estimator.

        Raises
        ------
        ValueError
            If the data do not match ``simploid_type``, or an option is not valid (see :func:`torch_bsf.fit`).
        """
        from sklearn.utils.validation import check_X_y

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        self._single_output = y.ndim == 1
        if self._single_output:
            y = y.reshape(-1, 1)
        if self.simploid_type is not None:
            self.simploid_type_ = tuple(int(a) for a in self.simploid_type)
        else:
            self.simploid_type_ = (X.shape[1] - 1,) if X.shape[1] > 1 else ()

        options: dict[str, Any] = {}
        if self.solver == "trainer":
            # Lightning trains in single precision.
            X, y = X.astype(np.float32), y.astype(np.float32)
            options = dict(
                batch_size=self.batch_size,
                max_epochs=self.max_epochs,
                accelerator=self.accelerator,
                devices=self.devices,
                precision=self.precision,
                **(self.trainer_kwargs or {}),
            )
        else:
            # Evaluate the basis in double precision; the solution keeps the floating dtype of y.
            X = X.astype(np.float64)
            if not np.issubdtype(y.dtype, np.floating):
                y = y.astype(np.float64)
        self.model_ = fit(
            self._params(X),
            torch.from_numpy(np.ascontiguousarray(y)),
            self.degree,
            simploid_type=self.simploid_type_,
            init=self.init,
            freeze=self.freeze,
            smoothness_weight=self.smoothness_weight,
            normalize=self.normalize,
            solver=self.solver,
            ridge=self.ridge,
            **options,
        )
        self.n_features_in_ = X.shape[1]
        self.n_outputs_ = y.shape[1]
        return self

    def predict(self, X: Any) -> npt.NDArray[Any]:
        """Predict using the fitted model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input parameters.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, n_values) or (n_samples,)
            Predicted values; one-dimensional if the model was fitted with a one-dimensional ``y``.
        """
        from sklearn.utils.validation import check_array, check_is_fitted

        check_is_fitted(self)
        X = check_array(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but {type(self).__name__} is expecting {self.n_features_in_} features as input."
            )
        cp = self.model_.control_points
        self.model_.eval()
        with torch.no_grad():
            ys = self.model_(self._params(X).to(device=cp.device, dtype=cp.dtype))
        y_pred = cast(npt.NDArray[Any], ys.cpu().numpy())
        return y_pred.ravel() if getattr(self, "_single_output", False) else y_pred

    def score(self, X: Any, y: Any, sample_weight: Any = None) -> float:
        """Return the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
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
