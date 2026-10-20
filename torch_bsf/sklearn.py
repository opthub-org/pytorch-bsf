from typing import Any, Iterable, Literal, cast

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
from torch_bsf.simploidal import fit_simploid
from torch_bsf.simploidal.fitting import _solve_least_squares


def _check_sklearn() -> None:
    """Raise a clear ImportError when scikit-learn is not installed."""
    if not _sklearn_available:
        raise ImportError("scikit-learn is required for torch_bsf.sklearn. Install it with: pip install scikit-learn")


def _fit_lstsq(params: torch.Tensor, values: torch.Tensor, degree: int, ridge: float) -> BezierSimplex:
    """Fit a Bézier simplex by exact (optionally ridge-regularized) linear least squares.

    A Bézier simplex with ``n_params`` parameters is the Bézier simploid of type
    ``(n_params - 1,)``, so this delegates to :func:`torch_bsf.simploidal.fit_simploid`
    and converts the result into a :class:`torch_bsf.BezierSimplex`.  A 0-simplex
    (``n_params == 1``), which is not a simploid, has the single basis function
    :math:`t_1^{\\mathrm{degree}}`.
    """
    n_params = int(params.shape[1])
    if n_params == 1:
        if degree < 0:
            raise ValueError(f"degree must be non-negative, but {degree} is given.")
        solution = _solve_least_squares(params**degree, values, ridge).to(values.dtype)
        constant: dict[tuple[int, ...], torch.Tensor] = {(degree,): solution[0]}
        return BezierSimplex(constant).to(values.dtype)
    simploid = fit_simploid(params, values, (n_params - 1,), degree, solver="lstsq", ridge=ridge)
    control_points = simploid.control_points.detach()
    data = {alpha[0]: row for alpha, row in zip(simploid.indices(), control_points)}
    return BezierSimplex(data).to(control_points.dtype)


class BezierSimplexRegressor(BaseEstimator, RegressorMixin):
    """Scikit-learn wrapper for Bézier Simplex Fitting.

    Parameters
    ----------
    degree : int, default=3
        The degree of the Bézier simplex.
    smoothness_weight : float, default=0.0
        The weight of smoothness penalty (``solver="trainer"`` only).
    init : BezierSimplex | ControlPointsData | None, default=None
        Initial control points or model (``solver="trainer"`` only).
    freeze : Iterable[Index] | None, default=None
        Indices of control points to freeze during training (``solver="trainer"`` only).
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
        Additional keyword arguments for lightning.pytorch.Trainer (``solver="trainer"`` only).
    solver : {"trainer", "lstsq"}, default="trainer"
        How the control points are computed.

        - ``"trainer"`` trains them with :func:`torch_bsf.fit`
          (:class:`lightning.pytorch.Trainer`).
        - ``"lstsq"`` solves the linear least-squares problem
          :math:`\\min_P \\|B(X) P - y\\|^2 + \\mathrm{ridge}\\,\\|P\\|^2` exactly in double precision,
          where :math:`B(X)` is the Bernstein basis evaluated at the samples.  The result is
          the same as :func:`torch_bsf.simploidal.fit_simploid` with ``simploid_type=(n_params - 1,)``
          and ``solver="lstsq"``; under-determined problems without ``ridge`` get the
          minimum-norm solution.  The training options marked "``solver="trainer"`` only" are
          ignored, except that ``init``, ``freeze`` and a nonzero ``smoothness_weight``
          raise :class:`ValueError` in :meth:`fit`.  The fitted model keeps the floating dtype of ``y``.
    ridge : float, default=0.0
        Weight of the Tikhonov penalty :math:`\\mathrm{ridge}\\,\\|P\\|^2` (``solver="lstsq"`` only;
        a nonzero value with ``solver="trainer"`` raises :class:`ValueError` in :meth:`fit`).

    Examples
    --------
    >>> import numpy as np
    >>> from torch_bsf.sklearn import BezierSimplexRegressor
    >>> X = np.array([[1.0, 0.0], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0.0, 1.0]])
    >>> y = X[:, :1] ** 2
    >>> reg = BezierSimplexRegressor(degree=2, solver="lstsq").fit(X, y)
    >>> np.allclose(reg.predict(X), y)
    True
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
        solver: Literal["trainer", "lstsq"] = "trainer",
        ridge: float = 0.0,
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
        self.solver = solver
        self.ridge = ridge

    def _check_solver_options(self) -> None:
        """Reject option combinations that the selected solver cannot honor."""
        if self.solver == "trainer":
            if self.ridge:
                raise ValueError("ridge is supported only by solver='lstsq'.")
            return
        if self.solver != "lstsq":
            raise ValueError(f"Unknown solver {self.solver!r}; expected 'trainer' or 'lstsq'.")
        if self.init is not None:
            raise ValueError("init is not supported by solver='lstsq' (the least-squares solution does not depend on it).")
        if self.freeze:
            raise ValueError("freeze is not supported by solver='lstsq'.")
        if self.smoothness_weight:
            raise ValueError("smoothness_weight is not supported by solver='lstsq'; use ridge instead.")

    def fit(self, X: Any, y: Any):
        """Fit the Bézier simplex model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_params)
            Training data (parameters on a simplex).
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
            If ``solver`` is unknown or an option is not supported by the selected solver.
        """
        # Validate data
        from sklearn.utils.validation import check_X_y

        self._check_solver_options()
        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
        self._single_output = y.ndim == 1
        if self._single_output:
            y = y.reshape(-1, 1)

        if self.solver == "lstsq":
            # Evaluate the basis in double precision; the solution keeps the floating dtype of y.
            if not np.issubdtype(y.dtype, np.floating):
                y = y.astype(np.float64)
            self.model_ = _fit_lstsq(
                torch.from_numpy(np.asarray(X, dtype=np.float64)),
                torch.from_numpy(np.ascontiguousarray(y)),
                self.degree,
                self.ridge,
            )
        else:
            # Fit using the core library's fit function
            self.model_ = fit(
                params=torch.from_numpy(X).float(),
                values=torch.from_numpy(y).float(),
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
        y_pred : ndarray of shape (n_samples, n_values) or (n_samples,)
            Predicted values; one-dimensional if the model was fitted with a one-dimensional ``y``.
        """
        from sklearn.utils.validation import check_array, check_is_fitted

        check_is_fitted(self)
        X = check_array(X)

        ts = torch.from_numpy(X).to(device=self.model_.device, dtype=self.model_.control_points.matrix.dtype)
        self.model_.eval()
        with torch.no_grad():
            ys = self.model_(ts)

        y_pred = cast(np.ndarray, ys.cpu().numpy())
        return y_pred.ravel() if getattr(self, "_single_output", False) else y_pred

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
