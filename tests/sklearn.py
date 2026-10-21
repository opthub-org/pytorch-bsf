from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

pytest.importorskip("sklearn")

from torch_bsf.preprocessing import NormalizeType
from torch_bsf.simploidal.fitting import Solver
from torch_bsf.sklearn import BezierSimploidRegressor


def _make_data(n_params: int = 3, n_values: int = 2, n_samples: int = 20):
    """Return (X, y) on the standard simplex with output dimension n_values."""
    rng = np.random.default_rng(42)
    raw = rng.random((n_samples, n_params))
    X = raw / raw.sum(axis=1, keepdims=True)
    y = (1.0 - X**2)[:, :n_values]
    return X.astype(np.float32), y.astype(np.float32)


class TestBezierSimploidRegressor:
    def test_fit_predict(self):
        X, y = _make_data()
        reg = BezierSimploidRegressor(
            degree=2,
            max_epochs=5,
            accelerator="cpu",
            devices=1,
            trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
        )
        reg.fit(X, y)
        preds = reg.predict(X)
        assert preds.shape == y.shape

    def test_score_returns_float(self):
        X, y = _make_data()
        reg = BezierSimploidRegressor(
            degree=2,
            max_epochs=5,
            accelerator="cpu",
            devices=1,
            trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
        )
        reg.fit(X, y)
        s = reg.score(X, y)
        assert isinstance(s, float)

    def test_predict_requires_fit(self):
        from sklearn.exceptions import NotFittedError

        reg = BezierSimploidRegressor(degree=2)
        X, _ = _make_data()
        with pytest.raises(NotFittedError):
            reg.predict(X)

    def test_get_set_params(self):
        reg = BezierSimploidRegressor(degree=3, max_epochs=50)
        params = reg.get_params()
        assert params["degree"] == 3
        assert params["max_epochs"] == 50
        reg.set_params(degree=2)
        assert reg.degree == 2

    def test_sklearn_interface_compliance(self):
        """Verify that BezierSimploidRegressor satisfies the core sklearn estimator interface."""
        from sklearn.base import clone

        reg = BezierSimploidRegressor(degree=2, max_epochs=5)
        # clone() relies on get_params() / __init__ signature parity
        cloned = clone(reg)
        assert cloned.degree == reg.degree
        assert cloned.max_epochs == reg.max_epochs
        # Cloned estimator should not share state
        assert cloned is not reg


class TestSetParams:
    def test_set_params_returns_self(self):
        """set_params() must return self to support method chaining."""
        reg = BezierSimploidRegressor(degree=3)
        result = reg.set_params(degree=2)
        assert result is reg

    def test_set_params_updates_multiple_params(self):
        """set_params() updates several attributes in one call."""
        reg = BezierSimploidRegressor(degree=3, max_epochs=100, smoothness_weight=0.1)
        reg.set_params(degree=1, max_epochs=10, smoothness_weight=0.5)
        assert reg.degree == 1
        assert reg.max_epochs == 10
        assert reg.smoothness_weight == 0.5

    def test_set_params_reflected_in_get_params(self):
        """get_params() reflects values changed by set_params()."""
        reg = BezierSimploidRegressor(degree=3, max_epochs=50)
        reg.set_params(degree=1)
        params = reg.get_params()
        assert params["degree"] == 1
        assert params["max_epochs"] == 50

    def test_set_params_reflected_in_fit(self):
        """After set_params(), fit trains with the updated degree."""
        X, y = _make_data()
        reg = BezierSimploidRegressor(
            degree=3,
            max_epochs=3,
            accelerator="cpu",
            devices=1,
            trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
        )
        reg.set_params(degree=1)
        reg.fit(X, y)
        assert reg.model_.degree == (1,)

    def test_set_params_invalid_key_raises(self):
        """set_params() with an unknown parameter name raises ValueError."""
        reg = BezierSimploidRegressor(degree=3)
        with pytest.raises(ValueError):
            reg.set_params(nonexistent_param=42)


class TestPipelineIntegration:
    _kwargs: dict[str, Any] = dict(
        degree=2,
        max_epochs=3,
        accelerator="cpu",
        devices=1,
        trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
    )

    def test_pipeline_fit_predict(self):
        """BezierSimploidRegressor works as the final step of a Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import FunctionTransformer

        def to_simplex(X):
            return X / X.sum(axis=1, keepdims=True)

        rng = np.random.default_rng(0)
        X_raw = rng.random((20, 3)).astype(np.float32)
        _, y = _make_data()

        pipe = Pipeline(
            [
                ("normalize", FunctionTransformer(to_simplex)),
                ("reg", BezierSimploidRegressor(**self._kwargs)),
            ]
        )
        pipe.fit(X_raw, y)
        preds = pipe.predict(X_raw)
        assert preds.shape == y.shape

    def test_pipeline_score(self):
        """Pipeline.score() delegates to BezierSimploidRegressor.score()."""
        from sklearn.pipeline import Pipeline

        X, y = _make_data()
        pipe = Pipeline([("reg", BezierSimploidRegressor(**self._kwargs))])
        pipe.fit(X, y)
        s = pipe.score(X, y)
        assert isinstance(s, float)

    def test_pipeline_set_params(self):
        """Pipeline.set_params() propagates to BezierSimploidRegressor via step__param notation."""
        from sklearn.pipeline import Pipeline

        X, y = _make_data()
        pipe = Pipeline([("reg", BezierSimploidRegressor(**self._kwargs))])
        pipe.set_params(reg__degree=1)
        assert pipe.named_steps["reg"].degree == 1
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == y.shape
        assert pipe.named_steps["reg"].model_.degree == (1,)

    def test_pipeline_clone(self):
        """clone() on a Pipeline containing BezierSimploidRegressor is safe."""
        from sklearn.base import clone
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([("reg", BezierSimploidRegressor(**self._kwargs))])
        cloned = clone(pipe)
        assert cloned.named_steps["reg"].degree == pipe.named_steps["reg"].degree
        assert cloned is not pipe


class TestGridSearchCVIntegration:
    _kwargs: dict[str, Any] = dict(
        max_epochs=3,
        accelerator="cpu",
        devices=1,
        trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
    )

    def test_gridsearchcv_direct(self):
        """GridSearchCV finds a best degree when searching over BezierSimploidRegressor."""
        from sklearn.model_selection import GridSearchCV

        X, y = _make_data(n_samples=30)
        reg = BezierSimploidRegressor(**self._kwargs)
        gs = GridSearchCV(reg, {"degree": [1, 2]}, cv=2, scoring="r2", refit=True, n_jobs=1)
        gs.fit(X, y)
        assert gs.best_params_["degree"] in [1, 2]
        preds = gs.predict(X)
        assert preds.shape == y.shape

    def test_gridsearchcv_in_pipeline(self):
        """GridSearchCV searches over BezierSimploidRegressor parameters inside a Pipeline."""
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline

        X, y = _make_data(n_samples=30)
        pipe = Pipeline([("reg", BezierSimploidRegressor(**self._kwargs))])
        gs = GridSearchCV(
            pipe,
            {"reg__degree": [1, 2]},
            cv=2,
            scoring="r2",
            refit=True,
            n_jobs=1,
        )
        gs.fit(X, y)
        assert gs.best_params_["reg__degree"] in [1, 2]
        preds = gs.predict(X)
        assert preds.shape == y.shape

    def test_gridsearchcv_cv_results(self):
        """GridSearchCV produces expected cv_results_ structure."""
        from sklearn.model_selection import GridSearchCV

        X, y = _make_data(n_samples=30)
        reg = BezierSimploidRegressor(**self._kwargs)
        gs = GridSearchCV(reg, {"degree": [1, 2]}, cv=2, scoring="r2", n_jobs=1)
        gs.fit(X, y)
        assert "mean_test_score" in gs.cv_results_
        assert len(gs.cv_results_["params"]) == 2


def _random_simplex(n_samples: int, n_params: int, seed: int) -> npt.NDArray[Any]:
    """Return float64 points sampled on the standard simplex."""
    rng = np.random.default_rng(seed)
    raw = rng.exponential(size=(n_samples, n_params))
    points: npt.NDArray[Any] = raw / raw.sum(axis=1, keepdims=True)
    return points


def _fit_reference(X: npt.NDArray[Any], y: npt.NDArray[Any], degree: int, ridge: float = 0.0):
    import torch

    from torch_bsf import fit

    return fit(torch.from_numpy(X), torch.from_numpy(y), degree, simploid_type=(X.shape[1] - 1,), solver="lstsq", ridge=ridge)


class TestLstsqSolver:
    def test_default_solver_is_lstsq(self):
        """Omitting solver solves the least-squares problem exactly, as torch_bsf.fit does."""
        reg = BezierSimploidRegressor()
        assert reg.solver == "lstsq"
        assert reg.ridge == 0.0
        assert reg.simploid_type is None
        params = reg.get_params()
        assert params["solver"] == "lstsq"
        assert params["ridge"] == 0.0

    def test_trainer_solver_still_trains(self):
        """solver='trainer' trains through torch_bsf.fit (float32 model)."""
        import torch

        X, y = _make_data()
        reg = BezierSimploidRegressor(
            degree=2,
            solver="trainer",
            max_epochs=2,
            accelerator="cpu",
            devices=1,
            trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
        )
        reg.fit(X, y)
        assert reg.model_.control_points.dtype == torch.float32
        assert reg.predict(X).dtype == np.float32

    @pytest.mark.parametrize("n_params, degree", [(2, 0), (2, 1), (2, 5), (3, 3), (4, 2)])
    def test_matches_fit_simploid(self, n_params: int, degree: int):
        """solver='lstsq' gives the same model as fit(..., solver='lstsq')."""
        import torch

        X = _random_simplex(60, n_params, seed=n_params * 10 + degree)
        y = np.stack([np.cos(3 * X[:, 0]), X.prod(axis=1), np.exp(X[:, -1])], axis=1)
        reg = BezierSimploidRegressor(degree=degree, solver="lstsq").fit(X, y)
        ref = _fit_reference(X, y, degree)

        X_test = _random_simplex(25, n_params, seed=999)
        expected = ref(torch.from_numpy(X_test)).detach().numpy()
        np.testing.assert_allclose(reg.predict(X_test), expected, rtol=0, atol=1e-8)
        assert reg.model_.degree == (degree,)
        assert reg.model_.simploid_type == (n_params - 1,)
        assert reg.model_.n_params == n_params
        assert reg.n_features_in_ == n_params
        assert reg.n_outputs_ == 3

    @pytest.mark.parametrize("degree", [0, 2])
    def test_zero_simplex_fits_mean(self, degree: int):
        """With a single parameter (a 0-simplex, i.e. a vertex), the least-squares fit is the mean of y."""
        X = np.ones((5, 1))
        y = np.arange(10.0).reshape(5, 2)
        reg = BezierSimploidRegressor(degree=degree, solver="lstsq").fit(X, y)
        assert reg.simploid_type_ == ()
        assert reg.model_.simploid_type == ()
        np.testing.assert_allclose(reg.predict(X[:1]), y.mean(axis=0, keepdims=True), rtol=0, atol=1e-12)

    def test_zero_simplex_negative_degree_raises(self):
        """A negative degree is rejected for a 0-simplex as well."""
        with pytest.raises(ValueError, match="degree"):
            BezierSimploidRegressor(degree=-1, solver="lstsq").fit(np.ones((3, 1)), np.ones((3, 2)))

    def test_one_dimensional_target(self):
        """A 1-D y is fitted as a single output and predicted as a 1-D array."""
        import torch

        X = _random_simplex(30, 3, seed=6)
        y = np.sin(3 * X[:, 0]) + X[:, 1] * X[:, 2]
        reg = BezierSimploidRegressor(degree=3, solver="lstsq").fit(X, y)
        assert reg.n_outputs_ == 1
        X_test = _random_simplex(10, 3, seed=7)
        y_pred = reg.predict(X_test)
        assert y_pred.shape == (10,)
        ref = _fit_reference(X, y.reshape(-1, 1), 3)
        np.testing.assert_allclose(y_pred, ref(torch.from_numpy(X_test)).detach().numpy().ravel(), rtol=0, atol=1e-8)
        assert isinstance(reg.score(X_test, np.sin(3 * X_test[:, 0]) + X_test[:, 1] * X_test[:, 2]), float)
        # Refitting with a 2-D y restores 2-D predictions.
        reg.fit(X, y.reshape(-1, 1))
        assert reg.predict(X_test).shape == (10, 1)

    def test_one_dimensional_target_trainer(self):
        """The trainer solver also accepts a 1-D y."""
        X, y = _make_data(n_values=1)
        reg = BezierSimploidRegressor(
            degree=2,
            solver="trainer",
            max_epochs=2,
            accelerator="cpu",
            devices=1,
            trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False},
        )
        reg.fit(X, y.ravel())
        assert reg.n_outputs_ == 1
        assert reg.predict(X).shape == (X.shape[0],)

    def test_integer_targets_give_floating_model(self):
        """Integer y is fitted in floating point instead of truncating the solution."""
        import torch

        X = _random_simplex(20, 2, seed=5)
        y = (10 * X).round().astype(np.int64)
        reg = BezierSimploidRegressor(degree=1, solver="lstsq").fit(X, y)
        assert reg.model_.control_points.dtype == torch.float64
        ref = _fit_reference(X, y.astype(np.float64), 1)
        np.testing.assert_allclose(reg.predict(X), ref(torch.from_numpy(X)).detach().numpy(), rtol=0, atol=1e-8)

    def test_control_points_match_fit_simploid(self):
        """The control points coincide with fit_simploid's, index by index."""
        X = _random_simplex(40, 3, seed=1)
        y = np.stack([X[:, 0] ** 3, np.sin(X[:, 1])], axis=1)
        reg = BezierSimploidRegressor(degree=4, solver="lstsq").fit(X, y)
        ref = _fit_reference(X, y, 4)
        for alpha in ref.indices():
            np.testing.assert_allclose(
                reg.model_[alpha].detach().numpy(),
                ref[alpha].detach().numpy(),
                rtol=0,
                atol=1e-8,
            )

    def test_high_dimensional_output_matches_exact_least_squares(self):
        """With 256 outputs, lstsq reproduces the exact least-squares fit (cf. issue #347)."""
        import torch

        from torch_bsf.simploidal import bernstein_basis

        n_samples, n_values, degree = 41, 256, 10
        X = _random_simplex(n_samples, 2, seed=0)
        freqs = np.linspace(0.5, 6.0, n_values)
        y = np.sin(np.outer(X[:, 0], freqs)) + np.cos(np.outer(X[:, 1], freqs[::-1]))
        reg = BezierSimploidRegressor(degree=degree, solver="lstsq").fit(X, y)
        assert reg.predict(X).shape == (n_samples, n_values)

        # Independent reference: numpy least squares on the Bernstein basis.
        basis = bernstein_basis((1,), degree, torch.from_numpy(X)).numpy()
        coef, *_ = np.linalg.lstsq(basis, y, rcond=None)
        X_test = _random_simplex(50, 2, seed=1)
        basis_test = bernstein_basis((1,), degree, torch.from_numpy(X_test)).numpy()
        np.testing.assert_allclose(reg.predict(X_test), basis_test @ coef, rtol=0, atol=1e-8)

        ref = _fit_reference(X, y, degree)
        np.testing.assert_allclose(reg.predict(X_test), ref(torch.from_numpy(X_test)).detach().numpy(), rtol=0, atol=1e-8)

    def test_recovers_polynomial_exactly(self):
        """Data generated by a Bézier simplex of the fitted degree is reproduced exactly."""
        X = _random_simplex(30, 3, seed=2)
        y = np.stack([X[:, 0] ** 2, X[:, 0] * X[:, 1], 1.0 - X[:, 2]], axis=1)
        reg = BezierSimploidRegressor(degree=2, solver="lstsq").fit(X, y)
        X_test = _random_simplex(10, 3, seed=3)
        y_test = np.stack([X_test[:, 0] ** 2, X_test[:, 0] * X_test[:, 1], 1.0 - X_test[:, 2]], axis=1)
        np.testing.assert_allclose(reg.predict(X_test), y_test, rtol=0, atol=1e-10)
        assert reg.score(X_test, y_test) == pytest.approx(1.0)

    @pytest.mark.parametrize("ridge", [1e-3, 1.0])
    def test_ridge_matches_fit_simploid(self, ridge: float):
        """ridge is passed through to the Tikhonov-regularized least-squares solve."""
        import torch

        X = _random_simplex(15, 3, seed=4)
        y = np.stack([np.sin(4 * X[:, 0]), X[:, 1] ** 3], axis=1)
        reg = BezierSimploidRegressor(degree=4, solver="lstsq", ridge=ridge).fit(X, y)
        ref = _fit_reference(X, y, 4, ridge=ridge)
        np.testing.assert_allclose(reg.predict(X), ref(torch.from_numpy(X)).detach().numpy(), rtol=0, atol=1e-8)
        unregularized = BezierSimploidRegressor(degree=4, solver="lstsq").fit(X, y)
        assert reg.model_.control_points.norm() < unregularized.model_.control_points.norm()

    def test_float32_input_keeps_dtype(self):
        """The fitted model keeps the dtype of y; float32 data give float32 predictions."""
        import torch

        X, y = _make_data()
        reg = BezierSimploidRegressor(degree=2, solver="lstsq").fit(X, y)
        assert reg.model_.control_points.dtype == torch.float32
        assert reg.predict(X).dtype == np.float32
        X64 = X.astype(np.float64)
        y64 = y.astype(np.float64)
        reg64 = BezierSimploidRegressor(degree=2, solver="lstsq").fit(X64, y64)
        assert reg64.model_.control_points.dtype == torch.float64
        np.testing.assert_allclose(reg.predict(X), reg64.predict(X64), rtol=0, atol=1e-5)

    def test_get_set_params_clone(self):
        """solver and ridge work with get_params, set_params and clone."""
        from sklearn.base import clone

        reg = BezierSimploidRegressor(degree=2, solver="lstsq", ridge=0.5)
        params = reg.get_params()
        assert params["solver"] == "lstsq"
        assert params["ridge"] == 0.5
        assert reg.set_params(solver="trainer", ridge=0.0, simploid_type=(1, 1)) is reg
        assert reg.simploid_type == (1, 1)
        reg.set_params(simploid_type=None)
        assert reg.solver == "trainer"
        assert reg.ridge == 0.0
        reg.set_params(solver="lstsq", ridge=0.25)
        cloned = clone(reg)
        assert cloned is not reg
        assert cloned.solver == "lstsq"
        assert cloned.ridge == 0.25
        X, y = _make_data()
        cloned.fit(X, y)
        assert not hasattr(reg, "model_")

    def test_gridsearchcv_over_solver_options(self):
        """GridSearchCV can search over degree and ridge with the lstsq solver."""
        from sklearn.model_selection import GridSearchCV

        X, y = _make_data(n_samples=30)
        gs = GridSearchCV(
            BezierSimploidRegressor(solver="lstsq"),
            {"degree": [1, 2, 3], "ridge": [0.0, 1e-3]},
            cv=3,
            scoring="r2",
            n_jobs=1,
        )
        gs.fit(X, y)
        assert gs.best_params_["degree"] in [1, 2, 3]
        assert gs.predict(X).shape == y.shape

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"solver": "trainer", "ridge": 0.1}, "ridge"),
            ({"freeze": [[2, 0, 0]]}, "init is required"),
            ({"smoothness_weight": -0.1}, "smoothness_weight"),
            ({"simploid_type": (1, 1)}, "params must have shape"),
            ({"solver": "newton"}, "Unknown solver"),
        ],
    )
    def test_invalid_option_combinations_raise(self, kwargs: dict[str, Any], match: str):
        """Invalid options raise ValueError in fit."""
        X, y = _make_data()
        reg = BezierSimploidRegressor(degree=2, **kwargs)
        with pytest.raises(ValueError, match=match):
            reg.fit(X, y)

    def test_negative_ridge_raises(self):
        """A negative ridge is rejected by the least-squares solver."""
        X, y = _make_data()
        with pytest.raises(ValueError, match="ridge"):
            BezierSimploidRegressor(degree=2, solver="lstsq", ridge=-1.0).fit(X, y)


class TestSimploidsAndConstraints:
    def test_simploid_type(self):
        """A Bézier simploid is fitted to parameters on a simploid."""
        import torch

        from torch_bsf.simploidal import sample_simploid

        t = sample_simploid((2, 1), 40, seed=0).double()
        X = torch.cat([t[:, :3] / t[:, :3].sum(1, keepdim=True), t[:, 3:] / t[:, 3:].sum(1, keepdim=True)], 1).numpy()
        y = np.stack([X[:, 0] * X[:, 3], X[:, 1] ** 2 * X[:, 4]], axis=1)
        reg = BezierSimploidRegressor(degree=(1, 1), simploid_type=(2, 1)).fit(X[:, :], y[:, :1])
        assert reg.simploid_type_ == (2, 1)
        np.testing.assert_allclose(reg.predict(X), y[:, :1], atol=1e-10)
        reg = BezierSimploidRegressor(degree=2, simploid_type=[2, 1]).fit(X, y)
        assert reg.model_.degree == (2, 2)
        np.testing.assert_allclose(reg.predict(X), y, atol=1e-10)
        expected = __import__("torch_bsf").fit(torch.from_numpy(X), torch.from_numpy(y), 2, simploid_type=(2, 1))
        np.testing.assert_allclose(reg.model_.control_points.detach().numpy(), expected.control_points.detach().numpy())

    def test_predict_checks_the_number_of_features(self):
        X, y = _make_data()
        reg = BezierSimploidRegressor(degree=2).fit(X, y)
        with pytest.raises(ValueError, match="features"):
            reg.predict(X[:, :2])

    @pytest.mark.parametrize("solver", ["lstsq", "trainer"])
    def test_init_freeze_and_smoothness(self, solver: Solver):
        """init, freeze and smoothness_weight work with both solvers."""
        from torch_bsf import BezierSimploid

        X = _random_simplex(30, 3, seed=0)
        y = X[:, :2] ** 2
        init = BezierSimploid.zeros((2,), 2, n_values=2).double()
        reg = BezierSimploidRegressor(
            degree=2,
            solver=solver,
            init=init,
            freeze=[(2, 0, 0)],
            smoothness_weight=0.01,
            max_epochs=2,
            accelerator="cpu",
            trainer_kwargs={"enable_progress_bar": False, "enable_model_summary": False, "logger": False},
        ).fit(X, y)
        assert reg.model_[(2, 0, 0)].tolist() == [0.0, 0.0]

    @pytest.mark.parametrize("normalize", ["max", "std"])
    def test_normalize_predicts_in_original_scale(self, normalize: NormalizeType):
        X = _random_simplex(30, 2, seed=0)
        y = 1000.0 * X[:, :1] + 5.0
        reg = BezierSimploidRegressor(degree=1, normalize=normalize).fit(X, y)
        np.testing.assert_allclose(reg.predict(X), y, atol=1e-8)
