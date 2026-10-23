"""Estimator integration with cloning, weighting and result evidence."""

import numpy as np
import pytest

pytest.importorskip("sklearn")
from sklearn.base import clone  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402

from torch_bsf.sklearn import BezierRegressor  # noqa: E402


def test_clone_and_grid_search():
    t = np.linspace(0, 1, 24)
    x = np.column_stack([t, 1 - t])
    y = t**2
    original = BezierRegressor(degree=2)
    assert clone(original).get_params() == original.get_params()
    searched = GridSearchCV(original, {"degree": [1, 2]}, cv=3).fit(x, y)
    fitted = searched.best_estimator_
    assert fitted.degree == 2
    assert fitted.result_.success
    assert fitted.model_ is fitted.result_.model
    assert fitted.diagnostics_ is fitted.result_.diagnostics
    assert np.allclose(fitted.predict(x), y)
    assert fitted.score(x, y) > 0.99999


def test_weighted_and_multioutput():
    x = np.eye(2)
    fitted = BezierRegressor(degree=1).fit(x, x, sample_weight=[1, 2])
    assert fitted.predict(x).shape == (2, 2)
    assert fitted.diagnostics_.weight_sum == 3


def test_one_feature_constant_regression():
    x = np.arange(6).reshape(-1, 1)
    y = np.arange(6.0)
    fitted = BezierRegressor().fit(x, y, sample_weight=[1, 1, 1, 1, 1, 2])
    assert fitted.model_.space.cells[0].dimensions == ()
    assert np.allclose(fitted.predict([[100.0], [-20.0]]), 20 / 7)


def test_regressor_rejects_inconsistent_feature_shapes():
    x = np.eye(2)
    with pytest.raises(ValueError):
        BezierRegressor(dimensions=()).fit(x, [1.0, 2.0])
    with pytest.raises(ValueError):
        BezierRegressor(dimensions=(2,)).fit(x, [1.0, 2.0])
    fitted = BezierRegressor(degree=1).fit(x, [1.0, 2.0])
    with pytest.raises(ValueError):
        fitted.predict([[1.0, 0.0, 0.0]])


def test_constrained_estimator_can_be_cloned_independently():
    from torch_bsf import FixedControls

    fixed = FixedControls({0: [3.0]})
    original = BezierRegressor(degree=1, fixed=fixed)
    copied = clone(original)
    fixed.values[0][0] = 7
    assert copied.fixed is not None and original.fixed is not None
    assert copied.fixed.values[0].item() == 3
    assert original.fixed.values[0].item() == 7
    assert copied.fit(np.eye(2), [0.0, 0.0]).result_.success
