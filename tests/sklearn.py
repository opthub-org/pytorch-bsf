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
