Scikit-learn integration
========================

Install ``pytorch-bsf[sklearn]`` and use the single estimator::

    from torch_bsf.sklearn import BezierRegressor
    from torch_bsf import QuadraticPenalty
    regressor = BezierRegressor(degree=3, penalty=QuadraticPenalty(ridge=.001))
    regressor.fit(X, y, sample_weight=weights)
    predictions = regressor.predict(X)
    diagnostics = regressor.diagnostics_

``result_`` contains all fit evidence and ``model_`` is the common BezierModel.
Constructor arguments support cloning and GridSearchCV. Product domains use
``dimensions=(2, 1)``. Every row of X contains the concatenated barycentric
coordinates. Use the Python Points API for mixed-cell observations.

A single input feature defaults to a constant model: the feature is a dummy
column and its numeric value is ignored. The fitted coefficient is the
weighted output mean for unregularized regression.
