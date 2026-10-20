Scikit-learn Integration
========================

PyTorch-BSF provides a high-level wrapper compatible with the `Scikit-learn <https://scikit-learn.org/>`_ estimator API. This allows you to integrate Bézier simplex fitting into standard machine learning workflows, including pipelines and cross-validation tools.

The ``BezierSimplexRegressor``
------------------------------

The ``BezierSimplexRegressor`` class inherits from ``sklearn.base.BaseEstimator`` and ``sklearn.base.RegressorMixin``. It supports the standard ``fit()``, ``predict()``, and ``score()`` methods.

.. code-block:: python

   import numpy as np
   from torch_bsf.sklearn import BezierSimplexRegressor

   # Your data (parameters must be on a simplex)
   X = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
   y = np.array([[1.0], [0.75], [0.0]])

   # Initialize and fit
   reg = BezierSimplexRegressor(degree=3, max_epochs=100)
   reg.fit(X, y)

   # Predict
   y_pred = reg.predict(X)

Choosing a solver
-----------------

The ``solver`` parameter selects how the control points are computed:

``solver="trainer"`` (default)
   Trains the control points with :func:`torch_bsf.fit`, i.e., gradient-based optimization with
   ``lightning.pytorch.Trainer`` for up to ``max_epochs`` epochs.  This is the original behavior and
   supports ``init``, ``freeze``, ``smoothness_weight``, minibatches and accelerators.

``solver="lstsq"``
   Since a Bézier simplex is linear in its control points, the squared error is minimized exactly by
   linear least squares on the Bernstein basis, solved in double precision.  The result is the same as
   :func:`torch_bsf.simploidal.fit_simploid` with ``simploid_type=(n_params - 1,)`` and
   ``solver="lstsq"``.  This is much faster and more accurate than training, especially for
   high-dimensional outputs (e.g., hundreds of values per sample), where training may stop far from the
   optimum.  The optional ``ridge`` parameter adds the Tikhonov penalty
   :math:`\mathrm{ridge}\,\|P\|^2` on the control points :math:`P`; without it, under-determined
   problems (fewer samples than control points) get the minimum-norm solution.

.. testcode::
   :pyversion: >= 3.10, < 3.15

   import numpy as np
   from torch_bsf.sklearn import BezierSimplexRegressor

   rng = np.random.default_rng(0)
   raw = rng.random((41, 2))
   X = raw / raw.sum(axis=1, keepdims=True)                  # 41 points on a segment
   y = np.sin(np.outer(X[:, 0], np.linspace(1, 6, 256)))    # 256-dimensional values

   reg = BezierSimplexRegressor(degree=10, solver="lstsq", ridge=1e-12)
   reg.fit(X, y)
   print(reg.predict(X).shape)
   print(reg.score(X, y) > 0.999)

.. testoutput::
   :pyversion: >= 3.10, < 3.15

   (41, 256)
   True

With ``solver="lstsq"``, the trainer-only options (``batch_size``, ``max_epochs``, ``accelerator``,
``devices``, ``precision`` and ``trainer_kwargs``) are ignored, and ``fit()`` raises ``ValueError`` if
``init``, ``freeze`` or a nonzero ``smoothness_weight`` is given, since the exact solution does not
depend on an initial guess and these constraints are not part of the least-squares problem.
Conversely, a nonzero ``ridge`` with ``solver="trainer"`` raises ``ValueError``.  The fitted model
keeps the floating dtype of ``y`` (``float64`` for ``float64`` or integer targets).

Both parameters work with ``get_params()``, ``set_params()`` and ``clone()``, so they can be searched by
``GridSearchCV`` like any other hyperparameter (e.g., ``{"solver": ["lstsq"], "ridge": [0.0, 1e-6, 1e-3]}``).

Integration with Pipelines
--------------------------

You can include ``BezierSimplexRegressor`` in a Scikit-learn ``Pipeline``. This is particularly useful if you need to perform data preprocessing (though note that Bézier simplex fitting usually expects parameters to be on a simplex).

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler

   pipe = Pipeline([
       ('scaler', StandardScaler()),
       ('bsf', BezierSimplexRegressor(degree=3, max_epochs=200))
   ])

   pipe.fit(X, y)

Using GridSearchCV
------------------

Since it's a standard estimator, you can use ``GridSearchCV`` to find the optimal hyperparameters, such as the degree or smoothness weight.

.. code-block:: python

   from sklearn.model_selection import GridSearchCV

   param_grid = {
       'degree': [2, 3, 4, 5],
       'smoothness_weight': [0.0, 0.01, 0.1]
   }

   grid = GridSearchCV(BezierSimplexRegressor(max_epochs=100), param_grid, cv=5)
   grid.fit(X, y)

   print(f"Best parameters: {grid.best_params_}")

API Reference
-------------

See the `API Documentation <../modules.html#torch_bsf.sklearn.BezierSimplexRegressor>`_ for a full list of supported parameters.
