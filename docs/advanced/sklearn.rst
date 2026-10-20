Scikit-learn Integration
========================

PyTorch-BSF provides a high-level wrapper compatible with the `Scikit-learn <https://scikit-learn.org/>`_ estimator API. This allows you to integrate Bézier simplex fitting into standard machine learning workflows, including pipelines and cross-validation tools.

The ``BezierSimploidRegressor``
-------------------------------

The ``BezierSimploidRegressor`` class inherits from ``sklearn.base.BaseEstimator`` and ``sklearn.base.RegressorMixin``. It supports the standard ``fit()``, ``predict()``, and ``score()`` methods, and fits the model with :func:`torch_bsf.fit`.

.. code-block:: python

   import numpy as np
   from torch_bsf.sklearn import BezierSimploidRegressor

   # Your data (parameters must be on a simplex)
   X = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
   y = np.array([[1.0], [0.75], [0.0]])

   # Initialize and fit
   reg = BezierSimploidRegressor(degree=2)
   reg.fit(X, y)

   # Predict
   y_pred = reg.predict(X)

By default, each row of ``X`` is a point on a simplex with ``n_features`` vertices, and a Bézier simplex is fitted (a single feature gives a constant model).  The fitted model is available as ``reg.model_``.

Choosing a solver
-----------------

The ``solver`` parameter selects how the control points are computed:

``solver="lstsq"`` (default)
   Since a Bézier simplex is linear in its control points, the squared error is minimized exactly by
   linear least squares on the Bernstein basis, solved in double precision.  This is fast and accurate,
   especially for high-dimensional outputs (e.g., hundreds of values per sample), where training may
   stop far from the optimum.  The optional ``ridge`` parameter adds the Tikhonov penalty
   :math:`\mathrm{ridge}\,\|P\|^2` on the control points :math:`P`; without it, under-determined
   problems (fewer samples than control points) get the minimum-norm solution.  The fitted model keeps
   the floating dtype of ``y`` (``float64`` for ``float64`` or integer targets).

``solver="trainer"``
   Trains the control points by gradient-based optimization with ``lightning.pytorch.Trainer`` for up to
   ``max_epochs`` epochs in single precision.  Use it for minibatches and accelerators
   (``batch_size``, ``max_epochs``, ``accelerator``, ``devices``, ``precision`` and ``trainer_kwargs``).

.. testcode::
   :pyversion: >= 3.10, < 3.15

   import numpy as np
   from torch_bsf.sklearn import BezierSimploidRegressor

   rng = np.random.default_rng(0)
   raw = rng.random((41, 2))
   X = raw / raw.sum(axis=1, keepdims=True)                  # 41 points on a segment
   y = np.sin(np.outer(X[:, 0], np.linspace(1, 6, 256)))    # 256-dimensional values

   reg = BezierSimploidRegressor(degree=10, ridge=1e-12)
   reg.fit(X, y)
   print(reg.predict(X).shape)
   print(reg.score(X, y) > 0.999)

.. testoutput::
   :pyversion: >= 3.10, < 3.15

   (41, 256)
   True

``init``, ``freeze``, ``smoothness_weight`` and ``normalize`` work with both solvers; the predictions
are in the scale of ``y`` whatever ``normalize`` is.  With ``solver="lstsq"``, the trainer-only options
listed above are ignored, and a nonzero ``ridge`` with ``solver="trainer"`` raises ``ValueError``.

All parameters work with ``get_params()``, ``set_params()`` and ``clone()``, so they can be searched by
``GridSearchCV`` like any other hyperparameter (e.g., ``{"ridge": [0.0, 1e-6, 1e-3]}``).

Bézier simploids
----------------

To fit a Bézier simploid, i.e., a tensor product of Bézier simplices (see :doc:`simploidal`), pass its
type as ``simploid_type``.  Each row of ``X`` is then the concatenation of the barycentric coordinates of
every factor, and ``degree`` is the degree of every factor or one degree per factor.  For example, on a
prism (a triangle times a segment):

.. testcode::
   :pyversion: >= 3.10, < 3.15

   import numpy as np
   from torch_bsf.simploidal import sample_simploid
   from torch_bsf.sklearn import BezierSimploidRegressor

   X = sample_simploid((2, 1), 40, seed=0).double().numpy()  # 3 + 2 columns
   y = X[:, :1] ** 2 * X[:, 3:4]

   reg = BezierSimploidRegressor(degree=(2, 1), simploid_type=(2, 1)).fit(X, y)
   print(reg.simploid_type_, reg.model_.degree)
   print(np.allclose(reg.predict(X), y, atol=1e-6))

.. testoutput::
   :pyversion: >= 3.10, < 3.15

   (2, 1) (2, 1)
   True

Integration with Pipelines
--------------------------

You can include ``BezierSimploidRegressor`` in a Scikit-learn ``Pipeline``. This is particularly useful if you need to perform data preprocessing (though note that Bézier simplex fitting usually expects parameters to be on a simplex).

.. code-block:: python

   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler

   pipe = Pipeline([
       ('scaler', StandardScaler()),
       ('bsf', BezierSimploidRegressor(degree=3))
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

   grid = GridSearchCV(BezierSimploidRegressor(), param_grid, cv=5)
   grid.fit(X, y)

   print(f"Best parameters: {grid.best_params_}")

API Reference
-------------

See the `API Documentation <../modules.html#torch_bsf.sklearn.BezierSimploidRegressor>`_ for a full list of supported parameters.
