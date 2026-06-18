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
