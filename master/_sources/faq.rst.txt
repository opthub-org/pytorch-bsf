Frequently asked questions
==========================

Do I need Conda, Lightning or MLflow?
-------------------------------------
No. Install the core with pip. Integrations are optional and explicit.

How do I predict?
-----------------
Call ``result.model(points)``. ``FitResult`` itself is not callable.

Does solver success imply identifiable coefficients?
----------------------------------------------------
No. Inspect data rank and condition diagnostics. Regularization can stabilize
a system without supplying missing observational information. See
:doc:`advanced/regularization`.

How do I choose a degree?
-------------------------
Use ``select_degree`` with explicit candidates and held-out folds, then
explicitly refit the chosen space. See :doc:`advanced/auto_degree`.

Where did the old APIs go?
--------------------------
See :doc:`migration`. The project is pre-1.0 and this release intentionally
consolidates model classes, solvers, sampling, plotting and persistence.
