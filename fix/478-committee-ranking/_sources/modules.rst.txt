API reference guide
===================

The Python reference describes the public objects and their arguments.
Begin with :doc:`quickstart` for a complete fit, or :doc:`advanced` for
practical choices and interpretation. The API pages supply exact signatures
once you know which operation you need.

Most workflows start with ``BezierSpace``, ``Samples`` and ``fit``. Use
``Points`` when cell identity must travel with coordinates, and use the
returned ``FitResult.model`` for prediction. The common types live in
``torch_bsf.core`` and are also available from the package root.

For numerical methods, consult ``fitting`` and ``solvers`` for regression,
or ``optimization`` for differentiable costs. Sampling, validation and
active learning build on the same space and model. Plotting, scikit-learn,
Lightning and MLflow integrations are optional; install the relevant extra
before executing their examples.

``geometry`` describes mutable topology construction, while a compiled
``BezierSpace`` records immutable geometry and coefficient layout. The
archive interface in ``io`` preserves that layout with model coefficients.
Private modules beginning with an underscore are implementation details.

.. toctree::
   :maxdepth: 4

   torch_bsf
