Output scaling
==============

Core fitting defines residuals and penalties in original output units. It has
no implicit normalization switch. Explicit transformations remain available
in ``torch_bsf.preprocessing`` when the application requires a different scale.
Fit transformed outputs and explicitly invert the transformation on predictions.

For unregularized independent output columns, affine output scaling does not
change the mathematical least-squares predictions after inverse scaling.
With penalties, changing output coordinates changes the objective; make this
choice explicitly and validate it with held-out errors.
