Fixed controls
==============

Specify values directly rather than supplying an initial regression model::

    from torch_bsf import FixedControls, fit
    row = space.control_index((2, 0), cell=0)
    result = fit(space, samples, fixed=FixedControls({row: [1., 2.]}))

Indices refer to global shared coefficient rows. All cells incident on a fixed
shared control inherit the same value. Constraints participate consistently in
data, ridge and smoothness terms; all-fixed problems need no factorization.

For cell-local references use ``FixedControls.from_cells(space, [(cell, index, value), ...])``.
Shared references normalize to one global row; contradictory values raise an error.
