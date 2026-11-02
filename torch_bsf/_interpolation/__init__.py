"""Private implementation of :mod:`torch_bsf.interpolation`, which re-exports the public names.

Modules, each importing only those above it:

* ``exact``: exact and power-of-two arithmetic, and argument checks;
* ``mesh``: :class:`~torch_bsf.interpolation.SimplicialMesh`;
* ``refinement``: :func:`~torch_bsf.interpolation.lattice_mesh` and :func:`~torch_bsf.interpolation.refine`;
* ``hermite``: :func:`~torch_bsf.interpolation.estimate_gradients` and
  :func:`~torch_bsf.interpolation.hermite_interpolant`;
* ``constructions``: the splits and continuities of :func:`~torch_bsf.interpolation.macro_interpolant` and the
  Clough–Tocher C1 construction;
* ``macro``: :class:`~torch_bsf.interpolation.MacroInterpolant`, :func:`~torch_bsf.interpolation.macro_interpolant`
  and :func:`~torch_bsf.interpolation.refinement_indicator`.
"""
