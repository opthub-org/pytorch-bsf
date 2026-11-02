r"""Shape-preserving cubic Bézier interpolation on simplicial complexes ("simplicial PCHIP").

A :class:`SimplicialMesh` glues user-supplied simplices of :math:`\mathbb{R}^d` into one
:class:`~torch_bsf.core.BezierSpace`, so that simplices sharing a face share the controls on it.
:func:`hermite_interpolant` fills that space with a cubic Bézier model that interpolates vertex data:

* a vertex control is the datum :math:`f_i`;
* the control on edge :math:`i \to j` next to vertex :math:`i` is :math:`f_i + g_i \cdot (x_j - x_i) / 3`,
  with a vertex gradient :math:`g_i`;
* the control at the centre of face :math:`(i, j, l)` is the sum of the six edge controls of the face
  divided by four, minus :math:`(f_i + f_j + f_l) / 6`, which reproduces quadratic data exactly.

Every control depends only on the data of the edge or face it lies on, so the interpolant is
continuous across shared faces. With ``clamp=True`` every edge (face) control is clipped into the
range of the data at the vertices of that edge (face). By the convex-hull property, the prediction
on each simplex then stays within the range of its vertex data, which bounds the sup error by the
modulus of continuity of the data at the mesh size. In one dimension the edge limiter restricts each
endpoint slope, separately on each incident interval, to the interval between zero and three times
the secant slope. The normalized slopes lie in the square :math:`[0, 3]^2` of the Fritsch–Carlson
monotonicity region :cite:p:`fritsch1980monotone`. The interpolant is C0, but need not be C1, even
on a chain of intervals. It agrees with PCHIP when supplied with its vertex slopes (the
Fritsch–Butland weighted harmonic means at interior vertices :cite:p:`fritsch1984method`, with
PCHIP's one-sided endpoint rule); the default :func:`estimate_gradients` uses different slopes.
"""

# The implementation lives in the private modules of torch_bsf._interpolation (see its docstring for the layout);
# this module is the public path. The private names imported "as" themselves are re-exported for the tests.
from torch_bsf._interpolation.constructions import C0, C1, CloughTocher, NoSplit
from torch_bsf._interpolation.constructions import _nearest_feasible as _nearest_feasible
from torch_bsf._interpolation.exact import _overlapping_boxes as _overlapping_boxes
from torch_bsf._interpolation.exact import _rational_inverse as _rational_inverse
from torch_bsf._interpolation.exact import _split_difference as _split_difference
from torch_bsf._interpolation.exact import _times_power_of_two as _times_power_of_two
from torch_bsf._interpolation.hermite import estimate_gradients, hermite_interpolant
from torch_bsf._interpolation.macro import MacroInterpolant, macro_interpolant, refinement_indicator
from torch_bsf._interpolation.macro import _refinement_scores as _refinement_scores
from torch_bsf._interpolation.mesh import SimplicialMesh
from torch_bsf._interpolation.mesh import _simplicial_set as _simplicial_set
from torch_bsf._interpolation.mesh import _Tagged as _Tagged
from torch_bsf._interpolation.refinement import _Bisection as _Bisection
from torch_bsf._interpolation.refinement import lattice_mesh, refine

__all__ = [
    "C0",
    "C1",
    "CloughTocher",
    "MacroInterpolant",
    "NoSplit",
    "SimplicialMesh",
    "estimate_gradients",
    "hermite_interpolant",
    "lattice_mesh",
    "macro_interpolant",
    "refine",
    "refinement_indicator",
]

# Present the public objects under this module, their public path, as before the split: this is the name used by
# the documentation (e.g. in rendered signatures) and by pickles.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name
