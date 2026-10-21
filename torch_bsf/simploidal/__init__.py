r"""Simploidal sets and simploidal Bézier spaces.

This module implements the simploidal sets of :cite:t:`peltier2018simploidal`
and the fitting of the Bézier spaces they carry.

A *simploid* of type :math:`(a_1, \ldots, a_n)` is a cartesian product
:math:`\Delta^{a_1} \times \cdots \times \Delta^{a_n}` of simplices; simplices
(type :math:`(a)`) and cubes (type :math:`(1, \ldots, 1)`) are special cases.
A *Bézier simploid* of type :math:`(a_1, \ldots, a_n)` and degree
:math:`(d_1, \ldots, d_n)` is the tensor product of Bézier simplices

.. math::

   B(\mathbf u^1, \ldots, \mathbf u^n)
   = \sum_{\boldsymbol\alpha^1 \in \Gamma^{a_1}_{d_1}} \cdots \sum_{\boldsymbol\alpha^n \in \Gamma^{a_n}_{d_n}}
     P_{(\boldsymbol\alpha^1, \ldots, \boldsymbol\alpha^n)}
     \prod_{i=1}^n B^{d_i}_{\boldsymbol\alpha^i}(\mathbf u^i),

where each :math:`\mathbf u^i` is a point of the standard :math:`a_i`-simplex
given in barycentric coordinates.  Throughout this module the parameter of a
simploid is the concatenation :math:`(\mathbf u^1, \ldots, \mathbf u^n)`, a
vector of length :math:`\sum_i (a_i + 1)`.

A *simploidal set* (:class:`SimploidalSet`) is an assembly of simploids glued
along common faces, described by face operators :math:`d^i_j` and degeneracy
operators :math:`s^i_j`.  A *simploidal Bézier space*
(:class:`SimploidalBezierSpace`) associates a Bézier simploid with every
simploid of a simploidal set.  As in Section 4.3 of the paper, only the
*proper* control points of each non-degenerate simploid are stored, so control
points on shared faces are stored once and the space is continuous across
them by construction.

Package layout
--------------
The public names below are re-exported from the submodules:

* ``torch_bsf.simploidal.combinatorics``: simploid types, control point indices, Bernstein bases and sampling;
* ``torch_bsf.simploidal.sets``: :class:`Simploid` and :class:`SimploidalSet`;
* ``torch_bsf.simploidal.bezier``: :class:`BezierSimploid`;
* ``torch_bsf.simploidal.space``: :class:`SimploidalBezierSpace`;
* ``torch_bsf.simploidal.fitting``: :func:`torch_bsf.fit`, :func:`torch_bsf.fit_kfold` and :func:`fit_simploidal_set`;
* ``torch_bsf.simploidal.sgd``: :func:`torch_bsf.fit_sgd`, which fits either model to the minimizers of a loss,
  e.g., to a Pareto set :cite:p:`hikima2025stochastic`;
* ``torch_bsf.simploidal.io``: :func:`save` and :func:`load`.

Both models are plain :class:`torch.nn.Module` objects that are linear in their only parameter;
the solvers of ``torch_bsf.simploidal.fitting`` rely on this alone.

Indexing convention
-------------------
Factor indices are 0-based, whereas the paper uses 1-based ones:

* ``face(i, j)`` is the paper's :math:`d^{i+1}_j` (remove vertex ``j`` of factor ``i``);
* ``degeneracy(i, j)`` with ``j >= 0`` is :math:`s^{i+1}_j` (repeat vertex ``j`` of factor ``i``);
* ``degeneracy(i, -1)`` is :math:`s^i_{-1}` (insert a degenerate edge as the new factor ``i``).

Examples
--------
Glue a triangle and a square along an edge and fit the resulting space:

>>> import torch
>>> from torch_bsf.simploidal import SimploidalSet, fit_simploidal_set, sample_simploid
>>> S = SimploidalSet()
>>> triangle = S.add_simplex(2)
>>> square = S.add_simploid((1, 1))
>>> edge = S.identify(triangle.face(0, 0), square.face(0, 0))
>>> S.check()
>>> t1, t2 = sample_simploid((2,), 50, seed=0), sample_simploid((1, 1), 50, seed=1)
>>> space = fit_simploidal_set(S, [(triangle, t1, t1[:, :1]), (square, t2, t2[:, 1:2])], degree=2)
>>> space.n_values
1
"""

from torch_bsf.simploidal.bezier import BezierSimploid
from torch_bsf.simploidal.combinatorics import (
    DegeneracyMap,
    SimploidIndex,
    SimploidType,
    bernstein_basis,
    sample_simploid,
    simploid_indices,
    simploid_n_params,
)
from torch_bsf.simploidal.fitting import Solver, fit_simploidal_set
from torch_bsf.simploidal.io import load, save
from torch_bsf.simploidal.sets import Simploid, SimploidalSet
from torch_bsf.simploidal.sgd import SGDLoss, fit_sgd, gram_matrix, linear_scalarization, sample_domain
from torch_bsf.simploidal.space import DegreeAssignment, SimploidalBezierSpace

__all__ = [
    "BezierSimploid",
    "DegeneracyMap",
    "DegreeAssignment",
    "Simploid",
    "SimploidalBezierSpace",
    "SimploidalSet",
    "SimploidIndex",
    "SimploidType",
    "SGDLoss",
    "Solver",
    "bernstein_basis",
    "fit_sgd",
    "fit_simploidal_set",
    "gram_matrix",
    "linear_scalarization",
    "load",
    "sample_domain",
    "sample_simploid",
    "save",
    "simploid_indices",
    "simploid_n_params",
]

# Present the public classes under the package, as their public path: this is the name used by the
# documentation and by pickles (e.g. models logged with MLflow).
for _cls in (BezierSimploid, Simploid, SimploidalBezierSpace, SimploidalSet):
    _cls.__module__ = __name__
del _cls
