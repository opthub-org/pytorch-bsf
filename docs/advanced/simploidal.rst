Single and assembled domains
============================

The same model and fitting API handles simplices, products and glued cells::

    from torch_bsf import BezierSpace, Samples, SimploidalSet, fit
    from torch_bsf.sampling import sample

    triangle_space = BezierSpace.simplex(2, degree=3)
    prism_space = BezierSpace.simploid((2, 1), degree=(3, 2))

    domain = SimploidalSet()
    triangle = domain.add_simplex(2)
    square = domain.add_simploid((1, 1))
    domain.identify(triangle.face(0, 0), square.face(0, 0))
    space = BezierSpace.on_set(domain, degree=2)
    t1 = sample(triangle_space, 30, seed=1).parameters
    t2 = sample(BezierSpace.simploid((1, 1), degree=2), 30, seed=2).parameters
    observations = Samples.from_cells(space, [
        (triangle, t1, t1[:, :1]), (square, t2, t2[:, :1])])
    result = fit(space, observations)

Shared face coefficients are stored once and ensure C0 continuity. The space
snapshots the topology builder; subsequent builder mutations do not alter it.
``space.cell_id(handle)`` resolves original builder handles to stable IDs.
``space.control_index(index, cell=handle)`` resolves local indices to shared rows.

``Points(parameters, cell_ids)`` preserves mixed-cell prediction order.
Coordinates are padded to ``space.width``; only each cell's active coordinates
are used. Each simplex factor has nonnegative barycentric coordinates summing
to one. Saved models preserve stable IDs rather than mutable builder handles.

Homology
--------

The topology of a simploidal set can be checked after it is built by
computing its homology (Section 4.4 of :cite:t:`peltier2018simploidal`).  The
chain group :math:`C_k` is spanned by the **non-degenerate** simploids of
dimension :math:`k`, and the boundary of a simploid :math:`\sigma` of type
:math:`(a_1, \ldots, a_n)` follows the sign rule of a product of simplices:

.. math::

   \partial \sigma = \sum_{i=1}^n \sum_{j=0}^{a_i} (-1)^{a_1 + \cdots + a_{i-1} + j} d^i_j \sigma,

where the terms whose face :math:`d^i_j \sigma` is degenerate are dropped.

* :meth:`~torch_bsf.geometry.SimploidalSet.boundary_matrix` returns the
  integer matrix of :math:`\partial_k`, with rows ``simploids(k - 1)`` and
  columns ``simploids(k)``; it satisfies
  ``boundary_matrix(k) @ boundary_matrix(k + 1) == 0``.
* :meth:`~torch_bsf.geometry.SimploidalSet.betti_numbers` returns the Betti
  numbers :math:`\beta_0, \ldots, \beta_{\dim}`, the ranks of the homology
  groups.  The ranks are computed exactly by integer elimination, over the
  rationals (``field="Q"``, default) or over :math:`\mathbb Z / 2`
  (``field="Z2"``); the two differ only for spaces whose integral homology has
  torsion, such as non-orientable surfaces.

.. testcode::
   :pyversion: >= 3.10, < 3.15

   from torch_bsf.geometry import SimploidalSet

   # A torus: a square whose opposite edges are identified
   torus = SimploidalSet()
   square = torus.add_simploid((1, 1))
   torus.identify(square.face(0, 0), square.face(0, 1))
   torus.identify(square.face(1, 0), square.face(1, 1))
   print(torus.betti_numbers())

   # A sphere: a square whose two edges face(1, *) are collapsed into
   # vertices and whose two other edges are identified
   sphere = SimploidalSet()
   square = sphere.add_simploid((1, 1))
   for j in range(2):
       edge = square.face(1, j)
       sphere.degenerate(edge, edge.face(0, 0), 0, -1)
   sphere.identify(square.face(0, 0), square.face(0, 1))
   print(sphere.betti_numbers())

   # A projective plane: two triangles glued along their edges
   rp2 = SimploidalSet()
   t1, t2 = rp2.add_simplex(2), rp2.add_simplex(2)
   rp2.identify(t1.face(0, 0), t2.face(0, 0))
   rp2.identify(t1.face(0, 1), t2.face(0, 2))
   rp2.identify(t1.face(0, 2), t2.face(0, 1))
   print(rp2.betti_numbers(), rp2.betti_numbers(field="Z2"))

.. testoutput::
   :pyversion: >= 3.10, < 3.15

   [1, 2, 1]
   [1, 0, 1]
   [1, 0, 0] [1, 1, 1]


Use ``Points.from_cells(space, [(cell, parameters), ...])`` to pack mixed-cell
prediction batches without padding. Multi-cell models require explicit Points
and never infer a cell from an untagged tensor.
