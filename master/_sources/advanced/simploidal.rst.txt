Simploidal Sets
===============

A Bézier simplex has a single simplex as its parameter domain.  Some data are
better described on a **product of simplices** — a prism (triangle ×
segment), a square (segment × segment) — or on **several cells glued along
common faces**.  The ``torch_bsf.simploidal`` module implements the
*simploidal sets* of :cite:t:`peltier2018simploidal`, which describe exactly
such assemblies, together with the Bézier spaces they carry and their fitting.

Simploids and Bézier simploids
------------------------------

A *simploid* of type :math:`(a_1, \ldots, a_n)` is the cartesian product
:math:`\Delta^{a_1} \times \cdots \times \Delta^{a_n}` of simplices.  A simplex
is a simploid of type :math:`(a)`, a cube one of type :math:`(1, \ldots, 1)`,
and a vertex one of type :math:`()`.

A *Bézier simploid* of type :math:`(a_1, \ldots, a_n)` and degree
:math:`(d_1, \ldots, d_n)` is the tensor product of Bézier simplices:

.. math::

   B(\mathbf u^1, \ldots, \mathbf u^n)
   = \sum_{\boldsymbol\alpha^1 \in \Gamma^{a_1}_{d_1}} \cdots \sum_{\boldsymbol\alpha^n \in \Gamma^{a_n}_{d_n}}
     P_{(\boldsymbol\alpha^1, \ldots, \boldsymbol\alpha^n)}
     \prod_{i=1}^n \binom{d_i}{\boldsymbol\alpha^i} (\mathbf u^i)^{\boldsymbol\alpha^i},

where :math:`\Gamma^a_d` is the set of multi-indices of length :math:`a + 1`
summing to :math:`d`, and every :math:`\mathbf u^i` lies on the standard
:math:`a_i`-simplex.  In ``torch_bsf`` the parameter of a simploid is the
concatenation :math:`(\mathbf u^1, \ldots, \mathbf u^n)` of the barycentric
coordinates of all factors, a vector of length :math:`\sum_i (a_i + 1)`.
A Bézier simploid of type :math:`(a)` is the Bézier simplex of degree
:math:`d_1` with :math:`a + 1` parameters.

:class:`~torch_bsf.simploidal.BezierSimploid` is a Lightning module like
:class:`~torch_bsf.bezier_simplex.BezierSimplex`, and
:func:`~torch_bsf.simploidal.fit_simploid` fits one to data:

.. testcode::
   :pyversion: >= 3.10, < 3.15

   import torch
   from torch_bsf.simploidal import fit_simploid, sample_simploid

   # Samples on a prism: (u0, u1, u2) on a triangle, (v0, v1) on a segment
   t = sample_simploid((2, 1), 200, seed=0)
   x = torch.cat([t[:, :3] * t[:, 3:4], t[:, 4:] ** 2], dim=1)

   bs = fit_simploid(t, x, simploid_type=(2, 1), degree=2)
   print(bs.simploid_type, bs.degree, bs.n_params, bs.n_values)
   print(torch.allclose(bs(t), x, atol=1e-4))

.. testoutput::
   :pyversion: >= 3.10, < 3.15

   (2, 1) (2, 2) 5 4
   True

A Bézier simploid can also be restricted to a face, degenerated, or
multiplied with another one (the Minkowski sum of the control points):

.. code-block:: python

   bottom = bs.face(1, 1)          # the triangle at v = (1, 0)
   pyramid_like = bs.degenerate(1, -1)  # insert a degenerate edge: type (2, 1, 1)
   product = bs.product(other)     # type (2, 1) + type of other

Simploidal sets
---------------

A *simploidal set* is a set of simploids equipped with

* **face operators** :math:`d^i_j`, mapping a simploid to the face obtained by
  removing vertex :math:`j` of its :math:`i`-th factor (the factor disappears
  when it was an edge), and
* **degeneracy operators** :math:`s^i_j`, mapping a simploid to a degenerate
  simploid of one more dimension that has the same shape: :math:`s^i_j`
  repeats vertex :math:`j` of factor :math:`i`, and :math:`s^i_{-1}` inserts a
  degenerate edge as a new factor.

These operators satisfy the commutation relations of Definition 1 in the
paper.  Degeneracies make it possible to represent, e.g., a square-based
pyramid as a cube whose top face is collapsed into a vertex.

As simplicial sets are presheaves on the simplex category, simploidal sets
are presheaves on a *simploid category*, and they form a topos; see
:doc:`simploidal_category` for the definitions and proofs.

:class:`~torch_bsf.simploidal.SimploidalSet` stores only the non-degenerate
simploids; degenerate ones are kept in a canonical form (a base simploid plus
a degeneracy map), so that equal degenerate simploids are the same Python
object.  It provides the basic construction operations of the paper:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Operation
   * - ``add_vertex``, ``add_simplex(dim)``, ``add_simploid(type)``
     - Add a complete cell with its whole boundary.
   * - ``product(other)``
     - Cartesian product of two simploidal sets (Algorithm 7).
   * - ``identify(a, b)``
     - Identify two simploids of the same type; their boundaries are identified face by face,
       ``a.face(i, j)`` with ``b.face(i, j)``, and so are the induced degeneracies.
   * - ``degenerate(simploid, face, i, j)``
     - Identify ``simploid`` with ``face.degeneracy(i, j)``: collapse it onto one of its faces.
   * - ``cone(simploids)``
     - Cone with a new vertex (Algorithm 8): a simploid of type :math:`(a_1, \ldots, a_k)` becomes one of
       type :math:`(a_1, \ldots, a_k, 1)` whose far end is degenerated into the new vertex.
   * - ``check()``
     - Verify the face commutation relations.

.. note::

   Factor indices are 0-based in ``torch_bsf``, whereas the paper uses 1-based ones:
   ``face(i, j)`` is :math:`d^{i+1}_j`, ``degeneracy(i, j)`` with :math:`j \ge 0` is :math:`s^{i+1}_j`,
   and ``degeneracy(i, -1)`` is :math:`s^i_{-1}`.

The following builds the example of Figure 1(c) of the paper, a tetrahedron
and a cube connected by a prism, and adds a pyramid on top of the cube:

.. testcode::
   :pyversion: >= 3.10, < 3.15

   from torch_bsf.simploidal import SimploidalSet

   S = SimploidalSet()
   tetra = S.add_simplex(3)
   prism = S.add_simploid((2, 1))
   cube = S.add_simploid((1, 1, 1))

   # Glue a triangle of the tetrahedron to an end of the prism,
   # and a square side of the prism to a face of the cube.
   S.identify(tetra.face(0, 0), prism.face(1, 0))
   S.identify(prism.face(0, 0), cube.face(0, 0))

   # Cone the opposite face of the cube: a square-based pyramid.
   apex, cones = S.cone([cube.face(0, 1)])
   pyramid = cones[cube.face(0, 1)]

   S.check()
   print([s.simploid_type for s in S.main_simploids()])
   print(pyramid.face(2, 0).is_degenerate, pyramid.face(2, 0).base is apex)

.. testoutput::
   :pyversion: >= 3.10, < 3.15

   [(3,), (2, 1), (1, 1, 1), (1, 1, 1)]
   True True

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

* :meth:`~torch_bsf.simploidal.SimploidalSet.boundary_matrix` returns the
  integer matrix of :math:`\partial_k`, with rows ``simploids(k - 1)`` and
  columns ``simploids(k)``; it satisfies
  ``boundary_matrix(k) @ boundary_matrix(k + 1) == 0``.
* :meth:`~torch_bsf.simploidal.SimploidalSet.betti_numbers` returns the Betti
  numbers :math:`\beta_0, \ldots, \beta_{\dim}`, the ranks of the homology
  groups.  The ranks are computed exactly by integer elimination, over the
  rationals (``field="Q"``, default) or over :math:`\mathbb Z / 2`
  (``field="Z2"``); the two differ only for spaces whose integral homology has
  torsion, such as non-orientable surfaces.

.. testcode::
   :pyversion: >= 3.10, < 3.15

   from torch_bsf.simploidal import SimploidalSet

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

Simploidal Bézier spaces
------------------------

:class:`~torch_bsf.simploidal.SimploidalBezierSpace` associates a Bézier
simploid with every simploid of a simploidal set; by default all factors have
a common degree :math:`d \ge 1`, as in the paper.  Following Section 4.3 of the
paper, every non-degenerate simploid stores only its **proper** control points
— those whose multi-indices have no zero component, i.e.
:math:`\prod_i \binom{d_i - 1}{a_i}` points for degrees :math:`(d_1, \ldots, d_n)`,
and one point for a vertex.  The control net of any simploid is rebuilt from its
proper points and the points of its faces (Property 2-1), and the control net
of a degenerate simploid from the one of its base (Property 2-2).
Consequently:

* every control point is stored exactly once;
* simploids sharing a face share the control points of that face, so the space
  is continuous across shared faces by construction;
* a degenerate face (such as the apex face of a pyramid) has the shape of its
  base (the apex vertex).

Per-factor degrees
~~~~~~~~~~~~~~~~~~

When the data varies very differently along different directions — say, a
prism that is linear across its triangles but cubic along its height — raising
every direction to the highest degree adds many control points and invites
over-fitting.  Instead of an integer, ``degree`` can be a mapping from
non-degenerate simploids to their degrees: an integer for all their factors, or
one degree per factor.

The degrees must be consistent so that shared faces keep a single control net:
a factor of a simploid and the corresponding factor of each of its faces have
the same degree.  Equivalently, each factor of a simploid has the degree of its
edges in the direction of that factor.  The given degrees propagate through
shared faces, the factors still undetermined get ``default_degree``, and an
inconsistent mapping raises :class:`ValueError`.  Edge factors inserted by a
degeneracy :math:`s^i_{-1}` impose no constraint, since a degenerate Bézier
simploid is constant along them; ``simploid_degree`` reports them with the
maximum degree of the space (its ``degree`` attribute).

.. testcode::
   :pyversion: >= 3.10, < 3.15

   import torch
   from torch_bsf.simploidal import SimploidalSet, SimploidalBezierSpace, sample_simploid

   # A prism glued to a cube along a square face
   S = SimploidalSet()
   prism = S.add_simploid((2, 1))
   cube = S.add_simploid((1, 1, 1))
   S.identify(prism.face(0, 0), cube.face(0, 0))

   # Linear across the triangles, cubic along the height; the cube inherits
   # these degrees on the shared face and gets the default on its last factor.
   space = SimploidalBezierSpace(S, {prism: (1, 3)}, n_values=1, default_degree=2)
   print(space.simploid_degree(prism), space.simploid_degree(cube))

   # Inconsistent degrees on the shared face are rejected
   try:
       SimploidalBezierSpace(S, {prism: (1, 3), cube: (2, 2, 3)}, n_values=1)
   except ValueError:
       print("inconsistent")

.. testoutput::
   :pyversion: >= 3.10, < 3.15

   (1, 3) (2, 1, 3)
   inconsistent

Fitting
-------

:func:`~torch_bsf.simploidal.fit_simploidal_set` fits a simploidal Bézier
space to samples attached to any non-degenerate simploids, each given in the
parameters of its simploid.  The proper control points minimize the total
squared error

.. math::

   \sum_{(\sigma, \mathbf t, \mathbf x)} \| B_\sigma(\mathbf t) - \mathbf x \|^2
   + \lambda \| P \|^2,

where :math:`B_\sigma` is the Bézier simploid of :math:`\sigma` and
:math:`\lambda` is the optional ``ridge`` weight.  Because the problem is
linear in the proper control points, the default solver ``"lstsq"`` solves it
exactly (the minimum-norm solution when it is under-determined and
``ridge=0``).  ``solver="trainer"`` trains the space with
:class:`lightning.pytorch.Trainer` instead, forwarding the extra keyword
arguments to it; this is useful for very large data sets.

.. testcode::
   :pyversion: >= 3.10, < 3.15

   import torch
   from torch_bsf.simploidal import SimploidalSet, fit_simploidal_set, sample_simploid

   # A triangle and a square sharing an edge
   S = SimploidalSet()
   tri = S.add_simplex(2)
   sq = S.add_simploid((1, 1))
   S.identify(tri.face(0, 2), sq.face(1, 1))

   # Noisy samples of two unrelated functions on the two cells
   t1 = sample_simploid((2,), 100, seed=0)
   t2 = sample_simploid((1, 1), 100, seed=1)
   x1 = torch.sin(3 * t1[:, :1]) + 0.01 * torch.randn(100, 1)
   x2 = t2[:, 2:3] ** 2 + 0.01 * torch.randn(100, 1)

   space = fit_simploidal_set(S, [(tri, t1, x1), (sq, t2, x2)], degree=3)

   # The two pieces agree on the shared edge
   w = sample_simploid((1,), 10, seed=2)
   on_tri = torch.cat([w, torch.zeros(10, 1)], dim=1)
   on_sq = torch.cat([w, torch.tensor([[1.0, 0.0]]).expand(10, 2)], dim=1)
   print(torch.allclose(space.evaluate(tri, on_tri), space.evaluate(sq, on_sq)))

.. testoutput::
   :pyversion: >= 3.10, < 3.15

   True

Useful methods of a fitted space:

* ``evaluate(simploid, t)`` — evaluate the Bézier simploid of any simploid;
* ``simploid_degree(simploid)`` — the degree of each factor of any simploid;
* ``forward(t, simploid_ids)`` — evaluate a batch mixing several simploids
  (parameters padded to a common width; ids from ``simploid_id``);
* ``control_points(simploid)`` — the full control net of any simploid;
* ``bezier_simploid(simploid)`` — a standalone
  :class:`~torch_bsf.simploidal.BezierSimploid` copy, e.g. for further
  processing.

Limitations
-----------

* Per-factor degrees must agree exactly on shared faces.  Pieces of different
  degrees on a shared face are not reconciled by degree elevation (which would
  store the face at the lower degree and constrain the higher-degree piece to
  its elevation); give the shared factors the higher degree instead.
* Degrees are chosen by the user; they are not selected automatically.
* A space captures the structure of its simploidal set when it is created;
  build the set completely before creating or fitting a space.
* After :meth:`~torch_bsf.simploidal.SimploidalSet.identify`, references to
  simploids that were merged away are stale; use the returned simploid or the
  faces of the surviving simploids.
