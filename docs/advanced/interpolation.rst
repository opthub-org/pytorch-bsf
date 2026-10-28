Shape-preserving interpolation on simplicial meshes
===================================================

When to interpolate instead of fitting
--------------------------------------

``fit`` estimates one global polynomial (or one per glued cell) by least
squares. When the response is piecewise smooth, with kinks where an active
set or a regime changes, a global polynomial of moderate degree oscillates
between design points and can overshoot the data. If the response is
optimized afterwards, an overshoot becomes a spurious optimum that the data
never showed.

:func:`~torch_bsf.interpolation.hermite_interpolant` builds a local
alternative from data at the vertices of a simplicial mesh: a continuous
piecewise cubic Bézier model that passes through every datum and, with the
default ``clamp=True``, never leaves the range of the data at the vertices of
the simplex it is evaluated in. On a chain of intervals it is the familiar
monotone piecewise cubic Hermite interpolation (PCHIP)
:cite:p:`fritsch1980monotone`; on triangles and tetrahedra it is its natural
generalization, which we call simplicial PCHIP.

Use it when evaluations are exact (or nearly so) and the design points are
given, for example a grid or the points chosen by an adaptive loop. With
noisy observations prefer a regularized fit: interpolation reproduces the
noise.

Interpolate on a mesh
---------------------

A :class:`~torch_bsf.interpolation.SimplicialMesh` takes vertex coordinates
of shape ``(n_vertices, d)`` and simplices of shape ``(n_simplices, d + 1)``
given by vertex numbers. The simplices must not be degenerate and must meet
face to face (the mesh must be *conforming*); ``SimplicialMesh`` checks this
and raises ``ValueError`` otherwise (see `Conforming meshes`_). The example
triangulates a jittered grid of the unit square and interpolates a function
with a kink along a diagonal.

.. testcode:: interpolation

    import torch
    from torch_bsf.interpolation import SimplicialMesh, hermite_interpolant

    n = 6
    i, j = torch.meshgrid(torch.arange(n + 1), torch.arange(n + 1), indexing="ij")
    vertices = torch.stack([i.flatten(), j.flatten()], dim=1).double() / n
    number = lambda a, b: a * (n + 1) + b
    simplices = []
    for a in range(n):
        for b in range(n):
            simplices.append([number(a, b), number(a + 1, b), number(a + 1, b + 1)])
            simplices.append([number(a, b), number(a, b + 1), number(a + 1, b + 1)])
    mesh = SimplicialMesh(vertices, simplices)

    def response(x):
        return torch.stack([(x[:, 0] - x[:, 1]).abs() + x[:, 0] ** 2, x.sum(1)], dim=1)

    model = hermite_interpolant(mesh, response(vertices))
    x = torch.rand(1000, 2, dtype=torch.float64)
    points = mesh.locate(x)
    prediction = model(points)
    assert torch.allclose(model(mesh.locate(vertices)), response(vertices), atol=1e-12)
    assert prediction.shape == (1000, 2)

``mesh.locate`` returns :class:`~torch_bsf.core.Points`: the cell ID of the
simplex containing each point and its barycentric coordinates there.
``model`` is an ordinary :class:`~torch_bsf.core.BezierModel` on
``mesh.space``, a :class:`~torch_bsf.core.BezierSpace` of degree three
compiled from the mesh. Everything that accepts a model, such as
differentiation with respect to the points or
:func:`~torch_bsf.splitting.subdivide`, accepts this one.

The barycentric coordinates of cell ``mesh.cell_ids[k]`` refer to the
vertices ``mesh.simplices[k]``, which lists the vertex numbers of simplex
``k`` in ascending order (the order you passed is not kept). A point on a
face shared by several simplices is assigned to one of them; the prediction
does not depend on the choice because the interpolant is continuous. Points
outside the mesh by more than ``tol`` raise ``ValueError``, while points
within ``tol`` of its boundary are projected onto it. Point location tests
every simplex, which is simple and exact but costs
``O(n_points * n_simplices)``.

The mesh keeps independent snapshots of the input vertices and simplices,
including CPU float64 tensors and NumPy arrays. Changing those original
inputs later does not change coordinates, point location, volumes or
interpolation. Treat ``mesh.vertices`` and ``mesh.simplices`` as read-only:
direct in-place edits are unsupported because cached geometry and the
compiled space would become inconsistent. Construct a new mesh to change
its geometry or topology. Copying tensor vertices preserves their autograd
connection.

How the cubic pieces are built
------------------------------

On each simplex the model is a cubic Bézier simplex. Its controls are set
from the data :math:`f_i` and gradients :math:`g_i` at the vertices
:math:`x_i`, like the Bézier form of a cubic Hermite segment:

* at vertex :math:`i`, the control is :math:`f_i`;
* on edge :math:`i \to j`, next to vertex :math:`i`, it is
  :math:`f_i + g_i \cdot (x_j - x_i) / 3`;
* at the centre of the triangle face :math:`(i, j, l)`, it is the sum of the
  six edge controls of that face divided by four, minus
  :math:`(f_i + f_j + f_l) / 6`.

A cubic simplex has no other controls. The face formula is the degree
elevation of the quadratic Bézier triangle whose edge midpoint controls
match the cubic edge controls :cite:p:`farin2002curves`, so the interpolant
reproduces every quadratic polynomial when the gradients are exact.

Each control depends only on the data of the edge or face that carries it.
``SimplicialMesh`` glues simplices that share a face so that the face's
controls are the same coefficient rows from both sides, which makes the
interpolant continuous (C0). It is not C1 in general: gradients across a
shared face can jump.

When ``gradients`` is omitted,
:func:`~torch_bsf.interpolation.estimate_gradients` fits a quadratic by
weighted least squares (weights :math:`1/|x_w - x_v|^2`) to the data on the
smallest :math:`k`-ring of mesh neighbours, :math:`k \ge 2`, that has two
points more than the quadratic has unknowns, and falls back to a linear fit
on the one-ring. Pass your own ``gradients`` of shape
``(n_vertices, d, n_outputs)`` if you know them.

.. testcode:: interpolation

    from torch_bsf.interpolation import estimate_gradients

    quadratic = 1 + vertices[:, :1] - vertices[:, 1:] ** 2 + vertices[:, :1] * vertices[:, 1:]
    exact = hermite_interpolant(mesh, quadratic, clamp=False)
    truth = 1 + x[:, :1] - x[:, 1:] ** 2 + x[:, :1] * x[:, 1:]
    assert torch.allclose(exact(points), truth, atol=1e-10)
    assert estimate_gradients(mesh, quadratic).shape == (mesh.n_vertices, 2, 1)

Range limiting and its error guarantee
--------------------------------------

With ``clamp=True`` (the default) each edge control is clipped, output by
output, into the range of the data at the two vertices of its edge, and each
face control into the range of the data at the three vertices of its face.
Clipping uses only the data of that edge or face, so shared controls are
still shared and continuity is kept. Vertex data are unchanged.

Bernstein polynomials are nonnegative and sum to one, so a Bézier simplex
takes values in the convex hull of its controls. Every control of a simplex
:math:`T` is now within the range of its vertex data, hence for every output

.. math::

   \min_{v \in T} f(v) \le p(x) \le \max_{v \in T} f(v), \qquad x \in T.

If the data come from a function :math:`f`, then for :math:`x \in T`,

.. math::

   |p(x) - f(x)| \le \max_{v \in T} |f(v) - f(x)| \le \omega_f(h),

where :math:`h` is the longest edge of the mesh and
:math:`\omega_f(\delta) = \sup\{|f(y) - f(z)| : |y - z| \le \delta\}` is
the modulus of continuity of :math:`f`. The bound holds for any gradients,
including poorly estimated ones, and for kinked or discontinuous-derivative
responses; refining the mesh therefore converges uniformly for any
continuous :math:`f` on the meshed domain. In particular, the interpolant
never predicts a value outside the observed range of a cell, so its minimum
over the domain equals the smallest datum and is attained at a design point:
minimizing the interpolant cannot produce a spurious optimum.

.. testcode:: interpolation

    y = response(vertices)
    corners = y[mesh.simplices]
    cell_of = {c: k for k, c in enumerate(mesh.cell_ids.tolist())}
    rows = torch.tensor([cell_of[c] for c in points.cell_ids.tolist()])
    assert (prediction >= corners.amin(dim=1)[rows] - 1e-12).all()
    assert (prediction <= corners.amax(dim=1)[rows] + 1e-12).all()

In one dimension, clipping keeps the slope ratios
:math:`\alpha = g_i h / \Delta` and :math:`\beta = g_{i+1} h / \Delta` of
each interval in :math:`[0, 3]`, where :math:`\Delta` is the change of the
data over the interval. This is the box condition of Fritsch and Carlson,
which makes the cubic monotone on every interval where the data are
monotone. Without clipping, a step in the data overshoots:

.. testcode:: interpolation

    line = torch.arange(8, dtype=torch.float64)[:, None]
    chain = SimplicialMesh(line, [[k, k + 1] for k in range(7)])
    step = torch.tensor([0., 0., 0., 0., 1., 1., 1., 1.], dtype=torch.float64)
    t = chain.locate(torch.linspace(0, 7, 701, dtype=torch.float64)[:, None])
    loose = hermite_interpolant(chain, step, clamp=False)(t)
    tight = hermite_interpolant(chain, step)(t)
    print(f"unclamped: [{loose.min():.3f}, {loose.max():.3f}]")
    print(f"clamped:   [{tight.min():.3f}, {tight.max():.3f}]")
    assert (tight.diff(dim=0) >= -1e-12).all()

.. testoutput:: interpolation

    unclamped: [-0.047, 1.047]
    clamped:   [0.000, 1.000]

Clipping trades accuracy for safety. Near a local extremum of the response,
the true function leaves the range of the vertex data of a cell, and the
clamped interpolant cannot follow it, so quadratic data are no longer
reproduced exactly in such cells. Use ``clamp=False`` when the response is
smooth and exact quadratic reproduction matters more than the range
guarantee.

Choose where to add design points
---------------------------------

:func:`~torch_bsf.interpolation.refinement_indicator` scores each simplex by
its volume times the difference between the cubic interpolant and linear
interpolation at its centroid. A large score means the data bend within a
large cell. Evaluating the response near the midpoint of the longest edge of
the top-scoring cell and remeshing is a simple adaptive design. The example
refines a one-dimensional kink, where remeshing is just sorting.

.. testcode:: interpolation

    from torch_bsf.interpolation import refinement_indicator

    def kink(s):
        return (s - 0.37).abs()

    def chain_mesh(s):
        s = s.sort().values
        return s, SimplicialMesh(s[:, None], [[k, k + 1] for k in range(len(s) - 1)])

    test = torch.linspace(0, 1, 1001, dtype=torch.float64)
    design, chain = chain_mesh(torch.linspace(0, 1, 5, dtype=torch.float64))
    first = (hermite_interpolant(chain, kink(design))(chain.locate(test[:, None]))[:, 0] - kink(test)).abs().max()
    for _ in range(6):
        worst = int(refinement_indicator(chain, kink(design)).argmax())
        a, b = chain.simplices[worst].tolist()
        design, chain = chain_mesh(torch.cat([design, (design[a:a + 1] + design[b:b + 1]) / 2]))
    last = (hermite_interpolant(chain, kink(design))(chain.locate(test[:, None]))[:, 0] - kink(test)).abs().max()
    assert last < first / 4

In higher dimensions, :func:`~torch_bsf.interpolation.refine` bisects the
selected simplices at the midpoint of their longest edge and returns a new
mesh. The new vertices are appended after the old ones, so the data already
observed keep their rows; evaluate the response only at the new vertices.
The example starts from the lattice mesh of the triangle described below.

.. testcode:: interpolation

    from torch_bsf.interpolation import lattice_mesh, refine

    adaptive = lattice_mesh(2, 2)
    observed = response(adaptive.vertices)
    while adaptive.n_vertices < 20:
        worst = int(refinement_indicator(adaptive, observed).argmax())
        old = adaptive.n_vertices
        adaptive = refine(adaptive, [worst])
        observed = torch.cat([observed, response(adaptive.vertices[old:])])
    assert adaptive.unmatched_faces == 0
    model = hermite_interpolant(adaptive, observed)

Conforming meshes
-----------------

Two simplices of a conforming mesh meet in a common face or not at all. The
interpolant is continuous only on such meshes, because a control on a face is
shared only by simplices that both have that whole face. ``SimplicialMesh``
counts the faces that break this rule, those shared by more than two
simplices and those across which the mesh continues without a matching face
(a hanging vertex, or two cells splitting a common square along different
diagonals), in ``unmatched_faces``. It raises ``ValueError`` if there are any,
or warns with ``strict=False``. Faces on the boundary of the meshed domain
are fine.

With the optional SciPy dependency (the ``sparse`` extra), a Delaunay
triangulation of points in general position can be passed directly:

.. testcode:: interpolation
    :skipif: __import__("importlib.util").util.find_spec("scipy") is None

    from scipy.spatial import Delaunay

    scattered = torch.cat([torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]), torch.rand(30, 2)]).double()
    delaunay = SimplicialMesh(scattered, Delaunay(scattered.numpy()).simplices)
    smooth = hermite_interpolant(delaunay, response(scattered))
    score = refinement_indicator(delaunay, response(scattered))
    assert score.shape == (delaunay.n_simplices,)

Points of a lattice, such as a grid of mixture weights, are co-spherical, so
their Delaunay triangulation is not unique. Qhull then returns flat
simplices, which ``SimplicialMesh`` rejects as degenerate, and removing them
leaves cells that split a flat polygon in different ways on its two sides:
the mesh is not conforming. For the degree-``k`` lattice of a simplex, use
:func:`~torch_bsf.interpolation.lattice_mesh` instead. It returns the Kuhn
(Freudenthal) triangulation :cite:p:`kuhn1960combinatorial`: ``k ** d``
simplices of equal volume (not congruent for ``d >= 3``) whose vertices are
the lattice points, in the order of
:func:`~torch_bsf.sampling.grid` with ``resolution=k``, given by their first
``d`` barycentric coordinates.

.. testcode:: interpolation

    mixtures = lattice_mesh(3, 4)
    print(mixtures.n_vertices, mixtures.n_simplices, mixtures.unmatched_faces)

.. testoutput:: interpolation

    35 64 0

Refinement by :func:`~torch_bsf.interpolation.refine` keeps a mesh
conforming. It bisects an edge in every simplex containing it, and before
that it refines each of those simplices whose longest edge is another one,
recursively, which is the longest-edge bisection of Rivara
:cite:p:`rivara1984algorithms`. Refining one simplex can therefore add more
than one vertex. Ties between edges of equal length, common on lattices, are
broken by vertex numbers in the same way everywhere. Starting from a lattice
mesh, every new vertex is a lattice point of a finer lattice (degree ``2k``,
then ``4k``, and so on).

Limitations
-----------

* Meshes of arbitrary point sets are not generated here; only lattice meshes
  and their refinements are. The simplices must be full-dimensional,
  non-degenerate and conforming. The conformity check does not detect
  simplices that overlap without sharing a face.
* The interpolant is cubic and C0. C1 macro-elements (Clough–Tocher,
  Powell–Sabin) and non-simplicial cells such as prisms are not provided.
* Quadratic reproduction requires enough neighbours around each vertex for
  the gradient estimate; near a sparse boundary only linear data are
  reproduced unless you pass gradients.
* Computation is in float64 on the CPU.
  :func:`~torch_bsf.interpolation.estimate_gradients` returns signed infinity
  without a warning for gradient components beyond the float64 range, even
  when the vertex data are finite (for example, on very short edges).
  :func:`~torch_bsf.interpolation.hermite_interpolant` and
  :func:`~torch_bsf.interpolation.refinement_indicator` raise ``ValueError``
  identifying overflow in the interpolant when its controls are non-finite.
  Clamping can clip infinite controls into the vertex data range, but cannot
  repair NaN controls. Rescale the vertex data or coordinates if this occurs.
