Pareto Set Learning by Stochastic Gradient Descent
==================================================

:func:`torch_bsf.fit` fits a model to data: Pareto-optimal solutions computed
beforehand.  When the objective functions themselves are available,
:func:`torch_bsf.fit_sgd` fits the model to the Pareto set directly, with the
preconditioned stochastic gradient descent of :cite:t:`hikima2025stochastic`.
It works with every model of ``torch_bsf.simploidal``: Bézier simplices,
Bézier simploids and simploidal Bézier spaces (see :doc:`simploidal`).

The algorithm
-------------

Let :math:`f = (f_1, \ldots, f_M): \mathbb R^L \to \mathbb R^M` be the
objectives and :math:`b(\mathbf t \mid P) = P^\top \mathbf z(\mathbf t)` a
Bézier simplex of degree :math:`D` on :math:`\Delta^{M-1}`, where
:math:`\mathbf z(\mathbf t)` is the vector of the Bernstein basis
polynomials.  The paper minimizes the linear scalarization weighted by the
parameter itself, averaged over the simplex:

.. math::

   \mathcal L(P) = \mathbb E_{\mathbf t \sim U(\Delta^{M-1})}\bigl[\mathbf t^\top f(b(\mathbf t \mid P))\bigr].

For strongly convex objectives, the minimizer of
:math:`\mathbf t^\top f(\mathbf x)` traces the whole Pareto set as
:math:`\mathbf t` moves on the simplex, so the minimizer of
:math:`\mathcal L` approximates the Pareto set, parametrized by the weight.

The expectation is replaced by the mean over :math:`n` fixed samples
:math:`\mathbf t_1, \ldots, \mathbf t_n`, and the control points are updated on
minibatches :math:`B` of them:

.. math::

   P^{(k+1)} = P^{(k)} - \alpha_k \Sigma^{-1} \nabla \mathcal L_B(P^{(k)}),
   \qquad
   \Sigma = \mathbb E_{\mathbf t \sim U(\Delta^{M-1})}\bigl[\mathbf z(\mathbf t)\,\mathbf z(\mathbf t)^\top\bigr].

The preconditioner :math:`\Sigma^{-1}` turns the step into the one that
minimizes the linearized loss plus
:math:`\frac{1}{2\alpha_k}\mathbb E_{\mathbf t}\|b(\mathbf t \mid P) - b(\mathbf t \mid P^{(k)})\|^2`:
it measures how far the *surface* moves rather than how far its control
points move.  The paper proves that this converges faster than plain SGD
(:math:`\Sigma = I`), whose steps are distorted by the ill-conditioning of the
Bernstein basis.  :func:`~torch_bsf.simploidal.gram_matrix` computes
:math:`\Sigma` exactly, and it is factorized once.

Fitting a Bézier simplex to a Pareto set
-----------------------------------------

The objectives :math:`f_m(\mathbf x) = \|\mathbf x - \mathbf e_m\|^2`
(:math:`m = 1, 2, 3`) have the triangle spanned by
:math:`\mathbf e_1, \mathbf e_2, \mathbf e_3` as their Pareto set, reached at
:math:`\mathbf x = \mathbf t`.  Write the objectives for a batch of points,
wrap them with :func:`~torch_bsf.simploidal.linear_scalarization`, and pass
an initial model, which fixes the type, the degree and the dimension
:math:`L`:

.. testcode::
   :pyversion: >= 3.10, < 3.15

   import torch
   import torch_bsf
   from torch_bsf.simploidal import BezierSimploid, linear_scalarization, sample_domain

   def objective(x):  # (batch, L) -> (batch, M)
       return ((x.unsqueeze(1) - torch.eye(3)) ** 2).sum(dim=-1)

   init = BezierSimploid.zeros((2,), 2, n_values=3)  # a quadratic Bézier triangle in R^3
   loss = linear_scalarization(objective)
   bs = torch_bsf.fit_sgd(init, loss, n_samples=100, batch_size=20, max_iter=100, step_size=0.2, seed=0)

   (t,) = sample_domain(bs, 1000, seed=1)  # fresh samples to check generalization
   print(torch.allclose(bs(t), t, atol=1e-3))

.. testoutput::
   :pyversion: >= 3.10, < 3.15

   True

The main options are:

* ``batch_size`` and ``n_samples``: the minibatch size :math:`m` and the
  number :math:`n` of fixed samples (or pass your own samples as ``inputs``);
* ``step_size`` and ``step_decay``: a constant step :math:`\alpha`, or
  :math:`\alpha / \sqrt{k + 1}` with ``step_decay="sqrt"``;
* ``preconditioner="none"``: plain SGD, the baseline of the paper.

The paper uses :math:`P_0 = O`, :math:`n = 100`, :math:`m = 20`, 100
iterations and :math:`\alpha = 0.05`, which are the defaults except for the
minibatch size (full batch by default).  The returned model is a copy, so
``init`` can be reused.

Generalization to simploids and simploidal spaces
-------------------------------------------------

Every model of ``torch_bsf.simploidal`` is linear in its control points,
``model(*inputs) == model.design_matrix(*inputs) @ P``, so the algorithm
applies to all of them with :math:`\mathbf z(\mathbf u)` the row of the
design matrix and :math:`\mathbf u` uniform on the domain of the model:

* For a **Bézier simploid** of type :math:`(a_1, \ldots, a_n)`,
  :math:`\mathbf u` is uniform on the product of simplices, and :math:`\Sigma`
  is the Kronecker product of the Gram matrices of its factors.
* For a **simploidal Bézier space**, :math:`\mathbf u` lies on the simploid
  :math:`c` with probability :math:`w_c` and is uniform on it, so
  :math:`\Sigma = \sum_c w_c S_c^\top \Sigma_c S_c`, where :math:`S_c` maps the
  stored control points to the control net of :math:`c`.  By default the
  weights are uniform over the main simploids (those that are not a face of
  another); pass ``weights`` to change them.  They must cover every stored
  control point, so that :math:`\Sigma` is invertible.

The loss is any function ``loss(values, *inputs)`` returning one value per
sample, where ``values`` is the output of the model and ``inputs`` are its
arguments: ``(t,)`` for a Bézier simploid and ``(t, simploid_ids)`` for a
space (see :func:`~torch_bsf.simploidal.sample_domain`).
:func:`~torch_bsf.simploidal.linear_scalarization` needs one parameter per
objective, i.e., a Bézier simplex; on other domains, choose how the
parameters weight the objectives.  For example, on a square
:math:`(u_0, u_1) \times (v_0, v_1)`, the weights
:math:`(u_0 v_0, u_0 v_1, u_1 v_0, u_1 v_1)` of four objectives sweep the
tetrahedron of weights along a bilinear patch:

.. testcode::
   :pyversion: >= 3.10, < 3.15

   from torch_bsf.simploidal import gram_matrix

   def objective4(x):
       return ((x.unsqueeze(1) - torch.eye(4)) ** 2).sum(dim=-1)

   def loss4(x, t):
       w = (t[:, :2].unsqueeze(2) * t[:, 2:].unsqueeze(1)).flatten(1)  # (batch, 4)
       return (w * objective4(x)).sum(dim=-1)

   init = BezierSimploid.zeros((1, 1), 1, n_values=4)
   patch = torch_bsf.fit_sgd(init, loss4, batch_size=20, step_size=0.2, seed=0)
   print(torch.allclose(patch.control_points, torch.eye(4), atol=1e-3))
   print(gram_matrix(init).shape)

.. testoutput::
   :pyversion: >= 3.10, < 3.15

   True
   torch.Size([4, 4])

A simploidal Bézier space is fitted in the same way; the loss then also
receives the index of the simploid of every sample.
