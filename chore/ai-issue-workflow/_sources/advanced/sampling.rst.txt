Advanced Sampling
=================

PyTorch-BSF provides utilities for generating parameter points on a simplex beyond the default uniform grid.

The ``torch_bsf.sampling`` Module
----------------------------------

Three functions are available:

``simplex_grid(n_params, degree)``
    Generates a uniform grid on the simplex using the stars-and-bars method.
    This is the default sampling used by ``fit()`` and the CLI.

``simplex_random(n_params, n_samples)``
    Generates random points uniformly distributed over the simplex via the Dirichlet distribution.
    Useful when you want stochastic coverage or a large number of samples.

.. code-block:: python

   from torch_bsf.sampling import simplex_random

   pts = simplex_random(n_params=3, n_samples=200)

``simplex_sobol(n_params, n_samples)``
    Generates quasi-random points using a scrambled Sobol sequence projected onto the simplex.
    Sobol sequences provide better coverage than purely random sampling, reducing clustering and gaps.
    Requires ``scipy`` (``pip install scipy`` or ``pip install pytorch-bsf[sampling]``).

.. code-block:: python

   from torch_bsf.sampling import simplex_sobol

   pts = simplex_sobol(n_params=3, n_samples=128)

When to Use Each Method
-----------------------

+-------------------+----------------------------------------------------------+
| Method            | Recommended use                                          |
+===================+==========================================================+
| ``simplex_grid``  | Small, structured datasets; default CLI behavior.        |
+-------------------+----------------------------------------------------------+
| ``simplex_random``| Large-scale or stochastic sampling experiments.          |
+-------------------+----------------------------------------------------------+
| ``simplex_sobol`` | When uniform coverage is critical (e.g., active learning |
|                   | initialization, quasi-Monte Carlo integration).          |
+-------------------+----------------------------------------------------------+

Sobol Sequence: Usage Conditions
---------------------------------

The following requirements and best-practice recommendations apply when using ``simplex_sobol``.

**scipy is required (optional dependency)**

``simplex_sobol`` is built on top of :class:`scipy.stats.qmc.Sobol`.
``scipy`` is an *optional* dependency of PyTorch-BSF and is not installed by default.

Install it with one of:

.. code-block:: bash

   pip install scipy
   # or, to install the sampling extras:
   pip install pytorch-bsf[sampling]

If ``scipy`` is not available, an :exc:`ImportError` is raised at call time with a clear message.

**n_params must be at least 2**

The Sobol sequence is generated in ``n_params - 1`` dimensions and then projected
onto the ``n_params``-dimensional simplex.  With only 1 parameter the simplex is
a single point {1}, so no sampling is needed; ``simplex_grid(1, degree)`` returns
that point directly.

**Use power-of-2 sample sizes for best results**

Sobol sequences are constructed in base 2.  Their low-discrepancy guarantee is
sharpest when ``n_samples`` is an exact power of 2: 2, 4, 8, 16, 32, 64, 128, 256, …

When a non-power-of-2 value is requested, :func:`simplex_sobol` still returns the
requested number of samples, but a ``UserWarning`` is emitted:

.. code-block:: python

   import warnings
   from torch_bsf.sampling import simplex_sobol

   with warnings.catch_warnings(record=True) as w:
       warnings.simplefilter("always")
       pts = simplex_sobol(n_params=3, n_samples=200)  # 200 is not a power of 2
   print(str(w[0].message))
   # UserWarning: simplex_sobol: n_samples=200 is not a power of 2. ...

To suppress the warning intentionally, round up to the next power of 2 and slice:

.. code-block:: python

   pts = simplex_sobol(n_params=3, n_samples=256)[:200]

Sobol Sequence: Precision Characteristics
------------------------------------------

**Low-discrepancy sequences**

A sequence is *low-discrepancy* if its points are distributed more evenly than
independent pseudo-random draws.  The *discrepancy* of a point set measures how
far the empirical distribution of points deviates from the uniform distribution
over a region.

Sobol sequences are among the most widely used low-discrepancy sequences in
quasi-Monte Carlo (QMC) methods.

**Convergence rate**

For an integrand with bounded variation, QMC integration using a Sobol sequence
of *N* points in *d* dimensions converges at roughly

.. math::

   \text{error} = O\!\left(\frac{(\log N)^{d-1}}{N}\right)

compared with the Monte Carlo rate of :math:`O(1/\sqrt{N})`.

In practice this means:

* For **small dimensions** (``n_params`` up to ~10) and **moderate sample sizes**
  (say N = 64–1024), a Sobol sequence typically gives noticeably more uniform
  coverage than an equal number of Dirichlet random samples.
* For **large dimensions** (``n_params`` >> 10), the :math:`(\log N)^{d-1}` factor
  grows quickly and the QMC advantage diminishes unless N is very large.
  Consider ``simplex_random`` for high-dimensional simplices.

**Scrambled Sobol**

``simplex_sobol`` always uses ``scramble=True`` when constructing
:class:`~scipy.stats.qmc.Sobol`.  Scrambling applies a random digital shift
to the sequence so that:

1. The resulting points are *still* low-discrepancy (coverage is preserved).
2. Each call produces a different realisation, enabling variance estimation by
   averaging results from multiple scrambled sequences.

**Projection onto the simplex**

The raw Sobol points live in the unit hypercube ``[0, 1]^(n_params - 1)``.
``simplex_sobol`` maps them to the simplex using the *sorted-differences* (also
called *order-statistics*) mapping:

1. Sort each ``(n_params - 1)``-dimensional point in ascending order.
2. Prepend 0 and append 1 to obtain ``n_params + 1`` boundary values.
3. Take consecutive differences to get ``n_params`` non-negative values
   that sum to 1.

This mapping is measure-preserving for the uniform distribution on the hypercube,
so the resulting simplex points are also (approximately) uniformly distributed
over the simplex and inherit the low-discrepancy properties of the Sobol sequence.

