A fast exact transform for codimension three
============================================

The positive double sum in :doc:`skeleton_risk_codimension_three_exact`
can be evaluated by a single homogeneous polynomial transform and one
angular sum.  This chapter keeps its uniform-simplex design, evaluation
and noise assumptions.  It applies to every nonvertex source size
:math:`s\geq2`, including the five-vertex edge and six-vertex triangle.
It improves the exact arithmetic operation count; it does not claim a
fixed-dimensional degree state, a minimal recurrence, or a bounded-length
classical special-function expression.

The coefficient identity
-------------------------

For :math:`D\geq s`, put :math:`N=D-s` and
:math:`T=2D+s-1=2N+3s-1`.  Use the explicit positive harmonic diagonals
:math:`\delta_3(j)` in :eq:`risk-nonvertex-q3-harmonic-diagonal` and define

.. math::

   F_N(u)&=\sum_{j=0}^N c_ju^j,\qquad
   c_j=\binom T{2j}\binom{2j}j\delta_3(j),\\
   P_{s,D}(z)&=(1+z)^T F_N\!\left(\frac z{(1+z)^2}\right),\qquad
   B_L=[z^L]P_{s,D}(z).

Despite its rational presentation, :math:`P_{s,D}` is a polynomial:
every exponent :math:`T-2j` is at least :math:`3s-1>0`.

.. prf:theorem:: A homogeneous transform for all angular modes
   :label: thm-risk-q3-homogeneous-transform

   For every integer :math:`s\geq2,D\geq s` and :math:`0\leq L\leq N`,

   .. math::
      :label: risk-q3-transformed-average

      \frac{B_L}{\binom TL^2}
       =\frac{\displaystyle\sum_{j=0}^L
                    \binom Lj\binom{T-L}j\delta_3(j)}{\binom TL}.

   Consequently the exact coefficient is

   .. math::
      :label: risk-q3-transform-coefficient

      a_{s-1}(s+3,D)=
       \binom{s+3}s^2\frac{s(s+1)(s+2)}
                {(2D+s)(2D+s+1)(2D+s+2)}
       \sum_{L=0}^N\binom{N-L+s-2}{s-2}
                       \frac{B_L}{\binom TL^2}.

   The coefficient is zero for the inactive positive degrees :math:`D<s`.

.. prf:proof::

   For :math:`0\leq j\leq L\leq N`, comparison of factorials gives

   .. math::

      \binom TL\binom Lj\binom{T-L}j
       =\binom T{2j}\binom{2j}j\binom{T-2j}{L-j}.

   Expanding :math:`z^j(1+z)^{T-2j}` shows that :math:`B_L` is the
   sum of the right side times :math:`\delta_3(j)`.  Terms with
   :math:`j>L` cannot contribute.  Division by :math:`\binom TL^2`
   proves :eq:`risk-q3-transformed-average`.  In the existing positive
   nonvertex formula reindex the angular degree as :math:`k=N-L`.
   Its other upper binomial index is :math:`T-L`, its normalizer is
   :math:`\binom TL`, and its multiplicity is
   :math:`\binom{N-L+s-2}{s-2}`.  This gives
   :eq:`risk-q3-transform-coefficient`.  All binomials and denominators
   are positive on this domain, including :math:`L=N=0`.  The inactive
   prefix contains no source bubbles and is returned separately.

The final sum has :math:`N+1` terms.  Writing a polynomial coefficient
does not by itself shorten its evaluation; the following algorithm is
part of the construction.

Divide and conquer over integers
---------------------------------

Set :math:`b(z)=(1+z)^2`.  For a coefficient list
:math:`C=(C_0,\ldots,C_{n-1})`, define

.. math::

   R(C)=\sum_{j=0}^{n-1} C_j z^j b(z)^{n-1-j}.

For :math:`n\geq2`, split after :math:`m=\lfloor n/2\rfloor` entries
into :math:`C^{\rm lo},C^{\rm hi}`.  Direct separation of the terms gives

.. math::
   :label: risk-q3-transform-divide-conquer

   R(C)=b^{n-m}R(C^{\rm lo})+z^mR(C^{\rm hi}),\qquad
   R((C_0))=C_0.

The powers :math:`b^r=(1+z)^{2r}` are cached by their integer exponent.
At a given recursion level there are at most two distinct list lengths,
so all cached powers together have :math:`O(n)` coefficients.

The implementation first sets
:math:`Q=\operatorname{lcm}_{0\leq j\leq N}
\operatorname{denominator}(\delta_3(j))` and uses the integers
:math:`C_j=Qc_j`.  It evaluates :eq:`risk-q3-transform-divide-conquer`
and then multiplies by :math:`(1+z)^{3s-1}`.  The first :math:`N+1`
coefficients are exactly :math:`QB_L`.  The required input binomial
weights have the nonsingular update

.. math::

   w_0=1,\qquad
   w_{j+1}=w_j\frac{(T-2j)(T-2j-1)}{(j+1)^2},\qquad
   w_j=\binom T{2j}\binom{2j}j.

Each quotient is an integer; its numerator is formed before exact
division.  Harmonic diagonals and the normalizers :math:`\binom TL`
also use their established nonsingular updates.

For fixed :math:`s`, the old triangular sum has exactly
:math:`(N+1)(N+2)/2` rational-weight terms.  If dense degree-:math:`n`
integer multiplication costs :math:`\mathsf M(n)`, the homogeneous
evaluation uses :math:`O(\mathsf M(N+1)\log(N+2))` integer operations
plus :math:`O(N+1)` fraction operations.  The supplied balanced Karatsuba
multiplier gives the stronger bound
:math:`O((N+1)^{\log_2 3})` integer operations for the transform, since
both recursive levels have exponent strictly larger than one.  This is
a subquadratic arithmetic count, not a constant number of operations.
The arrays and the cached powers contain :math:`O(N+1)` integer entries
at peak for fixed :math:`s`; integer storage in bytes grows with precision.

Bit lengths are not constant.  The displayed harmonic formula shows that
every diagonal denominator divides :math:`((N+3)!)^2`.  Thus :math:`Q`
has :math:`O(N\log(N+2))` bits.  The transform weights and binomial
coefficients add :math:`O(T)` bits.  Finally every angular denominator
divides :math:`Q(T!)^2`, so the sum can be accumulated with integer
bit lengths :math:`O(D\log(D+2))` at fixed :math:`s`.  Karatsuba uses
subtraction internally, but only on integers; no floating-point
cancellation or precision choice enters the algorithm.  The mathematical
coefficient identity retains positive weights and the slower positive
double sum remains available as a separate reference.

Reproduction and limits
------------------------

``tools.risk_research.codimension_three_transform`` is an optional
standard-library checkout tool.  ``mode_averages(s,D)`` returns its
averages in increasing radial length :math:`L`, and ``coefficient(s,D)``
returns the exact fraction.  The two first target families use :math:`s=2`
and :math:`s=3`; the proof and implementation allow every :math:`s\geq2`.
No sampling or production fitting API is changed.

.. code-block:: console

   $ .venv/bin/python -m tools.risk_research.codimension_three_transform --face-size 2 --degree 128 --output .research/skeleton-risk/q3-edge-transform.json
   $ .venv/bin/python -m tools.risk_research.codimension_three_transform --face-size 3 --degree 128 --output .research/skeleton-risk/q3-triangle-transform.json
   $ .venv/bin/python -m tools.risk_research.codimension_three_transform --face-size 2 --degree 128 --verify .research/skeleton-risk/q3-edge-transform.json --output .research/skeleton-risk/q3-edge-transform-rechecked.json

Records contain canonical decimal fractions and fingerprints of the
evaluator, its relevant source modules and this proof.  They use
CRLF-to-LF normalized hashes, while before/after source checks compare
raw bytes.  Verification recomputes the exact value and compares the
complete JSON record with scalar types preserved.  Duplicate keys,
nonfinite numbers and changed fields are rejected.  Failed calculation
or verification preserves an existing output.  The resolved output is
checked against proof/evaluator sources and the saved input before
calculation and again immediately before the atomic write.  The writer uses
the validated canonical parent with the original leaf name, so retargeting
the original parent alias cannot redirect it.  Decimal
serialization does not change Python's process-wide integer-string limit.

Tests use arbitrary diagonal basis vectors to check the transform's
coefficient identity, the pre-existing double sum to check complete mode
arrays, and original scalar/Gram formulas to check risk normalization.
Larger frozen original-kernel values and post-construction holdouts are
additional finite checks.  The all-degree statement follows from the
factorial identity and divide-and-conquer proof, rather than those finite
comparisons.  No finite-degree asymptotic remainder or simultaneous
parameter limit is asserted here.
