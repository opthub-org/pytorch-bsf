Fixed-degree risk comparisons in every dimension
================================================

This chapter uses the uniform-simplex design, evaluation and noise
assumptions of :doc:`skeleton_risk_theory`. Write :math:`M\geq2` for the
number of vertices and :math:`D\geq1` for the degree. The optimal risk of
dimension-based inductive fitting and the all-at-once risk are

.. math::

   R_{\mathrm{ind}}(M,D)
     =\left(\sum_{s=1}^{\min(D,M)}\sqrt{a_{s-1}(M,D)}\right)^2,
   \qquad
   R_{\mathrm{all}}(M,D)=\binom{M+D-1}{D}.

The polynomial dependence on :math:`M` in
:prf:ref:`thm-skeleton-rational` permits a complete comparison at each
listed degree without requiring a bounded-length tetrahedron vertex
formula or an explicit vertex degree recurrence. The positive single sum
is proved separately in :doc:`skeleton_risk_vertex`.
The proof below uses exact integers and rational numbers,
including the square-root enclosures; it needs no floating-point error
estimate or paid computer-algebra system.

Complete comparisons through degree twenty
------------------------------------------

.. prf:theorem:: Fixed-degree comparison for degrees nine through twenty
   :label: thm-risk-comparisons-nine-twenty

   For every row of the following table,
   :math:`R_{\mathrm{ind}}(M,D)<R_{\mathrm{all}}(M,D)` holds exactly in
   the two displayed ranges. In the remaining integer range the reverse
   inequality is strict. There are no ties.

   .. list-table:: All vertex counts, not a finite dimension cutoff
      :header-rows: 1

      * - Degree :math:`D`
        - First winning range
        - Second winning range
        - Certified infinite tail starts at
      * - 9
        - :math:`2\leq M\leq4`
        - :math:`M\geq71`
        - 115
      * - 10
        - :math:`2\leq M\leq4`
        - :math:`M\geq81`
        - 131
      * - 11
        - :math:`2\leq M\leq4`
        - :math:`M\geq92`
        - 146
      * - 12
        - :math:`2\leq M\leq5`
        - :math:`M\geq103`
        - 162
      * - 13
        - :math:`2\leq M\leq5`
        - :math:`M\geq114`
        - 178
      * - 14
        - :math:`2\leq M\leq5`
        - :math:`M\geq126`
        - 193
      * - 15
        - :math:`2\leq M\leq5`
        - :math:`M\geq137`
        - 209
      * - 16
        - :math:`2\leq M\leq5`
        - :math:`M\geq148`
        - 225
      * - 17
        - :math:`2\leq M\leq6`
        - :math:`M\geq160`
        - 240
      * - 18
        - :math:`2\leq M\leq6`
        - :math:`M\geq171`
        - 256
      * - 19
        - :math:`2\leq M\leq6`
        - :math:`M\geq182`
        - 272
      * - 20
        - :math:`2\leq M\leq6`
        - :math:`M\geq194`
        - 288

   The result is computer-assisted. The final column is a sufficient
   bound used in the proof; it is not the actual crossover. The
   classifications for :math:`D\leq8` are in
   :prf:ref:`prop-skeleton-wins`.

.. prf:proof::

   The next three steps give the exact polynomial construction, a finite
   radical comparison, and a bound for all remaining dimensions. Their
   implementation is ``tools/risk_research/comparison.py``. Applied to
   the twelve rows, they certify 2,391 finite comparisons and twelve
   infinite tails, with no unresolved cases. Each degree has :math:`D`
   polynomial rows; the twelve degrees contain 174 rows and 2,666
   rational coefficients. No shape for the losing set is assumed.

Complete comparisons at degrees twenty-one through thirty
----------------------------------------------------------

.. prf:theorem:: Fixed-degree comparison for degrees twenty-one through thirty
   :label: thm-risk-comparisons-twentyone-thirty

   For every row below, inductive fitting wins exactly in the two displayed
   ranges. All-at-once fitting wins strictly at every remaining integer
   vertex count. There are no ties.

   .. list-table:: Complete classifications for every integer vertex count
      :header-rows: 1

      * - Degree :math:`D`
        - First winning range
        - Second winning range
        - Certified infinite tail starts at
      * - 21
        - :math:`2\leq M\leq6`
        - :math:`M\geq205`
        - 304
      * - 22
        - :math:`2\leq M\leq6`
        - :math:`M\geq217`
        - 319
      * - 23
        - :math:`2\leq M\leq7`
        - :math:`M\geq228`
        - 335
      * - 24
        - :math:`2\leq M\leq7`
        - :math:`M\geq240`
        - 351
      * - 25
        - :math:`2\leq M\leq7`
        - :math:`M\geq252`
        - 367
      * - 26
        - :math:`2\leq M\leq7`
        - :math:`M\geq263`
        - 382
      * - 27
        - :math:`2\leq M\leq7`
        - :math:`M\geq275`
        - 398
      * - 28
        - :math:`2\leq M\leq7`
        - :math:`M\geq286`
        - 414
      * - 29
        - :math:`2\leq M\leq8`
        - :math:`M\geq298`
        - 430
      * - 30
        - :math:`2\leq M\leq8`
        - :math:`M\geq310`
        - 446

.. prf:proof::

   Apply the exact construction, finite radical enclosures and infinite-tail
   argument below. These ten degrees give 255 polynomial rows with 6,585
   rational coefficients. All 3,726 finite comparisons separate at a 32-bit
   fractional grid using exact integers; the ten rational tail inequalities
   cover every remaining vertex count. No shape for the losing set is assumed.

   A separate Newton interpolation of the original scalar angular-mode
   reference, using the proved degree bounds, reproduced every polynomial.
   Additional vertex counts beyond those used in the interpolation agree
   exactly. Independent rational radical enclosures and tail bounds reproduce
   all ten classifications. Finite checks verify these computations; the
   infinite tail is covered by the displayed rational inequality.

The last column is a sufficient proof threshold, rather than the exact
crossover. Reproduce this extension with ``--degrees 21:30`` in the command
below. The earlier degree-nine-through-twenty certificates remain unchanged.


Exact construction of the dimension polynomials
-----------------------------------------------

Put :math:`s=m+1`, :math:`n=D-s`, and define

.. math::

   F(x,y)=\sum_{h,l\geq0}
     \frac{\binom{h+l}{h}}{(h+1)(l+1)}x^hy^l,
   \qquad
   T_r(h,l)=[x^hy^l](F-1)^r.

Since :math:`F-1` has no constant term, :math:`T_r(h,l)=0` when
:math:`r>h+l`. Therefore, for every integer :math:`M\geq s`,

.. math::

   K_{M-s}(h,l)=[x^hy^l]F^{M-s}
    =\sum_{r=0}^{h+l}\binom{M-s}{r}T_r(h,l).

The binomial on the right is its falling-factorial polynomial in
:math:`M`; it vanishes when :math:`r>M-s`. Truncation to
:math:`0\leq h,l\leq n` is exact because all powers are nonnegative.
Define

.. math::

   A_{k,h}=(-1)^h\binom{n-k}{h}\binom{n+k+3s-1}{h},
   \qquad
   d_s(k)=\begin{cases}
     \binom{k+s-2}{s-2},&s\geq2,\\
     \mathbf1_{k=0},&s=1.
   \end{cases}

Substitution in :eq:`skeleton-scalar-sum` constructs

.. math::

   P_{D,s-1}(M)
    ={}&\frac1{(s-1)!}\sum_{k=0}^n d_s(k)
      \sum_{h,l=0}^{n-k}
        A_{k,h}A_{k,l}(2D+s-h-l-1)!h!l!\\
     &\hspace{2em}\cdot\sum_{r=0}^{h+l}\binom{M-s}{r}T_r(h,l).

Its degree is at most :math:`2(D-s)`, and

.. math::

   a_{s-1}(M,D)=\binom Ms^2\frac{P_{D,s-1}(M)}{(M)_{2D}}
   \qquad(M\geq s).

This is a direct symbolic construction, not interpolation. For
:math:`s>M` the face does not exist and its coefficient is set to zero;
one does not evaluate the polynomial as a physical risk outside its
stated range.

A rational bound for the infinite tail
--------------------------------------

Write :math:`P_{D,s-1}(M)=\sum_i p_{s,i}M^i`. Fix an integer
:math:`L\geq\max(2,D)` and put

.. math::

   B_s(L)=\frac1{(s!)^2}\sum_{i=0}^{2(D-s)}
              |p_{s,i}|L^{i-2(D-s)}.

For all :math:`M\geq L`, the inequalities
:math:`\binom Ms\leq M^s/s!` and :math:`(M)_{2D}\geq M^{2D}` give

.. math::

   0\leq a_{s-1}(M,D)
      \leq\frac1{(s!)^2}\sum_i|p_{s,i}|M^{i-2(D-s)}
      \leq B_s(L).

The last inequality follows because every exponent is nonpositive.
More generally an absent face :math:`s>M` has coefficient zero, so also
satisfies the bound. By Cauchy--Schwarz,

.. math::

   R_{\mathrm{ind}}(M,D)\leq D\sum_{s=1}^DB_s(L).

Since :math:`\binom{M+D-1}{D}` increases with :math:`M`, the single
strict rational inequality

.. math::

   D\sum_{s=1}^DB_s(L)<\binom{L+D-1}{D}

proves that inductive fitting wins for every :math:`M\geq L`.
The certificate records each :math:`B_s(L)`, both sides, and their
strictly positive difference. The displayed tail thresholds satisfy
this inequality exactly.

Strict comparisons for the finite prefix
----------------------------------------

For every remaining integer :math:`2\leq M<L`, evaluate the active
coefficients as exact fractions. For :math:`a=u/v\geq0`, choose
:math:`K=2^b` and compute the integer

.. math::

   t=\operatorname{isqrt}\left(\left\lfloor\frac{uK^2}{v}\right\rfloor\right).

Then :math:`t/K\leq\sqrt a\leq(t+1)/K`. If
:math:`t^2v=uK^2`, the two endpoints coincide at :math:`t/K`.
Let :math:`A` and :math:`B` be the sums of the lower and upper integer
numerators. Positivity gives

.. math::

   A^2\leq K^2R_{\mathrm{ind}}(M,D)\leq B^2.

Thus :math:`B^2<K^2R_{\mathrm{all}}` certifies an inductive win, and
:math:`A^2>K^2R_{\mathrm{all}}` certifies an all-at-once win.
Overlapping bounds require higher precision. If the configured precision
limit is reached, certification fails and no all-dimension result is
reported. An exact equality is therefore never silently included among
strict wins or losses.

All 2,391 finite comparisons in the theorem separate at :math:`b=32`.
This is a 32-bit fractional grid for the square roots, with arbitrarily
large integer arithmetic throughout; it is not a 32-bit floating-point
calculation.

Reproduction and verification
-----------------------------

From a source checkout with the development environment available, run

.. code-block:: console

   $ .venv/bin/python -m tools.risk_research.comparison --degrees 9:20 --output .research/skeleton-risk/comparisons.json

The command needs only the Python standard library. ``--degrees 9``
restricts it to the first new degree. ``--max-tail`` bounds the search for
a sufficient infinite-tail inequality, and ``--max-bits`` bounds finite
comparison precision. A failure preserves an existing output file.

The JSON certificate contains exact numerator/denominator decimal strings
for the polynomials, the dyadic endpoints and strict sign margin for
every finite :math:`M`, the tail inequalities, and the SHA256 of the
source used. Polynomial rows are ordered by :math:`s=1,\ldots,D`, with
ascending powers of :math:`M` within each row. The output is
deterministic for fixed source and arguments.

Independent validation includes equality with the orbit-counted
polynomials through degree eight, the original scalar-sum reference,
the rational Gram-propagation solver at degree nine on a triangle, all
triangle closed forms at degree twenty, and the independently derived
large-dimension limits. A separate exact interpolation of the scalar
reference, using the proved degree bounds and additional check points,
also reproduced all 174 polynomial rows and all comparison certificates
for degrees nine through twenty.

The extension to degrees twenty-one through thirty independently reproduces
another 255 polynomial rows, 6,585 rational coefficients, 3,726 finite
comparisons and ten infinite tails by Newton interpolation of the original
scalar reference. The degree-thirty regression also checks the previously
proved triangle, tetrahedron-vertex, five-vertex-vertex and five-vertex-face
formulas, together with every independently derived large-dimension limit.

These results give complete fixed-degree comparisons through thirty.
They do not assert a uniform formula for the two winning ranges as
functions of :math:`D`, or settle the remaining all-degree vertex
evaluations.
