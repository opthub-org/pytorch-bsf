A first correction with a quadratic large-vertex remainder
==========================================================

Keep the uniform-simplex assumptions, the fixed-degree-limit constant
:math:`C_M`, and the leading term :math:`B_M` from
:doc:`skeleton_risk_vertex_large_dimension`. The following expansion concerns
:math:`C_M` as the vertex count grows; it asserts no joint :math:`M,D`
limit and no finite-sample risk approximation.

.. prf:theorem:: First correction and an explicit second-order remainder
   :label: thm-risk-large-vertex-second-order

   Put :math:`n=M-2`, :math:`L=\log2`, and

   .. math::

      \beta=\frac{28L^2-18L+1}{8(2L-1)^2}.

   For every integer :math:`M\geq258`,

   .. math::
      :label: risk-large-vertex-second-order-error

      \left|\frac{C_M}{B_M}-1-\frac\beta n\right|
                \leq\frac{84}{n^2}.

   The coefficient satisfies :math:`13/8<\beta<53/32`.
   Consequently :math:`84/n^2` also bounds the relative error against
   the corrected leading term :math:`B_M(1+\beta/n)`.
   The earlier radius :math:`6/n` concerns approximation by :math:`B_M`
   alone; the centers are different.

.. prf:proof::

   **The correction is derived before numerical evaluation.** Use the exact
   normalization already established in
   :doc:`skeleton_risk_vertex_large_dimension_refined`:

   .. math::

      C_M/B_M=\sqrt{1+1/n}\,I_n/G_n,\qquad
      I_n=\int_0^\pi f(x)^nA(x)\,dx,\qquad
      G_n=\frac{\sqrt\pi}{2\sqrt{\alpha n}},

   where

   .. math::

      h(x)&=4\cos^2(x/2)\log(2\cos(x/2))+x\sin x,\\
      f(x)&=h(x)/(4L),\quad A(x)=\sec^2(x/2),\quad
      \alpha=\frac{2L-1}{8L}.

   Define
   :math:`a_4=(4L-3)/(192L)` and
   :math:`\lambda=a_4-\alpha^2/2<0`.
   The first two terms of the positive logarithm series give
   :math:`L>2/3+2/81=56/81>29/42`. Together with
   :math:`L<7/10` from the refined chapter, this implies
   :math:`2/29<\alpha<1/14`.

   **Sixth-order control on the fixed local interval.** For
   :math:`0\leq x\leq1`, write :math:`c=\cos x` and
   :math:`\ell=\log(2\cos(x/2))`. Twice differentiating the previously
   proved fourth derivative gives the exact identity

   .. math::

      h^{(6)}(x)=2c-x\sin x-2c\ell+2\sin x\tan(x/2)
          -\frac{c}{1+c}-\frac{c}{(1+c)^2}
          -\frac{2\sin^2x}{(1+c)^3}.

   Here :math:`c\geq1/2`, :math:`0<\ell<7/10`, and
   :math:`0\leq\tan(x/2)\leq4/7`.
   The absolute values of the seven terms sum to at most
   :math:`2+1+7/5+8/7+1/2+1/4+16/27<8`.
   Taylor's theorem and evenness therefore give

   .. math::

      |f(x)-1+\alpha x^2-a_4x^4|\leq x^6/240.

   Retain the signed bounds proved in the refined chapter. With
   :math:`u=1-f(x)` and :math:`d=7/832`, they are

   .. math::

      \alpha x^2\leq u\leq x^2/13,\quad
      |u-\alpha x^2|\leq x^4/192,\quad
      -dx^4\leq\log f(x)+\alpha x^2\leq0.

   Since :math:`u+\alpha x^2\leq(1/13+1/14)x^2`, expansion of
   :math:`\log(1-u)` through its quadratic term yields

   .. math::

      |\log f(x)+\alpha x^2-\lambda x^4|
       &\leq\left(\frac1{240}+\frac{27}{69888}+\frac1{6084}\right)x^6\\
       &=\frac{64289}{13628160}x^6<\frac{x^6}{200}.

   The last summand bounds
   :math:`\sum_{k\geq3}u^k/k\leq u^3/(3(1-u))`.
   Taking the limit at zero in the signed fourth-order bound also gives
   :math:`|\lambda|\leq d`.
   For the amplitude, put :math:`t=x/2`. From
   :math:`t\leq\tan t\leq8t/7` one obtains
   :math:`0\leq\tan t-t\leq64t^3/147`. Hence

   .. math::

      0\leq A(x)-1-x^2/4
       \leq(20/343)x^4<x^4/16,\qquad
      A\leq4/3,\quad A-1\leq x^2/3.

   **A Gaussian expansion with a quantified error.** Set
   :math:`w=\log f+\alpha x^2\leq0`. For :math:`v\leq0`,
   :math:`|e^v-1-v|\leq v^2/2`. The preceding bounds imply

   .. math::

      \left|f(x)^nA(x)
          -e^{-\alpha nx^2}(1+x^2/4+n\lambda x^4)\right|
      \leq e^{-\alpha nx^2}\left(
          \frac{nx^6}{150}+\frac{2d^2n^2x^8}{3}
                  +\frac{x^4}{16}+\frac{dnx^6}{3}\right).

   Extend this positive majorant to infinity. Using
   :math:`G_n^{-1}\int_0^\infty x^{2k}e^{-\alpha nx^2}dx
   =(2k-1)!!/(2\alpha n)^k`, its integral divided by
   :math:`G_n` is bounded by :math:`K_0/n^2`, where

   .. math::

      K_0=\frac1{80\alpha^3}+\frac{35d^2}{8\alpha^4}
                 +\frac3{64\alpha^2}+\frac{5d}{8\alpha^3}
       <\frac{34415865391}{443023360}<80.

   The last rational value substitutes :math:`\alpha=2/29`.

   **Both omitted tails are bounded separately.** The old monotonicity
   and exact second moment :math:`I_2<12`, together with
   :math:`f(1)<15/16` and :math:`G_n>3/\sqrt n`, give

   .. math::

      G_n^{-1}\int_1^\pi f(x)^nA(x)dx
          \leq4\sqrt n e^{-(n-2)/16}\leq1/n^2\quad(n\geq256).

   Indeed :math:`n^{5/2}e^{-(n-2)/16}` decreases for :math:`n\geq40`,
   and at 256 the assertion follows from
   :math:`22L<77/5<254/16` and
   :math:`4\,256^{5/2}=2^{22}`.

   For the Gaussian polynomial put :math:`a=\alpha n>n/16`.
   Integration by parts bounds the tails of powers 0, 2 and 4 by
   :math:`e^{-a}/(2a)` times, respectively,
   :math:`1`, :math:`1+1/(2a)`, and
   :math:`1+3/(2a)+3/(4a^2)`.
   Since :math:`d<1/100`, their sum with weights
   :math:`1,1/4,nd` is bounded, for :math:`n\geq256`, by

   .. math::

      1+\tfrac14(1+8/n)+nd(1+24/n+192/n^2)
           \leq149/100+n/100+98/(25n)<n/32.

   Consequently the absolute Gaussian-polynomial tail divided by
   :math:`G_n` is at most
   :math:`\sqrt n e^{-n/16}/12\leq1/n^2`.
   The same monotonicity proves the last bound from its value at 256;
   use :math:`e^{16}>2^{22}`.
   Combining both tails with the local integral proves

   .. math::

      \left|I_n/G_n-1-b/n\right|\leq82/n^2,\qquad
      b=\frac1{8\alpha}+\frac{3\lambda}{4\alpha^2}.

   **Exact normalization and the first correction.** Algebra gives
   :math:`\beta=b+1/2` with the displayed formula. Its derivative as
   a function of :math:`L` is
   :math:`-(10L-7)/(4(2L-1)^3)>0` on :math:`(2/3,7/10)`.
   Evaluation at the endpoints gives :math:`13/8<\beta<53/32`, so
   :math:`0<b<2`.
   Taylor's theorem gives
   :math:`|\sqrt{1+1/n}-1-1/(2n)|\leq1/(8n^2)`.
   Thus, for :math:`n\geq256`,

   .. math::

      \left|C_M/B_M-1-\beta/n\right|
       \leq\frac{83+1/8+(41+1/4)/n}{n^2}<\frac{84}{n^2}.

   This includes the square-root normalization rather than absorbing it
   into an unspecified correction. Division by :math:`1+\beta/n>1`
   proves the corrected relative-error assertion.

Certified evaluation and replay
-------------------------------

The standard-library module
``tools.risk_research.vertex_large_dimension_second_order`` encloses
:math:`\beta` by exact rational evaluation at the endpoints of the
existing logarithm-series enclosure. Monotonicity is used only on the
proved interval :math:`2/3<L<7/10`. For every :math:`M\geq258` it encloses
:math:`C_M` after removing
:math:`M^2(M-1)!(2\log2)^{M-1}`. The uncertainty in the logarithm and
leading normalization is separate from the analytic radius :math:`84/n^2`.
For smaller :math:`M` the new evaluator rejects the request; the earlier
tool supplies its existing valid fallback.

For positive rational :math:`\varepsilon`, the threshold routine returns

.. math::

   \max\left(258,\ 2+\left\lceil\sqrt{84/\varepsilon}\right\rceil\right).

It guarantees the error relative to the exact corrected leading term at
every larger integer vertex count. One percent is therefore guaranteed
throughout the new theorem's domain. Independently,
:math:`|C_M/B_M-1|\leq53/(32n)+84/n^2<1/100` already at :math:`n=256`,
and the right side decreases thereafter; the uncorrected leading term
also has a one-percent sufficient start of :math:`M=258`.
These are sufficient starts, not minimal classifications.

.. code-block:: console

   $ .venv/bin/python -m tools.risk_research.vertex_large_dimension_second_order --vertices 258 --bits 47 --relative-target 1/1000 --output .research/skeleton-risk/second-order258.json
   $ .venv/bin/python -m tools.risk_research.vertex_large_dimension_second_order --verify .research/skeleton-risk/second-order258.json

The new record schema stores the vertex count as a canonical decimal
string, while precision and backend flags retain strict integer and
boolean types. Rational fields use canonical decimal numerator and
denominator strings, with local chunked conversion for very large
integers. No old schema is changed. Verification accepts no generation
overrides, restores recorded precision/target/optional flags, and rejects
duplicate keys, extra fields, type changes, source edits and altered bounds.
CRLF-to-LF fingerprints and raw source guards follow the existing tools.

Output is written atomically in a caller-controlled local directory.
Generation rejects source/proof aliases before calculation and again
at the canonical parent plus original leaf name before writing.
An unrelated leaf symlink is replaced without changing its target.
There is no sandbox guarantee against a separate process renaming or
replacing that canonical directory itself.

Optional Arb restores the factorial/exponential scale. The optional
``--integral-check`` calls the original positive integral without using
the new expansion, restores working precision, and rounds saved balls
outward. Finite integral holdouts and rational-series normalization checks
are implementation evidence; the proof above supplies the infinite tail.
No fitted coefficient, production sampling API, mandatory dependency,
third correction, or growing-degree limit is introduced.
