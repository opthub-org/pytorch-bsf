Single-log vertex errors for every fixed M at least five
========================================================

Keep the uniform-simplex assumptions of :doc:`skeleton_risk_theory`.
The general leading constant and its rational enclosures are proved in
:doc:`skeleton_risk_vertex_general`. The explicit error in
:doc:`skeleton_risk_vertex_general_asymptotics` has logarithmic power
:math:`M-2`. Finite first absolute moments remove this dependence from
the logarithmic power, for every fixed :math:`M\geq5`.
The existing critical :math:`M=4` squared-log theorem and sharper
five-vertex evaluation remain unchanged.

Explicit constants and theorem
-------------------------------

Use the positive integers :math:`B_r,G_r,A_r,F_r` of
:doc:`skeleton_risk_vertex_general_asymptotics`. Define, for integers
:math:`p\geq0,\alpha\geq2`,

.. math::

   I_{p,\alpha}
     =\sum_{j=0}^p\frac{p!}{(p-j)!(\alpha-1)^{j+1}}
     =\int_1^\infty\frac{(1+\log x)^p}{x^\alpha}\,dx.

For :math:`k\geq2,r\geq3`, put

.. math::

   S_k&=G_k(1+I_{k-1,k}),\\
   T_r&=A_r(1+I_{r-2,r-1}),\\
   \widetilde U_r
      &=2^{r+1}F_r+(2+3^{r-1})A_r+18T_r.

For :math:`k,r\geq2`, define

.. math::

   \widetilde V_{k,r}
      =2^{r+1}F_rS_k+5\,2^kG_kA_r.

Set :math:`q=M-1\geq4` and

.. math::
   :label: risk-refined-vertex-error-constants

   J_q&=2qA_{q-1},\\
   \widetilde K_q
      &=2B_q+A_q+q\widetilde U_{q-1}
        +\sum_{k=2}^{q-2}\binom qk\widetilde V_{k,q-k}
        +8(1+q^2+J_q),\\
   \widetilde R_M
      &=\frac{M^2q!}{2^q}
          \left(96\widetilde K_q+(273+q(q+1))J_q\right).

All these constants are positive computable rational numbers.

.. prf:theorem:: Single-log vertex error for every fixed M at least five
   :label: thm-risk-refined-vertex-explicit-error

   For every pair of integers :math:`M\geq5,D\geq256`,

   .. math::
      :label: risk-refined-vertex-explicit-error

      \left|a_0(M,D)-\frac{C_M}{D^{M+1}}\right|
         \leq\widetilde R_M\frac{1+\log D}{D^{M+2}}.

   This is an explicit finite-degree bound for each fixed :math:`M`.
   It does not assert uniform convergence when :math:`M` grows with
   :math:`D`.

First absolute moments and elementary absorptions
---------------------------------------------------

Write :math:`L(x)=1+\log x`. Integration by parts proves the formula
for :math:`I_{p,\alpha}`. When :math:`\alpha\geq p`, the integrand is
decreasing on :math:`x\geq1`. Therefore a sum over positive integers
is at most its first term plus that integral. The already proved
coefficient bounds give

.. math::

   \sum_{l\geq0}l|b_k(l)|&\leq S_k\quad(k\geq2),\\
   \sum_{m\geq0}(m+1)|e_r(m)|&\leq T_r\quad(r\geq3).

The second displayed bound starts at :math:`r=3`: for :math:`r=2`,
:math:`e_2(m)=2/(m+1)^2` has a divergent first absolute moment.
Similarly the first absolute moment of :math:`b_1` diverges, but its
partial sum through :math:`n` is at most :math:`L(n+1)`.
These two critical cases explain why :math:`M=4` remains separate.

The following elementary inequalities hold on :math:`x\geq1`:

.. math::

   1\leq L(x)\leq x,\qquad L(x)^2/x\leq2.

For the second inequality set :math:`t=\log x\geq0`; the maximum of
:math:`(1+t)^2e^{-t}` is :math:`4/e<2` at :math:`t=1`.
The spare inverse powers in the convolution bounds thus absorb every
logarithm beyond the single critical factor.

A refined zero-sum split
-------------------------

For :math:`n\geq2`, put :math:`N=n+1`, :math:`H=\lfloor n/2\rfloor`
and :math:`L=L(N)`. Use the same exact zero-sum decomposition as in
:doc:`skeleton_risk_vertex_general_asymptotics`:

.. math::

   (b_k*e_r)(n)
      ={}&\sum_{l=0}^H b_k(l)(e_r(n-l)-e_r(n))
          -e_r(n)\sum_{l>H}b_k(l)\\
        &+\sum_{m=0}^{n-H-1}b_k(n-m)e_r(m).

For :math:`k=1,r\geq3`, subtract
:math:`-N^{-2}\sum_{m\geq0}e_r(m)` from the last term.
The four resulting error budgets are respectively

.. math::

   &2^{r+1}F_rL^{r-1}/N^{r+1},\qquad
     2A_rL^{r-2}/N^{r+1},\\
   &18T_r/N^3,\qquad
     3^{r-1}A_rL^{r-2}/N^{r+1}.

The first two use the first-difference estimate for :math:`e_r`,
the exact identity :math:`b_1(l)=-1/(l(l+1))`, and its telescoping
tail. The finite last-term difference uses the previously proved
:math:`|b_1(n-m)+N^{-2}|\leq18(m+1)/N^3` and the finite moment
:math:`T_r`. The omitted tail starts at :math:`n-H\geq N/3` and
uses the previous logarithmic tail inequality.
Since :math:`r\geq3`, the absorptions above prove

.. math::

   \left|(b_1*e_r)(n)+N^{-2}\sum_{m\geq0}e_r(m)\right|
      \leq\widetilde U_rL/N^3.

For :math:`k,r\geq2`, the three absolute error budgets are

.. math::

   &2^{r+1}F_rS_kL^{r-2}/N^{r+1},\\
   &2^kG_kA_rL^{k+r-3}/N^{k+r},\\
   &2^{k+2}G_kA_rL^{k-1}/N^{k+1}.

Here the first term uses the finite first moment :math:`S_k`,
the second uses the logarithmic tail of :math:`b_k`, and the last
uses :math:`\sum_m|e_r(m)|\leq2A_r` and :math:`n-m+1\geq N/2`.
The ratios :math:`(L/N)^{r-2}` and :math:`(L/N)^{k+r-3}` are at most
one; the last term leaves at most one factor :math:`L`.
Consequently

.. math::

   |(b_k*e_r)(n)|\leq\widetilde V_{k,r}L/N^3.

In the transformed diagonal identity, the residual terms satisfy

.. math::

   |\phi_q(n)|\leq2B_qL/N^3,\qquad
   |e_q(n)|\leq A_qL/N^3\quad(q\geq4).

For the first inequality factor
:math:`L^{q-2}/N^{q-3}=(L^2/N)(L/N)^{q-4}\leq2`.
For the second use :math:`(L/N)^{q-3}\leq1`.
The :math:`k=1` term supplies :math:`c_q=q\sum_me_{q-1}(m)`.
The added :math:`8(1+q^2+J_q)` covers :math:`n=0,1` using
:math:`\delta_q(0)=1`, :math:`\delta_q(1)=(q^2-3q+4)/4` and
:math:`0<c_q\leq J_q`. Thus for every :math:`n\geq0`,

.. math::
   :label: risk-refined-diagonal-explicit-error

   \left|\delta_q(n)-\frac{c_q}{(n+1)^2}\right|
      \leq\widetilde K_q\frac{L(n+1)}{(n+1)^3}.

Averaging and normalization
----------------------------

Apply the hypergeometric averaging proof of
:doc:`skeleton_risk_vertex_general_asymptotics` to
:eq:`risk-refined-diagonal-explicit-error`. Only the diagonal error
changes. Its logarithmic factor is now :math:`L(D)`.
The central event, Taylor remainder and small-index tail give

.. math::

   \left|\mathbb E\delta_q(J)-4c_q/D^2\right|
      \leq L(D)\left(
         \frac{64\widetilde K_q+209J_q}{D^3}
           +(\widetilde K_q+2J_q)4\,2^{-D/12}\right).

For :math:`D\geq256`, :math:`D^3\,4\,2^{-D/12}\leq32`.
Multiplying the exact prefactor, including its normalization error
:math:`q(q+1)J_q/D^3`, gives
:eq:`risk-refined-vertex-explicit-error` with precisely the constant
:eq:`risk-refined-vertex-error-constants`.

Certified intervals and a six-vertex improvement
-------------------------------------------------

The checkout tool ``tools.risk_research.vertex_refined_asymptotic`` uses
the exact integer :math:`1+\lceil\log_2D\rceil` in place of
:math:`1+\log D`. It reports the refined and previous analytic radii
separately, uses their minimum, and combines it with the exact rational
interval for :math:`C_M`. Below :math:`D=256` it evaluates the exact
univariate formula. Optional Arb output adds arithmetic rounding,
restores ambient precision, and is checked again after serialization.

At :math:`M=6`, the explicit constants are

.. math::

   \widetilde K_5=107393368,\qquad J_5=5280,\qquad
   \widetilde R_6=1392034027680.

With the existing 128-term rational interval for :math:`C_6`, the new
analytic error is at most one percent of :math:`C_6/D^7` at every
:math:`D\geq2^{38}`. The earlier sufficient threshold was
:math:`2^{56}`. These are sufficient thresholds, with conservative
constants, rather than exact smallest useful degrees.

In general, for a certified positive constant endpoint :math:`C_-`
and target :math:`\eta>0`, choose an integer :math:`t\geq8` such that

.. math::

   \widetilde R_M(t+2)\leq\eta C_-2^t.

Then every :math:`D\geq2^t` satisfies the target: writing
:math:`x=\log_2D`, the integer logarithmic majorant is at most
:math:`x+2`, and :math:`(x+2)2^{-x}` is decreasing for :math:`x\geq8`.
This certifies the entire tail, including integer-logarithm jumps.

.. code-block:: console

   $ python -m tools.risk_research.vertex_refined_asymptotic --vertices 6 --degree 100000 --relative-target 1/100 --output .research/vertex6-refined.json
   $ python -m tools.risk_research.vertex_refined_asymptotic --vertices 6 --degree 100000 --with-arb --bits 128 --output .research/vertex6-refined-ball.json

The JSON separates constant truncation, analytic error and optional
arithmetic rounding. Source hashes use the explicit CRLF-to-LF policy;
the concurrent-edit guard compares the original bytes and rejects
even a newline-only change during calculation. Existing output is
replaced atomically only after successful computation.

Independent checks cover exact logarithmic integrals, finite moments,
both convolution parities, low-degree scalar and Gram references,
the asymptotic boundary, native serialized balls and post-freeze
holdouts. Finite checks support the implementation; the all-degree
result follows from the inequalities above.
