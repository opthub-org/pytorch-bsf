Explicit vertex errors at every fixed vertex count
===================================================

Keep the uniform-simplex assumptions of :doc:`skeleton_risk_theory`.
The leading constant :math:`C_M` and its exact rational enclosures are
proved in :doc:`skeleton_risk_vertex_general`. Here the error of the
finite-degree coefficient itself is bounded, for every fixed :math:`M\geq4`.
The constants are deliberately conservative. The sharper existing
tetrahedron and five-vertex results remain useful.

Computable constants and the error theorem
------------------------------------------

For integers :math:`r\geq1` define

.. math::

   B_r=r!r^{r-1},\qquad
   G_r=2^{r+1}r\,r!(r+1)^{r-1}.

For :math:`r\geq2`, put

.. math::

   A_r&=\sum_{a=1}^{r-1}\binom ra B_aB_{r-a},\\
   F_r&=2^{r-2}\sum_{a=1}^{r-1}\binom ra
                    (G_aB_{r-a}+B_aG_{r-a}),\\
   U_r&=2^{r+1}F_r+(20+3^{r-1})A_r,\\
   V_{k,r}&=G_k\left(2^{r+2}F_r+5\,2^kA_r\right)
                 \quad(k\geq2).

Set :math:`q=M-1`, and define the following positive, explicit constants:

.. math::
   :label: risk-general-error-constants

   J_q&=2qA_{q-1},\\
   K_q&=B_q+A_q+qU_{q-1}
      +\sum_{k=2}^{q-2}\binom qk V_{k,q-k}
      +8(1+q^2+J_q),\\
   R_M&=\frac{M^2q!}{2^q}
          \left(96K_q+(273+q(q+1))J_q\right).

An empty sum is zero. In particular these formulas apply at :math:`M=4`.

.. prf:theorem:: Explicit vertex error for every fixed M at least four
   :label: thm-risk-general-vertex-explicit-error

   For all integers :math:`M\geq4` and :math:`D\geq256`,

   .. math::
      :label: risk-general-vertex-explicit-error

      \left|a_0(M,D)-\frac{C_M}{D^{M+1}}\right|
        \leq R_M\frac{(1+\log D)^{M-2}}{D^{M+2}}.

   Thus the relative error tends to zero for each fixed :math:`M`.
   There is no claim that this convergence is uniform when :math:`M`
   grows with :math:`D`. The squared logarithm at :math:`M=4` is retained;
   the sharper single logarithm known at :math:`M=5` is not assumed at
   other vertex counts.

Quantitative coefficient bounds
--------------------------------

Use the sequences :math:`\phi_r,b_r,e_r,\delta_q` of
:doc:`skeleton_risk_vertex_general`, and write :math:`L(x)=1+\log x`.
The already proved elementary symmetric formula gives

.. math::

   |\phi_r(j)|\leq B_r L(j+1)^{r-1}(j+1)^{-r}.

For :math:`j\geq1`, differentiating the binomial product at exponent
:math:`r`, rather than :math:`r-1`, gives

.. math::

   |b_r(j)|=\frac{(r!)^2(j-1)!}{(j+r)!}
       \left|E_{r-1}\left(1,\frac12,\ldots,\frac1r,
                  -1,-\frac12,\ldots,-\frac1{j-1}\right)\right|.

The total absolute reciprocal sum is at most
:math:`(r+1)L(j+1)`. Also :math:`j\geq(j+1)/2`, so

.. math::

   |b_r(j)|\leq G_r L(j+1)^{r-1}(j+1)^{-r-1}.

The bound holds at :math:`j=0` too, since :math:`b_r(0)=1`.
The series is absolutely summable and its sum is :math:`g(1)^r=0`.
Using :math:`\phi_r(j+1)-\phi_r(j)=b_r(j+1)` and the product difference
identity gives, for every :math:`j\geq0`,

.. math::
   :label: risk-general-cross-difference-bounds

   |e_r(j)|&\leq A_r L(j+1)^{r-2}(j+1)^{-r},\\
   |e_r(j+1)-e_r(j)|
       &\leq F_r L(j+1)^{r-2}(j+1)^{-r-1}.

Here :math:`L(j+2)\leq2L(j+1)` accounts for the factor :math:`2^{r-2}`
in :math:`F_r`.

We use a simple integral majorant repeatedly. For integers
:math:`p\geq0,\alpha>p+1,T\geq1`, monotonicity and differentiation give

.. math::
   :label: risk-general-log-tail

   \sum_{m=T}^{\infty}\frac{L(m+1)^p}{(m+1)^\alpha}
      \leq\int_T^\infty\frac{L(x)^p}{x^\alpha}\,dx
      \leq\frac{L(T)^p}{(\alpha-1-p)T^{\alpha-1}}.

Indeed the negative derivative of
:math:`L(x)^p/x^{\alpha-1}` is at least
:math:`(\alpha-1-p)L(x)^p/x^\alpha` on :math:`x\geq1`.
In particular :math:`\sum_j|e_r(j)|\leq2A_r`, so
:math:`0<c_q=q\sum_je_{q-1}(j)\leq J_q`.

A quantitative zero-sum convolution split
------------------------------------------

Fix :math:`n\geq2`, put :math:`N=n+1`, :math:`H=\lfloor n/2\rfloor`,
and abbreviate :math:`L=L(N)`. Since :math:`\sum_lb_k(l)=0`,

.. math::

   (b_k*e_r)(n)
    ={}&\sum_{l=0}^H b_k(l)(e_r(n-l)-e_r(n))
       -e_r(n)\sum_{l>H}b_k(l)\\
      &+\sum_{m=0}^{n-H-1}b_k(n-m)e_r(m).

For :math:`k=1`, use the exact formula
:math:`b_1(l)=-1/(l(l+1))` at positive indices. The first term is at most
:math:`2^{r+1}F_r L^{r-1}/N^{r+1}` in absolute value, because
:math:`\sum_{l=1}^H l|b_1(l)|\leq L`. The second is at most
:math:`2A_rL^{r-2}/N^{r+1}`, since its tail equals :math:`-1/(H+1)`.
For the last term, subtract
:math:`-N^{-2}\sum_{m\geq0}e_r(m)`. On its finite range,

.. math::

   \left|b_1(n-m)+N^{-2}\right|
      \leq\frac{18(m+1)}{N^3}.

This follows directly after clearing the positive denominator
:math:`N^2(n-m)(n-m+1)`: use :math:`n-m\geq N/3` and
:math:`n-m+1\geq N/2`. Also
:math:`\sum_{m\leq n}(m+1)|e_r(m)|\leq A_rL^{r-1}`.
For this last bound, take :math:`L(m+1)^{r-2}\leq L^{r-2}` outside
the sum and use :math:`\sum_{m=0}^n(m+1)^{1-r}\leq H_N\leq L`.
The missing tail starts at :math:`n-H\geq N/3`, and
:eq:`risk-general-log-tail` bounds it by
:math:`3^{r-1}A_rL^{r-2}/N^{r-1}`. Combining these bounds gives

.. math::

   \left|(b_1*e_r)(n)+N^{-2}\sum_{m\geq0}e_r(m)\right|
       \leq U_r L^{r-1}/N^3.

For :math:`k\geq2`, the same first term uses
:math:`\sum_{l=1}^H l|b_k(l)|\leq2G_kL^{k-1}`.
The absolute tail in the second term is at most
:math:`2^kG_kL^{k-1}/N^k`. In the last term use
:math:`|b_k(n-m)|\leq2^{k+1}G_kL^{k-1}/N^{k+1}` and
:math:`\sum_m|e_r(m)|\leq2A_r`. Since :math:`r,k\geq2`, this proves

.. math::

   |(b_k*e_r)(n)|\leq V_{k,r}L^{k+r-3}/N^3.

Substitute these two bounds into
:eq:`risk-general-transformed-diagonal`. The :math:`\phi_q` and
:math:`e_q` terms are bounded by :math:`(B_q+A_q)L^{q-1}/N^3`.
The added term :math:`8(1+q^2+J_q)` in :math:`K_q` also covers
:math:`n=0,1`: here :math:`\delta_q(0)=1` and
:math:`\delta_q(1)=(q^2-3q+4)/4`. Therefore, for every :math:`n\geq0`,

.. math::
   :label: risk-general-diagonal-explicit-error

   \left|\delta_q(n)-\frac{c_q}{(n+1)^2}\right|
      \leq K_q\frac{L(n+1)^{q-1}}{(n+1)^3}.

Averaging and normalization
----------------------------

Use the distribution :math:`\mathbb P(J=j)=W_{D,j}/Z_D` from
:doc:`skeleton_risk_vertex_general`. Its already proved moments and tail
bound are, for :math:`D\geq12`,

.. math::

   \mathbb E[J-D/2]&=-1/(2D),\\
   \mathbb E[(J-D/2)^2]&\leq D/4,\\
   \mathbb P(J\leq\lfloor D/4\rfloor)&\leq\epsilon_D=4\,2^{-D/12}.

For :math:`r(t)=(t+1)^{-2}`, the derivative bounds
:math:`|r'(t)|\leq2/t^3`, :math:`|r''(t)|\leq6/t^4` and
:math:`|r(t)-t^{-2}|\leq2/t^3` hold at :math:`t>0`.
Taylor expansion at :math:`D/2` on the central event gives
:math:`192/D^3` from the quadratic remainder, :math:`1/D^3` from
the mean, and :math:`16/D^3` from the shift of :math:`r(D/2)`.
The omitted Taylor polynomial is bounded by :math:`12/D^2`;
the omitted true value is bounded by one. Consequently

.. math::

   \left|\mathbb E r(J)-4/D^2\right|\leq209/D^3+2\epsilon_D.

On the central event, :math:`J+1>D/4`; on its complement use
:math:`L(J+1)\leq L(D)` in :eq:`risk-general-diagonal-explicit-error`.
Together with :math:`c_q\leq J_q`, this yields

.. math::

   \left|\mathbb E\delta_q(J)-4c_q/D^2\right|
      \leq L(D)^{q-1}\left(
         \frac{64K_q+209J_q}{D^3}+(K_q+2J_q)\epsilon_D\right).

For :math:`D\geq256`, :math:`D^3\epsilon_D\leq32`, as in the
five-vertex error proof: the expression decreases on that domain and
its value at 256 is below 32. Finally write the exact prefactor as

.. math::

   a_0(M,D)&=\frac{M^2q!}{2^qD^q}\,Q_D\,\mathbb E\delta_q(J),\\
   Q_D&=\prod_{h=1}^q(1+h/(2D))^{-1},\qquad
   0<Q_D\leq1,\quad1-Q_D\leq q(q+1)/(4D).

The normalization error adds at most
:math:`q(q+1)J_q L(D)^{q-1}/D^3` inside the parentheses.
Since :math:`C_M=4M^2q!c_q/2^q`, the resulting constant is exactly
:math:`R_M` in :eq:`risk-general-error-constants`. This proves the theorem.

Certified computation and finite checks
----------------------------------------

``tools.risk_research.vertex_general_asymptotic`` replaces
:math:`L(D)` by the exact integer :math:`1+\lceil\log_2D\rceil`.
It combines the rational enclosure of :math:`C_M` with both signs of the
analytic error, clipping only a negative lower endpoint to zero.
Below degree256 it uses the existing exact univariate construction.
Optional Arb output also includes arithmetic rounding, restores ambient
precision and is tested again after JSON serialization.

.. code-block:: console

   $ python -m tools.risk_research.vertex_general_asymptotic --vertices 6 --degree 100000 --output .research/vertex6-error.json
   $ python -m tools.risk_research.vertex_general_asymptotic --vertices 6 --degree 100000 --with-arb --bits 128 --relative-target 1/100 --output .research/vertex6-ball.json

The optional relative target returns a power-of-two threshold beyond which
the reported analytic bound is at most that fraction of the leading term.
If :math:`C_- >0` is the constant's certified lower endpoint, choose
:math:`t\geq\max(8,2(M-2))` with

.. math::

   R_M(t+2)^{M-2}\leq\eta C_-2^t.

Then all :math:`D\geq2^t` satisfy the target. Indeed, writing
:math:`x=\log_2D`, the integer logarithm is at most :math:`x+2`, and
:math:`(x+2)^{M-2}2^{-x}` decreases on that domain since
:math:`\log2>1/2`. A zero lower constant endpoint requires more terms.
The threshold can be very large: this documents the usefulness of the
conservative bound separately from its correctness.

Independent tests compare coefficient bounds and their differences with
exact logarithmic series, test the zero-sum splits against exact rational
convolutions, compare enclosures with the original scalar/Gram references
and frozen independent six-vertex kernel values, and reparse low- and
high-precision real balls. Such finite checks validate the implementation;
the all-degree claims follow from the inequalities above.
