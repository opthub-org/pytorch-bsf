Five-vertex vertex risk
=======================

The additive-kernel transform from :doc:`skeleton_risk_vertex` also resolves
the vertex coefficient on a simplex with five vertices. The result is one
sum with four harmonic states, and its leading constant involves
:math:`\zeta(2)-\zeta(3)`. A positivity argument for the transformed
diagonals applies to every fixed vertex count.

All statements use the uniform-simplex assumptions of
:doc:`skeleton_risk_theory`. The sum has :math:`D` terms. It is not a
bounded-length classical-special-function expression, and this page does
not assert an explicit recurrence in :math:`D`.
The broader construction and fixed-dimension leading constants are in
:doc:`skeleton_risk_vertex_general`. The five-vertex leading term has an
explicit remainder bound in :doc:`skeleton_risk_five_vertex_asymptotics`.

An exact harmonic weight
------------------------

Write :math:`H_j^{(r)}=\sum_{k=1}^j k^{-r}`, :math:`H_j=H_j^{(1)}`,
and define the depth-two finite harmonic sum

.. math::

   A_j=\sum_{k=1}^j\frac{H_{k-1}}{k^2}.

All four states vanish at zero. They update by
:math:`H_j^{(r)}=H_{j-1}^{(r)}+j^{-r}` and
:math:`A_j=A_{j-1}+H_{j-1}/j^2`; no inner summation is needed during
evaluation. For :math:`n=j+1`, abbreviate
:math:`H=H_j`, :math:`H_2=H_j^{(2)}`, :math:`H_3=H_j^{(3)}`, and
:math:`A=A_j`. Define

.. math::
   :label: risk-five-vertex-diagonal

   \Psi_j={}&\frac{24H^3}{n(n+1)(n+2)(n+3)}\\
    &-\frac{12(5n^3+7n^2-6n-12)H^2}
             {n^2(n+1)^2(n+2)^2(n+3)}\\
    &+\frac{24(2n+3)HH_2}{n(n+1)(n+2)(n+3)}\\
    &-\frac{24(2n^4+15n^3+44n^2+47n+24)H}
             {n^2(n+1)^2(n+2)^2(n+3)}\\
    &+\frac{12(4n^4+22n^3+41n^2+26n+12)H_2}
             {n^2(n+1)(n+2)^2(n+3)}\\
    &-\frac{48H_3}{n(n+1)(n+3)}
      -\frac{48A}{n(n+2)}
      +\frac{144}{n^2(n+1)^2(n+2)^2(n+3)}.

.. prf:theorem:: Five-vertex vertex coefficient at every degree
   :label: thm-risk-five-vertex-single

   For every integer :math:`D\geq1`, let
   :math:`W_{D,j}=\binom{D-1}j\binom{D+1}j` and
   :math:`Z_D=\binom{2D}{D-1}`. Then

   .. math::
      :label: risk-five-vertex-single

      a_0(5,D)=\frac{600}
       {Z_D(2D+1)(2D+2)(2D+3)(2D+4)}
       \sum_{j=0}^{D-1}W_{D,j}\Psi_j.

   Every :math:`\Psi_j` is strictly positive, with :math:`\Psi_0=1`.
   All denominators in the formula are nonzero on the stated domain.

.. prf:proof::

   Put :math:`g(t)=-(1-t)\log(1-t)/t` and
   :math:`\phi_r(j)=[t^j]g(t)^r/(1-t)`. Besides the three formulas in
   :doc:`skeleton_risk_vertex`, differentiating
   :math:`\binom\alpha{j+4}` four times at :math:`\alpha=3` gives

   .. math::

      \phi_4(j)=\frac{24\left(
       -H^3+\tfrac{11}2H^2-6H+1
       +(3H-\tfrac{11}2)H_2-2H_3\right)}
       {n(n+1)(n+2)(n+3)}.

   For sequences, let :math:`*` denote ordinary convolution and let
   :math:`g_k=[t^k]g(t)`. Expand the additive kernel to the fourth power.
   Terms in which one of :math:`g(x),g(y)` is absent combine by
   convolution. The result is

   .. math::
      :label: risk-five-vertex-quartic-convolutions

      [x^jy^j]\frac{(g(x)+g(y)-g(xy))^4}{(1-x)(1-y)}
       =-\phi_4(j)+8\phi_1(j)\phi_3(j)+6\phi_2(j)^2
        -24\bigl(g*(\phi_1\phi_2)\bigr)_j
        +12\bigl(g^2*\phi_1^2\bigr)_j.

   Here :math:`g^2` denotes the coefficient sequence of the square of the
   series, while the products of :math:`\phi` sequences are pointwise.
   In particular, the coefficient :math:`-24` includes both the
   :math:`g(x)^2g(y)g(xy)` term and its symmetric counterpart.

   The two remaining convolutions reduce to finite harmonic sums. Since

   .. math::

      g_0=1,\quad g_k=-\frac1{k(k+1)},\qquad
      [t^0]g^2=1,\quad
      [t^k]g^2=\frac{4H_{k-1}-6}{k(k+1)(k+2)}\quad(k\geq1),

   their contribution, with :math:`n=j+1`, is

   .. math::

      -24\bigl(g*(\phi_1\phi_2)\bigr)_j
       +12\bigl(g^2*\phi_1^2\bigr)_j
       =-\frac{48(1-H)}{n^2(n+1)}+\frac{12}{n^2}
        +\sum_{k=1}^{n-1}\bigl(R_H(k,n)H_{k-1}+R_0(k,n)\bigr),

   where

   .. math::

      p(k,n)&=\frac1{k^2(k+1)(n-k)(n-k+1)},\\
      q(k,n)&=\frac1{k(k+1)(k+2)(n-k)^2},\\
      R_H(k,n)&=48(q(k,n)-p(k,n)),\\
      R_0(k,n)&=48p(k,n)-72q(k,n).

   Partial fractions use only the seven denominators
   :math:`k,k^2,k+1,k+2,n-k,(n-k)^2,n-k+1`. The sums without
   :math:`H_{k-1}` are immediate harmonic numbers. For the remaining
   sums, put :math:`T=(H^2-H_2)/2`; their respective values are

   .. math::

      \sum_{k=1}^{n-1}\frac{H_{k-1}}k&=T,\\
      \sum_{k=1}^{n-1}\frac{H_{k-1}}{k^2}&=A,\\
      \sum_{k=1}^{n-1}\frac{H_{k-1}}{k+1}
        &=T+H/n-1+1/n,\\
      \sum_{k=1}^{n-1}\frac{H_{k-1}}{k+2}
        &=T+H\left(1/n+1/(n+1)\right)+1/(n(n+1))
          -7/4+1/(2n)+3/(2(n+1)),\\
      \sum_{k=1}^{n-1}\frac{H_{k-1}}{n-k}&=H^2-H_2,\\
      \sum_{k=1}^{n-1}\frac{H_{k-1}}{(n-k)^2}&=A+HH_2-H_3,\\
      \sum_{k=1}^{n-1}\frac{H_{k-1}}{n-k+1}&=H^2-H_2+2H/n-H.

   These identities follow by changing the order of the finite sums;
   for example, the penultimate sum is
   :math:`\sum_{i+r\leq n-1}1/(ir^2)`, and partial fractions at
   :math:`i+r=s` give :math:`2A+\sum_{s\leq n-1}H_{s-1}^{(2)}/s`
   :math:`{}=A+HH_2-H_3`. The formulas also hold when :math:`n=1`
   and the sums are empty. Substitution and collection give exactly
   :eq:`risk-five-vertex-diagonal`.

   The binomial-transform diagonalization now gives

   .. math::

      a_0(5,D)=\frac{25(D-1)!(D+1)!}{(5)_{2D}}
       \sum_j W_{D,j}\Psi_j.

   Use :math:`(D-1)!(D+1)!Z_D=(2D)!` to obtain
   :eq:`risk-five-vertex-single`. Positivity follows from the general
   argument below.

Positive diagonals for every vertex count
-----------------------------------------

.. prf:proposition:: Outside-kernel powers remain positive after transformation
   :label: prop-risk-positive-transformed-kernel

   For every integer :math:`q\geq1` and :math:`j\geq0`,

   .. math::

      [x^jy^j]\frac{(g(x)+g(y)-g(xy))^q}{(1-x)(1-y)}>0.

.. prf:proof::

   Let :math:`T_1,\ldots,T_q` be independent exponential random variables
   of mean one, and define the homogeneous polynomial

   .. math::

      Q_h(T)=\sum_{\alpha_1+\cdots+\alpha_q=h}
        \prod_{r=1}^q\frac{T_r^{\alpha_r}}{(\alpha_r+1)!}.

   Since :math:`\mathbb E[T_r^k]=k!`, multiplication of the outside
   kernel :math:`F` gives

   .. math::

      [x^hy^l]F(x,y)^q=\mathbb E[Q_h(T)Q_l(T)].

   The signed binomial transform in each index therefore identifies the
   diagonal in the proposition with

   .. math::

      \mathbb E\!\left[
       \left(\sum_{h=0}^j(-1)^h\binom jh Q_h(T)\right)^2\right].

   The polynomial inside the square is nonzero: its highest homogeneous
   part is :math:`(-1)^jQ_j`, including the case :math:`j=0`.
   Exponential density is strictly positive on the positive orthant, so
   the expectation is strictly positive. This proves positivity without
   requiring each term of a collected harmonic expression to be positive.

The leading asymptotic constant
-------------------------------

.. prf:corollary:: Five-vertex vertex asymptotic
   :label: cor-risk-five-vertex-asymptotic

   As :math:`D\to\infty`,

   .. math::
      :label: risk-five-vertex-asymptotic

      a_0(5,D)=\frac{7200(\zeta(2)-\zeta(3))}{D^6}
        +O\!\left(\frac{\log D}{D^7}\right).

.. prf:proof::

   The Euler sum :math:`A_j` tends to :math:`\zeta(3)`. Indeed,
   monotone convergence and the power series for the logarithm give

   .. math::

      \sum_{k=1}^\infty\frac{H_{k-1}}{k^2}
       =\frac12\int_0^1\frac{\log^2(1-t)}t\,dt
       =\frac12\int_0^1\frac{\log^2t}{1-t}\,dt
       =\zeta(3).

   Moreover, for :math:`j\geq1`,

   .. math::

      0\leq\zeta(2)-H_j^{(2)}\leq1/j,\qquad
      0\leq\zeta(3)-A_j\leq(2+\log j)/j.

   The second inequality follows from
   :math:`H_{k-1}\leq1+\log k` and comparison with the decreasing
   function :math:`(1+\log x)/x^2`. The coefficient of :math:`H_2`
   in :eq:`risk-five-vertex-diagonal` differs from
   :math:`48/(n(n+2))` by

   .. math::

      -\frac{12(2n^3+3n^2-2n-12)}
             {n^2(n+1)(n+2)^2(n+3)}=O(n^{-3}).

   All the other terms apart from :math:`-48A/(n(n+2))` are
   :math:`O(\log n/n^3)`; the cubic harmonic term is
   :math:`O(\log^3n/n^4)=O(\log n/n^3)`.
   Consequently

   .. math::

      \Psi_j=\frac{48(\zeta(2)-\zeta(3))}{(j+1)(j+3)}
        +O\!\left(\frac{\log(j+2)}{(j+1)^3}\right).

   Let :math:`J` have probabilities :math:`W_{D,j}/Z_D`. The mean,
   variance and small-index bound proved in :doc:`skeleton_risk_vertex`
   apply unchanged:

   .. math::

      \mathbb E[J]=\frac{D^2-1}{2D},\qquad
      \operatorname{Var}(J)=O(D),\qquad
      \mathbb P(J\leq\lfloor D/4\rfloor)\leq4\,2^{-D/12}
      \quad(D\geq12).

   On the complementary event, the remainder is uniformly
   :math:`O(\log D/D^3)`. The sequence :math:`\Psi_j` is bounded,
   so the omitted event contributes an exponentially small amount.
   Taylor expansion of :math:`1/((x+1)(x+3))` about the mean, with
   second derivative :math:`O(D^{-4})` and variance :math:`O(D)`, gives

   .. math::

      \mathbb E[\Psi_J]
       =\frac{192(\zeta(2)-\zeta(3))}{D^2}
         +O(\log D/D^3).

   Finally, :eq:`risk-five-vertex-single` is
   :math:`600\mathbb E[\Psi_J]/\prod_{r=1}^4(2D+r)`.
   Expanding this rational factor proves the assertion.

Reproduction and limits
-----------------------

``tools.risk_research.vertex_five.five_vertex_coefficient`` evaluates
:eq:`risk-five-vertex-single` using exact rational arithmetic, four harmonic
states, a binomial weight and an accumulator. The number of rational states
is fixed, while their bit lengths increase with :math:`D`. Floating-point
evaluation of the ungrouped weight can lose precision through cancellation;
the exact evaluator avoids that loss.

The tests compare the new expression with the original scalar kernel sum
at degrees 1 through 20, independent Gram propagation at degrees 1 through
5, and direct signed binomial transforms of :math:`F^4`. After the candidate
was frozen, a separate integer-polynomial :math:`F^4` computation at degrees
193, 227 and 251 agreed exactly. The committed holdout file records the
candidate and oracle source hashes. These checks support the implementation;
the algebraic and asymptotic arguments above establish the mathematical
identities.
