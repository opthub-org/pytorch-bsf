Quantitative tetrahedron vertex asymptotics
============================================

Under the uniform-simplex assumptions of :doc:`skeleton_risk_theory`, the
positive single sum in :doc:`skeleton_risk_vertex` gives a second asymptotic
term and a completely explicit error bound. Write
:math:`\gamma=\lim_n(H_n-\log n)` for Euler's constant and put
:math:`L_D=\log(D/2)+\gamma`.

.. prf:theorem:: Two-term vertex expansion with an explicit remainder
   :label: thm-risk-vertex-two-term-error

   Define

   .. math::
      :label: risk-vertex-two-term-expansion

      A_D=\frac{48\pi^2}{D^5}
       +\frac{576(L_D^2-L_D-1)-456\pi^2}{D^6}.

   For every integer :math:`D\geq256`,

   .. math::
      :label: risk-vertex-two-term-error

      \left|a_0(4,D)-A_D\right|
        \leq\frac{10^7(1+\log D)^2}{D^7}.

   In particular the next-order coefficient contains a squared logarithm;
   the error is :math:`O(\log^2D/D^7)`. This result concerns the
   coefficient itself, rather than the error of a floating-point evaluation.

The constant is conservative. The inequality holds at the beginning of its
domain even when its error bound is larger than the coefficient. It becomes
an increasingly accurate relative enclosure as the degree grows.

.. prf:proof::

   Put :math:`z=\zeta(2)=\pi^2/6`, :math:`B=1+\log D`,
   :math:`x=D/2`. Use the distribution from the single-sum proof,

   .. math::

      \mathbb P(J=j)
        =\frac{\binom{D-1}j\binom{D+1}j}{\binom{2D}{D-1}},
      \qquad 0\leq j<D.

   With :math:`G_j` as in :eq:`risk-tetrahedron-vertex-weight`,

   .. math::

      a_0(4,D)=\frac{72}{D^3}q_D\,\mathbb E[G_J],
      \qquad
      q_D=\frac1{(1+1/(2D))(1+1/D)(1+3/(2D))}.

   **Pointwise expansion.** Integral comparison gives

   .. math::

      0<H_j-\log j-\gamma<1/j,\qquad
      H_j^{(2)}=z-1/j+e_j,\quad 0\leq e_j\leq1/j^2.

   The first follows by summing
   :math:`0<\log(1+1/k)-1/(k+1)<1/k-1/(k+1)` from :math:`k=j`
   to infinity. The second follows by bounding the tail of
   :math:`\sum k^{-2}` between :math:`1/(j+1)` and :math:`1/j`.
   Also :math:`0<\gamma<1` and :math:`z<2`.

   Write :math:`\ell_t=\log t+\gamma` and define

   .. math::

      u(t)=t^{-2},\qquad
      f(t)=\frac{\ell_t^2-\ell_t-4z-1}{t^3}.

   For integers :math:`1\leq j<D`, direct rational expansions of the
   four terms of :math:`G_j` give

   .. math::
      :label: risk-vertex-pointwise-error

      G_j=zu(j)+f(j)+R_j,
      \qquad |R_j|\leq160B^2/j^4.

   Here are explicit bounds for that calculation. The errors in replacing
   the first denominator by :math:`j^{-3}` and the third rational factor
   by :math:`j^{-3}` are at most :math:`23/j^4` and :math:`71/j^4`.
   For the second term,

   .. math::

      \left|\frac1{(j+1)(j+3)}-\frac1{j^2}+\frac4{j^3}\right|
        =\frac{13j+12}{j^3(j+1)(j+3)}\leq25/j^4.

   Thus the error before replacing :math:`H_j` by :math:`\ell_j` is
   at most :math:`(23H_j^2+71H_j+25z+10)/j^4`.
   The replacement adds at most :math:`(2B+1)/j^4`.
   Since :math:`H_j,\ell_j\leq B`, their total is at most
   :math:`157B^2/j^4`, which implies the stated bound.

   **Moments.** Put :math:`X=J-x`, :math:`M_r=\mathbb E[X^r]`.
   Vandermonde's identity gives

   .. math::

      \mathbb E[(J)_{\underline r}]
       =\frac{(D-1)_{\underline r}(D+1)_{\underline r}}
                    {(2D)_{\underline r}},\quad 1\leq r\leq4,

   where the underline denotes a falling factorial. Converting these
   four moments to powers and centering at :math:`x=D/2` gives

   .. math::

      M_1&=-\frac1{2D},\\
      M_2&=\frac{D^3-2D+2}{4D(2D-1)},\\
      M_3&=-\frac{3D^2+2D-4}{8D(2D-1)},\\
      M_4&=\frac{3D^5-4D^4-12D^3+20D^2+16D-24}
                    {16D(2D-3)(2D-1)}.

   For :math:`D\geq12`, these imply

   .. math::

      M_2\leq D/4,\qquad |M_2-D/8|\leq1/4,
      \qquad |M_3|\leq1,\qquad M_4\leq D^2.

   These inequalities are rational: after clearing their positive
   denominators, substituting :math:`D=12+t` leaves nonnegative polynomial
   coefficients. They therefore hold on the whole stated domain.

   **Central range and tail.** Let :math:`C=\{J>\lfloor D/4\rfloor\}`
   and :math:`\epsilon_D=4\,2^{-D/12}`. The weight-ratio argument in
   :prf:ref:`cor-risk-tetrahedron-vertex-asymptotic` proves
   :math:`\mathbb P(C^c)\leq\epsilon_D` for :math:`D\geq12`.
   Along the segment from :math:`x` to any :math:`j\in C`,
   :math:`t>D/4` and :math:`0<\ell_t\leq B`.
   Restricted expectations below sum only over :math:`C`; neither
   :math:`u(0)` nor :math:`f(0)` is evaluated.

   Taylor expansion of :math:`u` through degree three uses
   :math:`u^{(4)}(t)=120/t^6`. Its expectation on the central range,
   with the moment bounds above, satisfies

   .. math::

      \left|\mathbb E[\mathbf1_Cu(J)]
           -\frac4{D^2}-\frac6{D^3}\right|
        \leq\frac{20511}{D^4}+\frac{40\epsilon_D}{D^2}.

   Indeed the linear, quadratic-correction, cubic and fourth-derivative
   errors are bounded by :math:`8,12,11,20480` times :math:`D^{-4}`.
   The absolute value of the cubic Taylor polynomial on the omitted tail
   is at most :math:`(4+8+12+16)/D^2=40/D^2`.

   For :math:`f`, differentiation gives

   .. math::

      f'(t)&=\frac{-3\ell_t^2+5\ell_t+12z+2}{t^4},\\
      f''(t)&=\frac{12\ell_t^2-26\ell_t-48z-3}{t^5}.

   The numerator bounds are :math:`34B^2` and :math:`137B^2`.
   First-order Taylor expansion with a second-derivative remainder yields

   .. math::

      \left|\mathbb E[\mathbf1_Cf(J)]
        -\frac{8(L_D^2-L_D-4z-1)}{D^3}\right|
        \leq\frac{17559B^2}{D^4}
          +\frac{360B^2\epsilon_D}{D^3}.

   Its linear expectation is at most :math:`272B^2/D^5\leq23B^2/D^4`;
   the quadratic remainder is at most :math:`17536B^2/D^4`.
   On the tail the absolute linear Taylor polynomial is at most
   :math:`(88+272)B^2/D^3`.

   The positive single-sum weight also satisfies :math:`0<G_j\leq1`
   for every :math:`j\geq0`, as shown in the tetrahedron comparison.
   Combine this tail bound with :eq:`risk-vertex-pointwise-error` and
   :math:`j^{-4}<256/D^4` on :math:`C`. Since
   :math:`40960+2\cdot20511+17559=99541<100000`,

   .. math::

      \mathbb E[G_J]={}&\frac{4z}{D^2}
       +\frac{8(L_D^2-L_D-1)-26z}{D^3}+\mathcal R_D,\\
      |\mathcal R_D|\leq{}&\frac{100000B^2}{D^4}
       +\epsilon_D\left(1+\frac{80}{D^2}+\frac{360B^2}{D^3}\right).

   **Outer normalization.** Direct multiplication gives
   :math:`0<q_D\leq1` and
   :math:`|q_D-1+3/D|\leq7/D^2` for :math:`D\geq12`.
   Thus the coefficient of :math:`D^{-6}` is

   .. math::

      72\left(8(L_D^2-L_D-1)-26z-12z\right)
       =576(L_D^2-L_D-1)-456\pi^2.

   Put :math:`T_D=8(L_D^2-L_D-1)-26z`. The remaining error before
   multiplication by :math:`72/D^3` is bounded by

   .. math::

      \frac{7\cdot4z+(3+7/D)|T_D|}{D^4}
         \leq\frac{360B^2}{D^4},

   using :math:`4z<8`, :math:`|T_D|\leq76B^2` and
   :math:`3+7/D<4`. Thus the additional normalization error is at most
   :math:`25920B^2/D^7`.
   The remaining tail, divided by :math:`B^2/D^7`, is at most

   .. math::

      72\epsilon_D(D^4+80D^2+360D)<10^6,\qquad D\geq256.

   Each summand decreases there, because :math:`\log2>1/2`.
   At :math:`D=256`, replace :math:`\epsilon_D` by :math:`2^{-19}`;
   the resulting rational expression is less than :math:`10^6`.
   Finally :math:`7200000+25920+1000000<10^7`, proving
   :eq:`risk-vertex-two-term-error`.

Reproducible enclosures
-------------------------

``tools/risk_research/vertex_asymptotic.py`` returns the rational upper
bound :math:`10^7(1+\lceil\log_2D\rceil)^2/D^7`, which dominates
:eq:`risk-vertex-two-term-error` without evaluating a rounded logarithm.
The optional real-ball evaluator uses python-flint/Arb for :math:`\pi`,
:math:`\gamma` and the logarithm, and includes both arithmetic and analytic
error in its returned coefficient enclosure. Its precision context is restored.

For an error bound using only the project Python and standard library:

.. code-block:: console

   python -m tools.risk_research.vertex_asymptotic --degree 10000 --bound-only --output .research/vertex-error.json

For a real-ball enclosure, run the same command without ``--bound-only``
in the prepared Sage environment, which already contains python-flint.
Finite exact comparisons and moment checks detect implementation errors;
the all-degree guarantee follows from the inequalities above.
