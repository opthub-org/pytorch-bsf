Quantitative five-vertex vertex asymptotics
===========================================

Under the uniform-simplex assumptions of :doc:`skeleton_risk_theory`,
:doc:`skeleton_risk_five_vertex` gives a positive single sum and the leading
constant of the vertex coefficient on five vertices. The following bound
replaces its unspecified asymptotic remainder by an explicit one. Put
:math:`c=\zeta(2)-\zeta(3)`, so :math:`0<c<1`.

.. prf:theorem:: Five-vertex leading term with an explicit remainder
   :label: thm-risk-five-vertex-explicit-error

   For every integer :math:`D\geq256`,

   .. math::
      :label: risk-five-vertex-leading-error

      \left|a_0(5,D)-\frac{7200c}{D^6}\right|
         \leq\frac{3000000(1+\log D)}{D^7}.

The constant is conservative: at the start of the domain the error bound
can exceed the coefficient. The bound is :math:`O(\log D/D^7)` and
therefore gives a relative enclosure that shrinks to zero. It is a bound
on the mathematical coefficient, in addition to any numerical rounding.

.. prf:proof::

   **A pointwise bound.** Use :math:`\Psi_j` from
   :eq:`risk-five-vertex-diagonal`, and put :math:`n=j+1` and
   :math:`B_n=1+\log n`. Integral comparison gives

   .. math::

      H_{n-1}\leq B_n,\qquad H_{n-1}^2\leq2n,\qquad
      H_{n-1}^{(2)}<2,\qquad H_{n-1}^{(3)}<5/4.

   For the squared bound, the maximum of
   :math:`(1+\log t)^2/t` over :math:`t\geq1` is :math:`4/e<2`.
   The last bound follows from
   :math:`\zeta(3)<1+1/8+\int_2^\infty t^{-3}\,dt=5/4`.

   Separate the two terms containing the eventual constant:

   .. math::

      \Psi_j=\frac{48(H_j^{(2)}-A_j)}{n(n+2)}+R_j.

   The remaining cubic, quadratic, mixed, linear, second-harmonic,
   third-harmonic and rational terms have the following absolute bounds,
   in their order in :eq:`risk-five-vertex-diagonal`:

   .. math::

      |R_j|&\leq\frac{48H_j+120+96H_j+48H_j+48+60+1}{n^3}\\
            &=\frac{192H_j+229}{n^3}.

   Here the second-harmonic contribution is the difference between its
   original coefficient and :math:`48/(n(n+2))`. The rational inequalities
   behind this bound hold throughout :math:`n\geq1`. To check them
   without an asymptotic expansion, write

   .. math::

      Q&=(n+1)^2(n+2)^2(n+3),\\
      P_2&=5n^3+7n^2-6n-12,\\
      P_4&=2n^4+15n^3+44n^2+47n+24,\\
      P_5&=2n^3+3n^2-2n-12.

   Then :math:`5Q\pm n^2P_2` have nonnegative coefficients;
   :math:`2Q-nP_4=3n^4+18n^3+55n^2+56n+24`;
   and
   :math:`2(n+1)(n+2)(n+3)-n^2(2n+3)=9n^2+22n+12`.
   Similarly :math:`2(n+1)(n+2)^2(n+3)\pm nP_5` have nonnegative
   coefficients. Finally :math:`Q-144n`, after substituting
   :math:`n=1+t`, has nonnegative coefficients. These respectively bound
   the quadratic, linear, mixed, second-harmonic correction and rational
   factors. All cleared denominators are positive.

   For :math:`j\geq1`, the Euler-sum identity from
   :prf:ref:`cor-risk-five-vertex-asymptotic` gives

   .. math::

      0\leq H_j^{(2)}-A_j-c
         =\sum_{k=j+1}^{\infty}\frac{H_{k-1}-1}{k^2}
         \leq\zeta(3)-A_j
         \leq\frac{2+\log j}{j}
         \leq\frac{4B_n}{n}.

   Consequently the additional error in replacing
   :math:`H_j^{(2)}-A_j` by :math:`c` is at most
   :math:`192B_n/n^3`. Since :math:`H_j\leq B_n` and :math:`B_n\geq1`,
   :math:`192+229+192=613<625` proves

   .. math::
      :label: risk-five-vertex-pointwise-bound

      \left|\Psi_j-\frac{48c}{(j+1)(j+3)}\right|
         \leq\frac{625(1+\log(j+1))}{(j+1)^3}.

   At :math:`j=0`, :math:`\Psi_0=1` and :math:`0<c<1`, so the same
   bound follows from :math:`|1-16c|\leq15<625`.
   The function :math:`(1+\log n)/n^3` decreases for :math:`n\geq1`.
   Positivity of the transformed kernel therefore also gives the uniform
   bound :math:`0<\Psi_j\leq16+625=641`.

   **Averaging the rational leading term.** Use the distribution

   .. math::

      \mathbb P(J=j)=
       \frac{\binom{D-1}j\binom{D+1}j}{\binom{2D}{D-1}},
      \qquad 0\leq j<D.

   Put :math:`x=D/2`, :math:`X=J-x`, :math:`B=1+\log D` and
   :math:`C=\{J>\lfloor D/4\rfloor\}`. The moments and the small-index
   bound in :doc:`skeleton_risk_vertex_asymptotics` give, for
   :math:`D\geq12`,

   .. math::

      \mathbb E[X]&=-1/(2D),\\
      \mathbb E[X^2]&=
        \frac{D^3-2D+2}{4D(2D-1)}\leq D/4,\\
      \mathbb P(C^c)&\leq\epsilon_D=4\,2^{-D/12}.

   Write :math:`r(t)=1/((t+1)(t+3))`. For :math:`t>0`,

   .. math::

      r(t)=\frac12\int_1^3(t+u)^{-2}\,du,

   hence :math:`|r'(t)|\leq2/t^3` and :math:`|r''(t)|\leq6/t^4`.
   Also :math:`|r(t)-t^{-2}|\leq4/t^3`. On the segment from
   :math:`x` to any :math:`j\in C`, :math:`t>D/4`. First-order Taylor
   expansion thus has expected remainder at most
   :math:`192/D^3`. Its linear expectation has absolute value at most
   :math:`8/D^4\leq1/D^3`, while
   :math:`|r(x)-4/D^2|\leq32/D^3`.
   On the omitted range the absolute Taylor polynomial is bounded by
   :math:`(4+8)/D^2`, since :math:`|X|\leq D/2`. It follows that

   .. math::

      \left|\mathbb E[\mathbf1_Cr(J)]-4/D^2\right|
         \leq225/D^3+12\epsilon_D/D^2.

   On :math:`C`, :math:`j+1>D/4` and :math:`B_{j+1}\leq B`.
   Combining :eq:`risk-five-vertex-pointwise-bound`, the rational
   expectation bound and :math:`\Psi_j\leq641` on the omitted range yields

   .. math::

      \left|\mathbb E[\Psi_J]-\frac{192c}{D^2}\right|
        \leq\frac{40000B+10800}{D^3}
          +\frac{576\epsilon_D}{D^2}+641\epsilon_D.

   For :math:`D\geq256`, both :math:`D^3\epsilon_D` and
   :math:`D\epsilon_D` decrease: their logarithmic derivatives are
   :math:`3/D-\log2/12` and :math:`1/D-\log2/12`, respectively, and
   :math:`\log2>1/2`. At 256, :math:`\epsilon_D<2^{-19}`.
   Therefore :math:`D^3\epsilon_D<32` and
   :math:`576D\epsilon_D<576/2048<1`. Since :math:`B\geq1`,

   .. math::

      \left|\mathbb E[\Psi_J]-\frac{192c}{D^2}\right|
          \leq\frac{71313B}{D^3}.

   **Outer normalization.** The exact single sum is

   .. math::

      a_0(5,D)=\frac{75}{2D^4}q_D\mathbb E[\Psi_J],\qquad
      q_D=\prod_{r=1}^4(1+r/(2D))^{-1}.

   Here :math:`0<q_D\leq1` and
   :math:`1-q_D\leq\sum_{r=1}^4r/(2D)=5/D`. Thus the final error is
   at most

   .. math::

      \left(\frac{75\cdot71313}{2}+36000\right)\frac{B}{D^7}
        =\frac{5420475}{2}\frac{B}{D^7}
        <3000000\frac{B}{D^7},

   which proves :eq:`risk-five-vertex-leading-error`.

Reproducible enclosures
-----------------------

``tools/risk_research/vertex_five_asymptotic.py`` returns the exact rational
majorant :math:`3000000(1+\lceil\log_2D\rceil)/D^7`. The integer bit count
dominates the logarithm without a floating-point conversion. For this bound
alone, no optional numerical dependency is needed:

.. code-block:: console

   python -m tools.risk_research.vertex_five_asymptotic --degree 100000 --bound-only --output .research/five-vertex-error.json

Without ``--bound-only``, use the prepared Sage environment, which includes
python-flint. Arb encloses :math:`\zeta(2)` and :math:`\zeta(3)`, then adds
both arithmetic rounding and the proved analytic error. The local precision
context is restored. Reports record the evaluator hash and numerical version.
Native tests also reparse the emitted JSON balls and compare them with the
independent exact single sum. Finite checks validate the implementation;
the all-degree guarantee is the inequality proof above.
