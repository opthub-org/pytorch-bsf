The vertex leading constant at large vertex count
=================================================

Keep the uniform-simplex assumptions and the positive leading constant
:math:`C_M` of :doc:`skeleton_risk_vertex_general`. This chapter concerns
that constant as :math:`M` grows. The degree limit used to define
:math:`C_M` remains a separate fixed-:math:`M` theorem; no simultaneous
:math:`M,D` limit follows by substitution.

A quantified large-M theorem
----------------------------

Write :math:`q=M-1`, :math:`n=M-2`, :math:`L=\log2` and
:math:`a=L-1/2>0`. Define

.. math::

   B_M=\frac{M^2(M-1)!\,(2\log2)^{M-1}\sqrt{M-1}}
                  {4\sqrt{\pi\log2\,(\log2-1/2)}}.

.. prf:theorem:: Explicit large-vertex asymptotic constant
   :label: thm-risk-large-vertex-constant

   For every integer :math:`M\geq1026`,

   .. math::
      :label: risk-large-vertex-error

      \left|\frac{C_M}{B_M}-1\right|\leq\frac{25}{M-2}.

   Consequently :math:`C_M\sim B_M` as :math:`M\to\infty`.
   In particular, the relative error is at most one percent for every
   integer :math:`M\geq2502`.
   These constants and the starting vertex count are conservative.

.. prf:proof::

   **The unique maximum and the exact normalization.** Put
   :math:`h(\theta)=2\operatorname{Re}g(e^{i\theta})` with the existing
   :math:`g(z)=-(1-z)\log(1-z)/z`. For :math:`0\leq x<\pi`, its
   explicit real form is

   .. math::

      h(\pi-x)=4\cos^2(x/2)\log(2\cos(x/2))+x\sin x.

   It extends continuously to zero at :math:`x=\pi`, is positive before
   that point, and strictly decreases. To check monotonicity, set
   :math:`c=\cos\theta` in the already established positive integral:

   .. math::

      \operatorname{Re}g(e^{i\theta})
       =\int_0^1\frac{(1+u)(1-c)}{1-2uc+u^2}\,du.

   The derivative of its integrand with respect to :math:`c` is
   :math:`-(1+u)(1-u)^2/(1-2uc+u^2)^2<0` for :math:`u<1`.
   This proves the unique maximum :math:`H=h(\pi)=4L`.
   Set :math:`f(x)=h(\pi-x)/H`, :math:`A(x)=\sec^2(x/2)` and

   .. math::

      I_n=\int_0^\pi f(x)^n A(x)\,dx,\qquad
      \alpha=\frac{a}{4L}.

   Symmetry and :math:`|1-e^{i(\pi-x)}|^2=4\cos^2(x/2)` give exactly

   .. math::
      :label: risk-large-vertex-normalization

      \frac{C_M}{M^2q!\,(2L)^q}=\frac{q}{4\pi L}I_n.

   **Local estimates with explicit constants.** The elementary bounds
   :math:`2/3<L<7/10` imply :math:`1/16<\alpha<1/12`.
   They follow, for example, by bounding the positive series
   :math:`L=2\sum_{j\geq0}3^{-2j-1}/(2j+1)`.
   For :math:`0\leq x\leq1/2`, direct differentiation gives

   .. math::

      \frac{d^4}{dx^4}h(\pi-x)
       =x\sin x+2\cos x\log(2\cos(x/2))
            +\frac{\cos x}{1+\cos x}-2.

   Its absolute value is at most one on this interval. Indeed,
   :math:`\cos x\geq7/8`, :math:`\cos(x/2)\geq31/32`, and
   :math:`\log(2\cos(x/2))\geq1/2`; the last inequality follows from
   :math:`e<3<(31/16)^2`. A lower bound is
   :math:`7/8+7/15-2>-1`. An upper bound is
   :math:`1/4+2(7/10)+1/2-2<1`.
   Taylor's theorem, including the vanishing odd derivatives, yields

   .. math::

      |f(x)-1+\alpha x^2|\leq x^4/64.

   Monotonicity and this bound imply
   :math:`0\leq1-f(x)\leq(33/256)x^2\leq33/1024`.
   The inequality
   :math:`|\log(1-u)+u|\leq u^2/(2(1-u))` therefore gives

   .. math::

      |\log f(x)+\alpha x^2|\leq x^4/32,\qquad
      \log f(x)\leq-x^2/20.

   For the second assertion one may use
   :math:`f(x)\leq1-(\alpha-1/256)x^2`
   and :math:`\alpha-1/256>15/256>1/20`.
   Also :math:`1\leq A(x)\leq9/8` and
   :math:`A(x)-1=\tan^2(x/2)\leq x^2/2`.

   With :math:`G_n=\int_0^\infty e^{-\alpha nx^2}\,dx
   =\sqrt\pi/(2\sqrt{\alpha n})`, the mean-value inequality for
   exponentials gives

   .. math::

      \left|\int_0^{1/2}f(x)^nA(x)\,dx
             -\int_0^{1/2}e^{-\alpha nx^2}\,dx\right|
      &\leq\int_0^\infty
         \left(\frac{9n}{256}x^4+\frac12x^2\right)e^{-nx^2/20}\,dx\\
      &\leq \frac{21}{n}G_n.

   The Gaussian moments evaluate the ratio in the last line to at most

   .. math::

      \frac{2\sqrt\alpha}{n}
       \left(\frac{27\,20^{5/2}}{2048}
                      +\frac{20^{3/2}}8\right)<\frac{21}{n}.

   For an entirely rational check, use :math:`2\sqrt\alpha<3/5`,
   :math:`\sqrt{20}<9/2`; the resulting numerator is
   :math:`5373/256<21`.

   **The singular part is controlled, not discarded.** Although
   :math:`A(x)` is singular at :math:`x=\pi`, the exact second moment
   is finite. The known :math:`c_3=\pi^2` gives

   .. math::

      I_2=\frac{\pi^3}{12L^2}<12.

   Thus :math:`I_n` exists for all :math:`n\geq2`.
   Monotonicity and the local estimate give
   :math:`f(x)\leq f(1/2)\leq79/80` for :math:`x\geq1/2`, so

   .. math::

      \frac1{G_n}\int_{1/2}^\pi f(x)^nA(x)\,dx
        \leq4\sqrt n\,(79/80)^{n-2}.

   Here :math:`G_n>3/\sqrt n` follows from :math:`\alpha<1/12`
   and :math:`\pi>3`. Separately, the elementary Gaussian-tail bound
   gives

   .. math::

      \frac1{G_n}\int_{1/2}^\infty e^{-\alpha nx^2}\,dx
        \leq\frac{16}{3\sqrt n}e^{-n/64}.

   For every real :math:`n\geq1024`, each of these two displayed
   right sides is at most :math:`1/n`. For the first, dominate
   :math:`(79/80)^{n-2}` by :math:`e^{-(n-2)/80}` and differentiate
   the logarithm of :math:`n^{3/2}e^{-(n-2)/80}`; it decreases after
   :math:`n=120`. At :math:`n=1024`,
   :math:`e^{1022/80}>2^{18}>4\,1024^{3/2}`, using
   :math:`18\log2<18(7/10)<1022/80`.
   The second follows in the same way: :math:`n^{1/2}e^{-n/64}`
   decreases after :math:`n=32`, and at 1024 its required bound follows
   from :math:`e^{16}>2^{16}>512/3`.
   Combining the three errors proves

   .. math::

      \left|I_n/G_n-1\right|\leq24/n\qquad(n\geq1024).

   The exact normalization makes this a relative bound with leading term
   :math:`\widetilde B_M=B_M\sqrt{q/n}`. Finally,
   :math:`1\leq\sqrt{q/n}=\sqrt{1+1/n}\leq1+1/(2n)`, and hence

   .. math::

      |C_M/B_M-1|\leq(24/n)(1+1/(2n))+1/(2n)\leq25/n.

   This proves the theorem, including the prefactor and remainder.

Certified evaluation and independent integration
------------------------------------------------

The standard-library evaluator
``tools.risk_research.vertex_large_dimension`` saves exact rational
intervals for the normalized constant in
:eq:`risk-large-vertex-normalization`. It encloses :math:`\log2` by its
positive series, :math:`\pi` by Machin's alternating arctangent formula,
and square roots by integer square roots. Removing the factorial and
exponential scale keeps reports short even at large :math:`M`.

The simple relative radius is :math:`25/(M-2)` for :math:`M\geq1026`.
For :math:`4\leq M<1026`, the same proof gives a weaker valid radius.
Put

.. math::

   E_n=\frac{21}{n}+4\sqrt n(79/80)^{n-2}
               +\frac{16}{3\sqrt n}(64/65)^n.

Then :math:`(1+1/(2n))E_n+1/(2n)` bounds
:math:`|C_M/B_M-1|`; use :math:`e^{-1/64}<64/65`.
The tool rounds :math:`\sqrt n` outward and clips a negative lower
constant endpoint to zero. The small-:math:`M` fallback can be wide;
the existing exact-series enclosure remains useful there.

Optional python-flint restores the factorial/exponential scale with Arb.
A separate integration path directly evaluates the analytic interval of
:math:`I_n`; it does not use the large-:math:`M` error bound. At a cutoff
:math:`0<b<\pi`, its omitted endpoint interval satisfies

.. math::

   0\leq\int_b^\pi f(x)^nA(x)\,dx\leq12f(b)^{n-2}.

The callback forwards the analytic flag to the logarithm, and both the
cutoff kernel and finiteness are checked. This integration path covers
the singular neighborhood by a proved tail, including :math:`n=2`.
Exact rational-series constant intervals provide another independent
comparison at moderate :math:`M`.

For example, from a source checkout:

.. code-block:: console

   $ .venv/bin/python -m tools.risk_research.vertex_large_dimension --vertices 2502 --relative-target 1/100 --output .research/skeleton-risk/constant2502.json
   $ .venv/bin/python -m tools.risk_research.vertex_large_dimension --vertices 1026 --with-arb --integral-check --output .research/skeleton-risk/constant1026.json

The second command requires optional python-flint. Precision settings are
restored after ball calculations. Saved ball strings are rounded outward;
source changes or computation failures preserve an existing output file.
No public sampling API or mandatory package dependency is added.

The theorem concerns :math:`C_M` alone. A simultaneous growing-:math:`M,D`
vertex asymptotic requires separate control of the :math:`M` dependence
in the finite-degree remainder from
:doc:`skeleton_risk_vertex_general_asymptotics`.
