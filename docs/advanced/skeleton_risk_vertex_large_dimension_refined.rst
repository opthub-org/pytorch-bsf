A sharper elementary large-vertex guarantee
============================================

Keep the uniform-simplex assumptions, the fixed-degree-limit constant
:math:`C_M=M^2(M-1)!2^{3-M}c_{M-1}`, and the leading term :math:`B_M`
from :doc:`skeleton_risk_vertex_large_dimension`. This chapter improves
the explicit error and starting vertex count. It does not assert a
second correction coefficient or a joint :math:`M,D` limit.

.. prf:theorem:: Improved first-order large-vertex guarantee
   :label: thm-risk-refined-large-vertex-constant

   For every integer :math:`M\geq162`,

   .. math::
      :label: risk-refined-large-vertex-error

      \left|C_M/B_M-1\right|\leq\frac6{M-2}.

   In particular, one percent is guaranteed at every integer
   :math:`M\geq602`. The previous sufficient start was :math:`2502`;
   the previous simple radius was :math:`25/(M-2)` starting at 1026.
   These are sufficient thresholds, not the smallest possible ones.

.. prf:proof::

   **A signed fourth-derivative bound on a larger interval.** Set
   :math:`n=M-2`, :math:`L=\log2`,
   :math:`\alpha=(L-1/2)/(4L)`, and retain the exact definitions

   .. math::

      h(x)=4\cos^2(x/2)\log(2\cos(x/2))+x\sin x,\quad
      f(x)=h(x)/(4L),\quad A(x)=\sec^2(x/2).

   The positive series for :math:`L` gives
   :math:`29/42<842/1215<L<7/10`, so
   :math:`2/29<\alpha<1/14`. For :math:`0\leq x\leq1`,
   :math:`\cos(x/2)\geq7/8`. Integrating
   :math:`v\leq\tan v\leq8v/7` on :math:`[0,x/2]` gives
   :math:`-x^2/6\leq\log\cos(x/2)\leq-x^2/8`.
   Differentiation gives the exact identity

   .. math::

      h^{(4)}(x)=x\sin x+2\cos x\log(2\cos(x/2))
                      +\frac{\cos x}{1+\cos x}-2.

   For its lower bound, use
   :math:`x\sin x\geq x^2-x^4/6`,
   :math:`\cos x\geq1-x^2/2`,
   :math:`\log(2\cos(x/2))\geq2/3-x^2/6` and
   :math:`\cos x/(1+\cos x)\geq1/3`. The first two terms sum
   to at least :math:`4/3`; hence :math:`h^{(4)}\geq-1/3`.
   For the upper bound, the alternating sine and cosine estimates give

   .. math::

      h^{(4)}(x)
      &\leq x^2-x^4/6+x^6/120
          +2(1-x^2/2+x^4/24)(7/10-x^2/8)\\
      &\qquad+1/2-x^2/8+x^4/96-2\\
      &=-1/10-3x^2/40+13x^4/480-x^6/480<0.

   Here :math:`\cos x/(1+\cos x)
   =1/2-(1-\cos x)/(2(1+\cos x))
   \leq1/2-x^2/8+x^4/96`. Since :math:`h` is even, Taylor's
   theorem and :math:`L>2/3` therefore imply

   .. math::

      1-\alpha x^2-x^4/192\leq f(x)\leq1-\alpha x^2.

   Put :math:`u=1-f(x)`. The bound
   :math:`1/14+1/192<1/13` proves :math:`0\leq u\leq x^2/13`.
   Thus :math:`f>0` and

   .. math::

      \log f(x)\leq-\alpha x^2,\qquad
      |\log f(x)+\alpha x^2|
       \leq\left(\frac1{192}+\frac1{312}\right)x^4
       =\frac7{832}x^4.

   This uses :math:`0\leq-\log(1-u)-u\leq u^2/(2(1-u))`.
   Also :math:`1\leq A\leq64/49<4/3` and
   :math:`A-1=\tan^2(x/2)\leq16x^2/49<x^2/3`.

   **Gaussian moments with the same exponent.** Let
   :math:`G_n=\sqrt\pi/(2\sqrt{\alpha n})`. The signed logarithmic
   estimate lets both exponentials use :math:`e^{-\alpha nx^2}`:

   .. math::

      |f(x)^nA(x)-e^{-\alpha nx^2}|
       \leq\left(\frac43\frac7{832}nx^4+\frac{x^2}3\right)
                    e^{-\alpha nx^2}.

   Extend the positive majorant to infinity and use the second and
   fourth Gaussian moments. After division by :math:`G_n`, the error
   on :math:`[0,1]` is at most

   .. math::

      \frac1n\left(\frac7{832\alpha^2}+\frac1{6\alpha}\right)
       <\frac1n\left(\frac{5887}{3328}+\frac{29}{12}\right)
       <\frac{17}{4n}.

   **The singular endpoint remains controlled.** Monotonicity of
   :math:`f` and the exact second moment
   :math:`I_2=\pi^3/(12L^2)<12` are proved in the earlier chapter.
   They apply to :math:`I_n=\int_0^\pi f(x)^nA(x)\,dx` for all
   :math:`n\geq2`. Since :math:`f(1)\leq1-\alpha<15/16` and
   :math:`G_n>3/\sqrt n`,

   .. math::

      \frac1{G_n}\int_1^\pi f(x)^nA(x)\,dx
       \leq4\sqrt n(15/16)^{n-2},\qquad
      \frac1{G_n}\int_1^\infty e^{-\alpha nx^2}\,dx
       \leq\frac8{3\sqrt n}e^{-n/16}.

   Each right side is at most :math:`1/(2n)` for all real
   :math:`n\geq160`. For the first, logarithmic differentiation shows
   that :math:`n^{3/2}e^{-(n-2)/16}` decreases after 24. At 160,
   :math:`e^{158/16}>2^{14}>8\,160^{3/2}`: use
   :math:`14L<49/5<79/8` and
   :math:`\sqrt{160}<127/10`, giving :math:`8\,160^{3/2}<16256<16384`.
   The second follows because :math:`\sqrt n e^{-n/16}` decreases
   after 8, and :math:`e^{10}>2^{14}>(16/3)\sqrt{160}`.
   Consequently

   .. math::

      |I_n/G_n-1|\leq\frac{21}{4n}\qquad(n\geq160).

   **The normalization is retained exactly.** Equation
   :eq:`risk-large-vertex-normalization` gives
   :math:`C_M/B_M=\sqrt{1+1/n}\,I_n/G_n`. Hence

   .. math::

      |C_M/B_M-1|
       \leq\frac{23}{4n}+\frac{21}{8n^2}\leq\frac6n
           \qquad(n\geq160).

   The final inequality only requires :math:`n\geq21/2`, and the
   stated domain is therefore sufficient. This proves the all-tail
   claim and its one-percent threshold.

Evaluation, replay and limits
------------------------------

The standard-library tool
``tools.risk_research.vertex_large_dimension_refined`` keeps the exact
normalization and elementary logarithm, pi and integer-square-root
enclosures. For :math:`4\leq M<162`, the same proof gives a valid
fallback. With :math:`n=M-2`, put

.. math::

   E_n=\frac{17}{4n}+4\sqrt n(15/16)^{n-2}
                         +\frac8{3\sqrt n}(16/17)^n.

Then :math:`(1+1/(2n))E_n+1/(2n)` bounds the relative error; here
:math:`e^{-1/16}<16/17`. The fallback can exceed one and the lower
constant endpoint is clipped to zero. The simple radius is used only
on its proved domain. The threshold routine returns
:math:`\max(162,2+\lceil6/\varepsilon\rceil)` for positive rational
target :math:`\varepsilon`, valid at every larger integer vertex count.

.. code-block:: console

   $ .venv/bin/python -m tools.risk_research.vertex_large_dimension_refined --vertices 602 --bits 47 --relative-target 1/100 --output .research/skeleton-risk/refined602.json
   $ .venv/bin/python -m tools.risk_research.vertex_large_dimension_refined --verify .research/skeleton-risk/refined602.json

``--verify`` accepts no generation overrides. It restores recorded
precision, target and optional Arb/integral flags, recomputes all fields,
and compares JSON scalar types strictly. Duplicate keys, extra fields,
changed proof/source fingerprints and altered intervals are rejected.
Fingerprints canonicalize CRLF to LF; the calculation guard compares
raw source bytes before replacing output. Failed generation preserves
an existing output, and verification preserves its input.

Optional ``--with-arb --integral-check`` restores the scale and calls
the existing positive-integral implementation without using the refined
error bound. Arb precision is restored and saved balls round outward.
Exact-series and positive-integral finite comparisons check normalization
and implementation independently; the theorem above supplies the
infinite-tail guarantee. Neither these checks nor the improved error
assert a higher-order expansion or a simultaneous growing-degree limit.
