Uniform errors for growing nonvertex faces
==========================================

Retain the uniform-simplex design, evaluation and noise assumptions of
:doc:`skeleton_risk_theory`. Write s for the number of vertices of the
source face, q for its outside codimension and :math:`M=s+q`.
The quantity below is a coefficient of sample-size-asymptotic risk;
it is not a finite-sample risk bound. The positive leading constant
:math:`A_{s,q}` and explicit rational :math:`Q_q` are those of
:doc:`skeleton_risk_nonvertex_general_error`.

The earlier theorem keeps s,q fixed when describing its asymptotic
scope. Its displayed finite-degree inequality, however, holds at
**every** integer :math:`s,q\geq2` whenever
:math:`D\geq\max(4,2(s-1))`. All dependence on s is explicit.
This chapter derives a uniform moving range from that inequality.
No interchange of a fixed-s limit and a supremum is used.

.. prf:corollary:: A growing-face range at every fixed codimension
   :label: thm-risk-nonvertex-growing-faces

   Put :math:`L(D)=1+\log D` and :math:`H(D)=\lfloor D^{1/4}\rfloor`.
   For every integer :math:`q\geq2` and :math:`D\geq16`,

   .. math::
      :label: risk-nonvertex-growing-faces

      \sup_{2\leq s\leq H(D)}
       \left|\frac{a_{s-1}(s+q,D)}{A_{s,q}D^{s-q-2}}-1\right|
      \leq\begin{cases}
       88L(D)/\sqrt D,&q=2,\\
       \bigl((39/2)Q_q+q(q+1)/4\bigr)/\sqrt D,&q\geq3.
      \end{cases}

   In particular this error tends uniformly to zero over the stated
   face range when q is fixed. The constants need not be sharp.

.. prf:proof::

   The elementary bound :math:`L(D)\leq2D^{1/4}` holds for D>=1:
   set :math:`u=\log D` and maximize :math:`(1+u)e^{-u/4}` on
   u>=0. Its maximum is :math:`4e^{-3/4}<2`, because
   :math:`e^{3/4}>1+3/4+(3/4)^2/2>2`.
   Set :math:`x=D^{1/4}\geq2` and :math:`H=H(D)\leq x`.
   Then :math:`2(s-1)\leq2H\leq2x\leq x^4=D`.
   Thus the full range lies in the earlier theorem's domain.

   At q>=3 its numerator beyond the square-root term satisfies

   .. math::

      2+(s-2)L(D)+(s-1)(s-2)
       \leq2+HL(D)+H^2\leq2+3\sqrt D\leq5\sqrt D.

   Also :math:`H\leq\sqrt D` and :math:`\sqrt D\geq1` give

   .. math::

      \frac{q(2s+q-1)}{4D}
       \leq\frac{q(q+1)}{4\sqrt D}.

   Insert these inequalities into
   :eq:`risk-nonvertex-general-finite-error`. The Q coefficient is
   :math:`29/2+5=39/2`.

   At q2, :math:`Q_2=4`, and that same earlier upper bound is

   .. math::

      \frac{58L}{\sqrt D}+\frac{s+1/2}D
        +\frac{8(L+1)}D
        +\frac{4(s-2)L^2}D
        +\frac{4(s-1)(s-2)L}D.

   Its last four terms are bounded respectively by
   :math:`(3/2)L/\sqrt D`, :math:`16L/\sqrt D`,
   :math:`8L/\sqrt D`, and :math:`4L/\sqrt D`.
   For the third use :math:`HL\leq2H D^{1/4}\leq2\sqrt D`;
   for the last use :math:`H^2\leq\sqrt D`.
   The sum of constants is :math:`58+3/2+16+8+4=175/2\leq88`.
   Each bound is independent of s, proving the supremum claim.

More general envelopes
-----------------------

The original explicit inequality also implies the following qualitative
statements for integer envelopes :math:`h(D)\geq2` eventually:
at any fixed q>=3, :math:`h(D)=o(\sqrt D)` suffices for uniform
convergence over :math:`2\leq s\leq h(D)`; at q2,
:math:`h(D)=o(\sqrt{D/\log D})` suffices.
Indeed the extra terms are bounded by fixed-q constants times
:math:`h/D+hL/D+h^2/D` in the first case, and
:math:`h/D+L/D+hL^2/D+h^2L/D` in the second.
All tend to zero under the respective hypotheses; the earlier theorem's
domain also holds eventually. No effective threshold search for an
arbitrary h is asserted. The evaluator below uses only the concrete
fourth-root envelope. Growth of q together with D, an optimal face
range, and minimal constants or starting degrees are outside this result.

Exact rational all-tail certificates
-------------------------------------

For :math:`k\geq4`, let :math:`\rho_k=2^{k/2}` at even k and
:math:`\rho_k=(4/3)2^{\lfloor k/2\rfloor}` at odd k.
The odd choice is a lower bound for :math:`\sqrt{2^k}` because
:math:`(4/3)^2<2`. The preceding corollary gives

.. math::
   :label: risk-nonvertex-growing-dyadic

   \sup_{D\geq2^k}\ \sup_{2\leq s\leq\lfloor D^{1/4}\rfloor}
    \left|\frac{a_{s-1}(s+q,D)}{A_{s,q}D^{s-q-2}}-1\right|
   \leq E_q(k):=\begin{cases}
      88(k+1)/\rho_k,&q=2,\\
      \bigl((39/2)Q_q+q(q+1)/4\bigr)/\rho_k,&q\geq3.
      \end{cases}

For q2, :math:`L(D)/\sqrt D` decreases on D>=16, since its
derivative has the sign of :math:`1-\log D`.
At :math:`D=2^k`, :math:`L(2^k)\leq k+1` uses :math:`\log2<1`.
The q>=3 envelope decreases directly. In particular **the face limit
keeps growing at every future degree**; it is not frozen at
:math:`\lfloor (2^k)^{1/4}\rfloor`.
Both rational envelopes decrease for k>=4 and tend to zero.
For q2 the even-to-odd ratio is :math:`3(k+2)/(4(k+1))<1`, and
the odd-to-even ratio is :math:`2(k+2)/(3(k+1))<1`.
Consequently exact search terminates mathematically for every fixed q
and every positive rational error target. It returns a sufficient
dyadic threshold, not a minimal degree threshold.

The standard-library evaluator computes H with nested integer square
roots, without a face-sized or degree-sized table. For a single degree
it uses E at the containing dyadic block; this may be looser than the
analytic bound. Its source-bound schema is
``skeleton-risk-nonvertex-growing-faces-v1``. Each record stores its
q,D, exact face limit, rational bound, optional target and tail,
work limit, and normalized SHA256 hashes of the evaluator, proof and
all dependencies of the earlier finite-degree theorem.
The builder and verifier compare raw source bytes at both ends;
the CLI also checks the saved input bytes during replay.

.. code-block:: bash

   python -S -m tools.risk_research.nonvertex_growing_faces \
     --outside 5 --degree 18446744073709551635 \
     --relative-target 1/37 --work-limit 190000 --output /tmp/growing.json
   python -S -m tools.risk_research.nonvertex_growing_faces \
     --verify /tmp/growing.json

The selected-input counter is
:math:`(q+1)^3+q+2+(2q+2)(b_D+b_\epsilon)`; it is checked
before constructing Q or large dyadic powers. Here b_D is the supplied
degree's bit length (or the current trial exponent plus one), and
b_epsilon is the maximum numerator/denominator bit length, or zero
without a target. A record with a target stores the maximum counter
over its requested degree and successful threshold search.
An explicit limit up to 1,000,000 may be selected. Resource refusal
raises an error and preserves an existing output file; it does not
contradict mathematical stopping. The counter is not a bound on CPU,
RSS, cumulative arithmetic work, or rational bit complexity.
These records bound normalized degree error only. Leading-constant
numerical error remains separate, and production APIs are unchanged.
