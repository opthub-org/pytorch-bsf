A classical closed form for the codimension-four constant
========================================================

Keep the uniform-simplex assumptions of :doc:`skeleton_risk_theory`.
The general constant and fixed-codimension asymptotic are already proved
in :doc:`skeleton_risk_vertex_general` and
:doc:`skeleton_risk_codimension_three_exact`. This chapter evaluates the
previously unevaluated constant :math:`c_5` by a finite combination of
ordinary Riemann zeta values. It does not change either asymptotic domain.

.. prf:theorem:: The fifth constant
   :label: thm-risk-c5-closed-form

   .. math::

      c_5=900\zeta(4)-600\zeta(3)-120\zeta(2)>0.

   Consequently the six-vertex leading constant is

   .. math::

      C_6=540c_5
         =486000\zeta(4)-324000\zeta(3)-64800\zeta(2),

   and, for every fixed integer :math:`s\geq2`,

   .. math::

      a_{s-1}(s+4,D)\sim A_{s,4}D^{s-6},\qquad
      A_{s,4}=\binom{s+4}{s}^{\!2}
         \frac{s(s+1)(s+2)(s+3)}{80(s-2)!}\,c_5.

   These are leading constants as the degree tends to infinity with
   :math:`s` fixed, not a finite-degree or finite-sample error bound.

.. prf:proof::

   Write :math:`H=H_{n-1}` and :math:`G=H_{n-1}^{(2)}`, including
   :math:`H_0=G_0=0`. The proved elementary-symmetric formula gives

   .. math::

      \phi_1(n-1)&=1/n,\\
      \phi_2(n-1)&=\frac{2(1-H)}{n(n+1)},\\
      \phi_3(n-1)&=\frac{6(H^2-G-3H+1)}{n(n+1)(n+2)}.

   Thus, with :math:`t_n=e_4(n-1)/24`,

   .. math::

      t_n=\frac{2(H^2-G-3H+1)}{n^2(n+1)(n+2)}
              +\frac{(H-1)^2}{n^2(n+1)^2}.

   Define

   .. math::

      U(n,H,G)=\frac{6n^2+9n+2}{2n^2(n+1)}H^2
          -\frac{2n+3}{2n(n+1)}G+\frac{H}{n+1}.

   Direct rational simplification, using
   :math:`H_n=H+1/n` and :math:`H_n^{(2)}=G+1/n^2`, proves the
   finite identity

   .. math::
      :label: risk-c5-finite-telescope

      t_n=U(n+1,H_n,H_n^{(2)})-U(n,H,G)
                 +\frac{3H^2-G-5H-1}{n^2}.

   All series here converge absolutely: use
   :math:`H_{n-1}\leq1+\log n`, :math:`G\leq2` and the displayed
   rational denominators. The first boundary is :math:`U(1,0,0)=0`,
   and the other boundary tends to zero, since
   :math:`U(n,H,G)=O((1+\log n)^2/n)`. Telescoping is therefore valid
   without assigning values to any divergent harmonic series.

   Use the descending-index convention
   :math:`\zeta(k_1,\ldots,k_r)=\sum_{n_1>\cdots>n_r\geq1}
   n_1^{-k_1}\cdots n_r^{-k_r}`. Expanding the finite square gives

   .. math::

      \sum_{n\geq1}\frac{H_{n-1}^2}{n^2}
          &=2\zeta(2,1,1)+\zeta(2,2),\\
      \sum_{n\geq1}\frac{H_{n-1}^{(2)}}{n^2}&=\zeta(2,2),\qquad
      \sum_{n\geq1}\frac{H_{n-1}}{n^2}=\zeta(2,1).

   The needed low-weight identities are
   :math:`\zeta(2,1)=\zeta(3)`,
   :math:`\zeta(2,1,1)=\zeta(4)` and
   :math:`\zeta(2,2)=3\zeta(4)/4`. For completeness, the iterated
   integral of the word :math:`0^{k_1-1}1\cdots0^{k_r-1}1` over
   :math:`1>t_1>\cdots>t_w>0`, with
   :math:`\omega_0=dt/t,\omega_1=dt/(1-t)`, equals the corresponding
   multiple zeta value, by expanding the geometric series and integrating.
   The substitution :math:`t_i\mapsto1-t_{w+1-i}` reverses the word and
   interchanges 0 and 1. It sends 011 to 001 and 0111 to 0001,
   proving the first two identities. The diagonal/off-diagonal split of
   :math:`\zeta(2)^2` gives :math:`2\zeta(2,2)+\zeta(4)`; Euler's
   values :math:`\zeta(2)=\pi^2/6,\zeta(4)=\pi^4/90` prove the third.
   These standard identities are also described in Henrik Bachmann's
   `Multiple Zeta Values, version 24
   <https://www.henrikbachmann.com/uploads/7/7/6/3/77634444/mzv_2025_lecture_notes_v24.pdf>`_,
   Corollary 2.6 and the discussion on printed page 65.

   Since :math:`c_5=5\sum e_4(j)=120\sum t_n`, the telescope gives
   :math:`120(6\zeta(2,1,1)+2\zeta(2,2)-5\zeta(2,1)-\zeta(2))`,
   proving the claimed formula. Positivity follows from the already
   proved positive integral for :math:`c_5`; it is not inferred from
   floating-point cancellation. Substituting into the existing fixed-M
   and fixed-codimension theorems proves both consequences.

Exact rational evaluation of the finite expression
--------------------------------------------------

For integers :math:`k\geq2`, define

.. math::

   d_{m,k}=\sum_{j=0}^m\frac{(-1)^j\binom mj}{(j+1)^k}
     =\frac1{(k-1)!}\int_0^1(1-x)^m(-\log x)^{k-1}\,dx.

They satisfy :math:`0<d_{m,k}\leq1`. Summing the nonnegative geometric
series inside the integral proves

.. math::

   \zeta(k)=\frac{1}{1-2^{1-k}}
          \sum_{m\geq0}2^{-m-1}d_{m,k}.

Indeed the integral of :math:`1/(1+x)` equals
:math:`(1-2^{1-k})\zeta(k)` by the even/odd split of the absolutely
convergent ordinary zeta series. Therefore the sum of the first
:math:`N\geq1` terms is a rational lower bound, with upper error
:math:`2^{-N}/(1-2^{1-k})`. No floating-point or unproved sign is used.

The evaluator uses the equivalent coefficient formula

.. math::

   d_{m,k}=\frac1{m+1}[z^{k-1}]
             \prod_{r=1}^{m+1}(1-z/r)^{-1}.

To prove it, sum the integral in powers of :math:`z` near zero and
evaluate :math:`\int_0^1x^{-z}(1-x)^m dx
=m!/\prod_{r=1}^{m+1}(r-z)`, by integration by parts.
Updating the complete homogeneous coefficients requires only a linear
number of rational operations for fixed k. Rational integer bit lengths
increase with N. This is an evaluator for a finite special-value formula;
its requested numerical precision still requires increasing work.
Negative coefficients in the c5 formula reverse the relevant endpoints.
The positive-integral theorem permits clamping its lower bound to zero.

Use and scope
--------------

.. code-block:: console

   $ .venv/bin/python -m tools.risk_research.c5_closed_form --terms 64 --face-size 2 --output .research/skeleton-risk/c5.json
   $ .venv/bin/python -m tools.risk_research.c5_closed_form --verify .research/skeleton-risk/c5.json

The record contains c5, C6 and A_(s,4) intervals, the exact finite zeta
coefficients, and source/proof hashes. Verification restores saved terms
and face size and compares the entire strictly typed JSON record.
Hashes normalize CRLF to LF; raw-source guards still detect concurrent
edits. A failed generation or verification preserves existing output.
Outputs use a checked canonical parent and the original leaf name in a
directory controlled by the caller. This is not a sandbox against another
process replacing that canonical directory itself.

Tests check the finite telescope using exact arithmetic, compare the
complete-homogeneous recurrence with the alternating binomial definition,
and exercise saved options, record tampering and protected outputs.
Frozen independent checks use the original general-vertex rational
series and the original positive Arb integral, rather than another
evaluation of the new zeta expression. These finite checks validate the
implementation and normalizations; the identities above provide the
proof. No c6 or arbitrary-q classical reduction, degree-error rate,
finite-state recurrence, or simultaneous parameter limit is asserted.
