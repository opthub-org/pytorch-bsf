A classical leading constant for eight vertices
================================================

Keep the uniform-simplex design, evaluation and noise assumptions of
:doc:`skeleton_risk_theory`. The finite construction of
:doc:`skeleton_risk_leading_constant_mzv` already handles every fixed
constant index. This chapter reduces its seventh-index expression using
only convergent identities of weight at most six.

.. prf:theorem:: A classical expression for c7
   :label: thm-risk-c7-closed-form

   .. math::

      c_7={}&211680\zeta(6)-189000\zeta(5)-75250\zeta(4)
                -18900\zeta(3)-3920\zeta(2)\\
             &-78960\zeta(2)\zeta(3)+85680\zeta(2)\zeta(4)
                +50400\zeta(3)^2+5040\zeta(2)^3>0.

   Consequently, for eight vertices fixed,

   .. math::

      a_0(8,D)\sim\frac{C_8}{D^9},\qquad C_8=10080c_7.

   This is a leading-constant identity, not a finite-degree remainder,
   an all-degree comparison or a minimal classical-zeta basis.

The finite-MZV coefficient calculation
-------------------------------------

Use strict descending indices
:math:`\zeta(k_1,\ldots,k_r)=\sum_{n_1>\cdots>n_r\geq1}
n_1^{-k_1}\cdots n_r^{-k_r}`. The proved general construction gives

.. math::
   :label: risk-c7-finite-mzv

   c_7={}&-3920\zeta(2)-18900\zeta(2,1)-62020\zeta(2,1,1)\\
         &-267960\zeta(2,1,1,1)+352800\zeta(2,1,1,1,1)\\
         &+100800\bigl(\zeta(2,1,1,2)+\zeta(2,1,2,1)
                            +\zeta(2,2,1,1)\bigr)\\
         &-78960\bigl(\zeta(2,1,2)+\zeta(2,2,1)\bigr)
              -17640\zeta(2,2)+30240\zeta(2,2,2).

For completeness, this is an exact rational specialization, not a
coefficient fit. At :math:`n\geq1`, take

.. math::

   P_a=\sum_{l=0}^{a-1}(-1)^l a!(a-1)!
       E_{a-1-l}(1,1/2,\ldots,1/(a-1))H_{n-1}(1^l).

The initial harmonic kernels are the finite product

.. math::

   7e_6(n-1)=7\sum_{a=1}^{5}\binom6a
                \frac{P_aP_{6-a}}{(n)_a(n)_{6-a}}.

Expand each product of finite all-one words with the proved finite
stuffle rule. For each rational kernel, the general partial-fraction
identity :math:`f(n)=G(n+1)-G(n)+A/n^2` and finite summation by parts
reduce the remaining word depth. The terminal empty-word contribution
is :math:`A\zeta(2)-G(1)`; a nonempty word :math:`(s,v)` contributes
:math:`A\zeta(2,s,v)` and the shorter kernel
:math:`-G(n+1)/n^s`. These explicit rational operations give precisely
the twelve coefficients in :eq:`risk-c7-finite-mzv`; their rational
unit coefficient sums to zero at this index.

The general theorem proves these transformations before taking limits,
including all finite upper boundaries :math:`G(N+1)H_N(w)`, their
vanishing, and absolute convergence. Specializing its finite coefficient
calculation to index seven does not require a second convergence
assumption. The coefficient table is reproducible by
``leading_constant_expression(7)``; tests additionally retain every
finite upper boundary and compare with the original elementary-symmetric
coefficient route.

Reduction by convergent products and duality
--------------------------------------------

The iterated-integral duality proved in
:doc:`skeleton_risk_c5_closed_form` reverses a binary word and exchanges
zero and one. In the present descending convention it gives

.. math::

   \zeta(2,1)&=\zeta(3),&
   \zeta(2,1,1)&=\zeta(4),&
   \zeta(2,1,1,1)&=\zeta(5),\\
   \zeta(2,1,1,1,1)&=\zeta(6),&
   \zeta(2,1,2)&=\zeta(2,3),&
   \zeta(2,2,1)&=\zeta(3,2),\\
   \zeta(2,1,1,2)&=\zeta(2,4),&
   \zeta(2,1,2,1)&=\zeta(3,3),&
   \zeta(2,2,1,1)&=\zeta(4,2).

Each equality is directly checked by the corresponding finite binary
word; the duality of its convergent integral supplies the equality of
values. The earlier chapter also proves
:math:`\zeta(2,2)=3\zeta(4)/4`.

Splitting the absolutely convergent products into greater, smaller and
equal summation indices gives

.. math::

   \zeta(2,3)+\zeta(3,2)&=\zeta(2)\zeta(3)-\zeta(5),\\
   \zeta(2,4)+\zeta(4,2)&=\zeta(2)\zeta(4)-\zeta(6),\\
   2\zeta(3,3)&=\zeta(3)^2-\zeta(6).

For the triple product, partition the three indices at a finite cutoff
into the six strict orderings, the three choices for either equal pair,
and the all-equal case. Taking the absolutely convergent limit gives

.. math::

   \zeta(2)^3=6\zeta(2,2,2)
                  +3\zeta(2,4)+3\zeta(4,2)+\zeta(6).

In particular, with
:math:`T=\zeta(2,4)+\zeta(3,3)+\zeta(4,2)`,

.. math::

   T&=\zeta(2)\zeta(4)+\tfrac12\zeta(3)^2-\tfrac32\zeta(6),\\
   \zeta(2,2,2)&=
     \frac{\zeta(2)^3-3\zeta(2)\zeta(4)+2\zeta(6)}6.

Substituting into :eq:`risk-c7-finite-mzv` proves the displayed classical
expression by rational algebra. The positive original integral proves
:math:`c_7>0` independently of its signed representation. Finally the
existing normalization gives
:math:`C_8=8^2\,7!\,2^{-5}c_7=10080c_7`.
No numerical integer-relation search, assumption of basis independence,
or new even-zeta evaluation is used. The retained classical monomials
need not be independent; reducing this bounded index does not initiate
an automatic sequence of further individual constants.

Rational intervals and saved records
-------------------------------------

``tools.risk_research.c7_closed_form`` uses the positive rational
ordinary-zeta intervals proved in :doc:`skeleton_risk_c5_closed_form`.
For the fixed orders two through six their Euler-transformed tail is
at most :math:`2^{-N}/(1-2^{1-k})`. Positive factor intervals multiply
endpointwise, including repeated factors in :math:`\zeta(3)^2` and
:math:`\zeta(2)^3`. A negative coefficient reverses the resulting
endpoints. The original positive integral permits intersecting the final
c7 interval with the nonnegative half-line.

For these fixed orders and monomials the rational-operation count is
linear in N; integer lengths grow. This does not assert a uniform
complexity bound for arbitrary zeta weight.

.. code-block:: console

   $ python -m tools.risk_research.c7_closed_form --terms 64 --output .research/skeleton-risk/c7.json
   $ python -m tools.risk_research.c7_closed_form --verify .research/skeleton-risk/c7.json

The independent record schema stores exact c7/C8 intervals, signed
ordinary-zeta and monomial coefficients, descending MZV coefficients,
terms and normalized source hashes. Reverification restores nondefault
terms and compares the complete strictly typed record. The fingerprints
bind imported research utilities, the finite construction and its proof,
and the identities used here. Stable CRLF and LF sources share hashes;
raw snapshots still reject any in-flight source change. Direct builders,
verifiers and both CLI actions preserve that contract.

Duplicate or unknown fields, changed types, coefficients and sources
are rejected. Protected evaluator/proof destinations cannot be replaced;
failed generation and verification preserve existing output/input.
The atomic writer fixes the canonical parent and original leaf in a
caller-controlled directory. It is not a sandbox against another process
replacing that directory inode.

After the formula and evaluator are frozen, fresh cutoff/precision
holdouts compare with the original rational series and positive Arb
integral. Index seven itself was already used in the previous phase,
so it is not described as an unused index. These finite checks verify
implementation and normalization; the convergent identities prove the
formula. No finite-sample guarantee, finite-D remainder or arbitrary-index
classical-zeta reduction follows from this chapter.
