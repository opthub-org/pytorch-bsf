Finite-degree errors at every fixed nonvertex codimension
==========================================================

Keep the uniform-simplex design, evaluation and noise assumptions of
:doc:`skeleton_risk_theory`. This chapter supplies an explicit finite-degree
error for **every fixed pair of integers** :math:`s,q\geq2`, where
:math:`M=s+q`. It does not require a new proof for each next codimension.
The q5 coefficient using c6 is one instance of the theorem. These are
coefficients of sample-size-asymptotic risk, not finite-sample risks.

Write :math:`L(x)=1+\log x` for :math:`x\geq1`, and retain the positive
diagonal :math:`d_j=\delta_q(j)` of
:doc:`skeleton_risk_codimension_three_exact`. Its total mass is

.. math::

   S_q=\sum_{j\geq0}d_j=\frac{c_{q+1}}{q+1}\geq d_0=1,\qquad
   A_{s,q}=\binom{s+q}s^2\frac{(s)_q}{2^q(s-2)!}S_q>0.

The leading order :math:`a_{s-1}(s+q,D)\sim A_{s,q}D^{s-q-2}`
is already proved. The result below quantifies its finite-D error.

Computable diagonal constants
--------------------------------

Let :math:`J_q,K_q` be the explicit constants of
:eq:`risk-general-error-constants`, and let :math:`\widetilde K_q`
be the explicit rational constant of
:eq:`risk-refined-vertex-error-constants`. Define

.. math::
   :label: risk-nonvertex-general-majorants

   Q_q&=\begin{cases}
       4,&q=2,\\
       J_3+4K_3,&q=3,\\
       J_q+\widetilde K_q,&q\geq4,
       \end{cases}\\
   \lambda_q(x)&=\begin{cases}L(x),&q=2,\\1,&q\geq3,\end{cases}
   &\tau_q(x)&=\begin{cases}L(x)+1,&q=2,\\1,&q\geq3.\end{cases}

All are explicitly computable from q, using finite rational arithmetic.
The resulting constants are deliberately conservative. In particular,
the separate sharper q3 and q4 error theorems remain useful.

.. prf:theorem:: Explicit general nonvertex finite-degree error
   :label: thm-risk-nonvertex-general-finite-error

   For all integers :math:`s,q\geq2` and
   :math:`D\geq\max(4,2(s-1))`, put
   :math:`P_{s,q}=q(2s+q-1)/4` and :math:`B_s=(s-1)(s-2)`.
   Then

   .. math::
      :label: risk-nonvertex-general-finite-error

      \left|\frac{a_{s-1}(s+q,D)}{A_{s,q}D^{s-q-2}}-1\right|
      \leq F_{s,q}(D)
      :=\frac{29Q_q\lambda_q(D)}{2\sqrt D}
       +\frac{P_{s,q}+Q_q\left(2\tau_q(D)
              +\lambda_q(D)((s-2)L(D)+B_s)\right)}D.

   Every term on the right decreases on this domain. Thus this proves
   :math:`O(D^{-1/2})` at every fixed :math:`s,q\geq3`, also at
   :math:`s=2,q\geq3`, and :math:`O(\log D/\sqrt D)` for fixed
   :math:`s\geq2,q=2`. It is not a bound uniform in growing s or q.

   For every integer :math:`k\geq2` with :math:`2^k\geq2(s-1)`,
   an exact rational all-tail bound is

   .. math::
      :label: risk-nonvertex-general-dyadic-error

      \sup_{D\geq2^k}
       \left|\frac{a_{s-1}(s+q,D)}{A_{s,q}D^{s-q-2}}-1\right|
      \leq \frac{29Q_q\ell_{q,k}}{2\rho_k}
       +\frac{P_{s,q}+Q_q(2t_{q,k}
            +\ell_{q,k}((s-2)(k+1)+B_s))}{2^k},

   where :math:`\rho_k=2^{k/2}` for even k and
   :math:`\rho_k=(4/3)2^{\lfloor k/2\rfloor}` for odd k,
   :math:`\ell_{2,k}=k+1,t_{2,k}=k+2`, and
   :math:`\ell_{q,k}=t_{q,k}=1` for :math:`q\geq3`.
   This bound tends to zero, so exact search terminates for every fixed
   pair s,q and every positive rational target. An evaluator may reject
   requests exceeding its disclosed selected-work limit.

.. prf:proof::

   **Global diagonal majorants, including q2.** At q2 the elementary
   symmetric formula gives, with :math:`N=j+1`,

   .. math::

      d_j=\frac{2H_j}{N(N+1)}+\frac2{N^2(N+1)}
          \leq\frac{4L(N)}{N^2}.

   At q3, :eq:`risk-general-diagonal-explicit-error` and
   :math:`c_3\leq J_3` apply. The elementary inequality
   :math:`L(x)^2/x\leq4` follows by setting :math:`u=\log x`
   and maximizing :math:`(1+u)^2e^{-u}` on :math:`u\geq0`.
   At :math:`q\geq4`, use
   :eq:`risk-refined-diagonal-explicit-error`,
   :math:`c_q\leq J_q`, and :math:`L(x)\leq x`.
   Hence, for every q in the claimed domain and every j,

   .. math::

      0<d_j\leq\frac{Q_q\lambda_q(j+1)}{(j+1)^2}.

   The q2 logarithm cannot be absorbed into a constant inverse-square
   bound: its displayed exact formula has harmonic growth.

   **Averaging without a difference assumption.** For
   :math:`0\leq L\leq D-s`, put :math:`B=2D+s-L-1`,
   :math:`T=L+B`, and
   :math:`\Pr(J=j)=\binom Lj\binom Bj/\binom TL`.
   Vandermonde's identity gives

   .. math::

      \mathbb E\frac1{(J+1)(J+2)}
       &=\frac{(T+1)(T+2)}{(L+1)(L+2)(B+1)(B+2)},\\
      \mathbb E\frac{L-J}{(J+1)(J+2)}
       &=\frac{L(T+1)}{(L+1)(B+1)(B+2)},\\
      \mathbb E(L-J)&=L^2/T.

   Since :math:`B\geq L`, these imply
   :math:`\mathbb E(J+1)^{-2}\leq8/((L+1)(L+2))` and
   :math:`\mathbb E[(L-J)/((J+1)(J+2))]\leq2/(D+2s+1)`.
   On :math:`J=L` the averaging difference vanishes. On the other
   events use :math:`\mathbf1_{J<L}\leq L-J` and
   :math:`(J+1)^{-2}\leq2/((J+1)(J+2))`.
   As :math:`J+1,L+1\leq D`, the global bound above yields both

   .. math::

      |\mathbb E d_J-d_L|
        &\leq Q_q\lambda_q(D)\left(\frac4{D+2s+1}+\frac1T\right)
         \leq\frac{9Q_q\lambda_q(D)}{2D},\\
      |\mathbb E d_J-d_L|
        &\leq\frac{10Q_q\lambda_q(D)}{(L+1)(L+2)}.

   Split the sum over L at an arbitrary positive integer H, and sum
   the reciprocal tail exactly. Choosing :math:`H=\lfloor\sqrt D\rfloor`
   proves

   .. math::

      \sum_{L=0}^{D-s}|\mathbb E d_J-d_L|
       \leq Q_q\lambda_q(D)\left(\frac{9H}{2D}+\frac{10}{H+1}\right)
       \leq\frac{29Q_q\lambda_q(D)}{2\sqrt D}.

   This argument uses only the majorant and positivity, not a
   q-dependent discrete-derivative estimate.

   **Angular weights and tail.** Set :math:`n=D-s` and

   .. math::

      w_L=\prod_{r=2}^{s-1}(1-(L+r)/D),\qquad
      U_D=\sum_{L=0}^{n}w_L\mathbb E d_J.

   The empty product at s2 is one. At all active degrees D>=s,
   every factor lies in [0,1], and the finite-product inequality gives

   .. math::

      \sum_{L=0}^n(1-w_L)d_L
       \leq\frac{Q_q\lambda_q(D)}D
          ((s-2)H_{n+1}+B_s).

   The inverse-square tail is at most :math:`Q_q/(n+1)` for q>=3.
   At q2, :math:`L(x)/x^2` decreases for x>=1, so comparison with
   its integral from n+1 to infinity gives
   :math:`Q_q(2+\log(n+1))/(n+1)`.
   Both are at most :math:`Q_q\tau_q(D)/(D-s+1)`.
   Consequently an unsimplified error, valid at every active degree, is

   .. math::

      |U_D-S_q|\leq E_{s,q}(D)
       :=Q_q\left[\lambda_q(D)\left(\frac{9H}{2D}+\frac{10}{H+1}
              +\frac{(s-2)H_{D-s+1}+B_s}D\right)
             +\frac{\tau_q(D)}{D-s+1}\right].

   **Exact normalization.** The positive-diagonal coefficient identity,
   with no asymptotic substitution, is

   .. math::

      \frac{a_{s-1}(s+q,D)}{A_{s,q}D^{s-q-2}}
       =P_D\frac{U_D}{S_q},\qquad
      P_D=\prod_{i=0}^{q-1}(1+(s+i)/(2D))^{-1}.

   Here :math:`0<P_D\leq1`, :math:`S_q\geq1`, and
   :math:`1-P_D\leq P_{s,q}/D`. Thus the normalized error is at
   most :math:`1-P_D+E_{s,q}(D)`. For D>=2(s-1), use
   :math:`1/(D-s+1)\leq2/D` and
   :math:`H_{D-s+1}\leq L(D)` to obtain the theorem.
   The functions :math:`D^{-1/2}`, :math:`L(D)/D`, and D^-1 decrease
   for D>=1. At q2 the additional functions :math:`L(D)/\sqrt D`
   and :math:`L(D)^2/D` decrease for D>=4. Hence evaluation at 2^k
   bounds every later degree. Finally :math:`\log2<1` and
   :math:`\sqrt2>4/3` prove the rational bound. Its right side decreases
   with k>=2, including both transitions between even and odd exponents:
   the q2 root factors are :math:`(2m+1)/2^m` and
   :math:`3(2m+2)/(4\,2^m)`, and all polynomial-over-dyadic terms
   decrease on that domain.

Certificates, costs and scope
-------------------------------

``tools.risk_research.nonvertex_general_error`` specializes the theorem
with exact fractions, ``isqrt`` and integer bit-length logarithm bounds.
It takes the smaller of the unsimplified point bound and the applicable
all-tail bound. It constructs no array proportional to D. Inactive
degrees D<s have coefficient exactly zero and normalized error one.
The leading-constant interval comes from the independently proved original
rational series for :math:`C_{q+2}`, with its exact nonvertex scale;
``constant_terms`` controls that numerical truncation only.

The selected-work counter is
:math:`(q+1)^3+(q+1)^2T+s+q+b(q+|s-q-2|+2)`, where T is the
requested constant cutoff and b is the relevant degree/input bit count.
Each entry point checks this counter before its expensive arithmetic.
Its default limit is 200,000 and its accepted maximum is 1,000,000.
This is an input guard, not a proved bound for bit complexity, CPU time
or RSS. Different entry points may additionally check target or threshold
bit lengths, and generation records the counter for its saved degree and
cutoff. Rejected requests do not invalidate the all-integer theorem.

.. code-block:: console

   $ python -m tools.risk_research.nonvertex_general_error --face-size 3 --outside 5 --degree 1152921504606846976 --constant-terms 31 --relative-target 1/100 --work-limit 190000 --output .research/nonvertex-general.json
   $ python -m tools.risk_research.nonvertex_general_error --verify .research/nonvertex-general.json

The new schema restores all saved parameters and rejects scalar-type
changes, unknown or tampered fields, duplicate JSON keys and conflicting
generation overrides. Direct generation and verification, and the outer
CLI, compare bound raw source bytes before and after their calculations;
canonical source hashes alone normalize CRLF to LF. Bound source/proof
output destinations are rejected before computation, and canonical output
parents are pinned before the atomic write. Existing saved schemas are
unchanged. Output directories must remain under caller control; these
checks do not provide a sandbox against hostile directory replacement.

The theorem leaves q3/q4 sharper estimates and the separate growing-M,D
vertex regime intact. It claims neither O(log(D)/D) at arbitrary q nor
uniformity when s,q,D grow together. Finite independent checks of original
scalar/Gram, integer outside kernels and positive integrals validate the
implementation; the displayed inequalities establish the infinite tail.
