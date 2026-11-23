Computable vertex centers and a single-log growing domain
=========================================================

Keep the uniform-simplex assumptions and normalization of
:doc:`skeleton_risk_vertex_general`. Write :math:`q=M-1`,
:math:`n=M-2`, :math:`L=\log2`, and

.. math::

   S_M=M^2q!(2L)^q,\qquad
   b_M=\frac{\sqrt q}{4\sqrt{\pi L(L-1/2)}},\qquad B_M=S_Mb_M.

The results below concern the sample-size-asymptotic vertex risk
coefficient. Finite-sample risks and the separate comparison region
:math:`D\geq8M^2` are not their subject.

A larger domain from the single-log error
------------------------------------------

.. prf:theorem:: A single-log growing-vertex approximation
   :label: thm-risk-composite-joint-domain

   For every integer :math:`D\geq2^{256}`, put

   .. math::

      N=\lfloor\log_2D\rfloor,\qquad
      \ell=\lceil\log_2(N+2)\rceil,\qquad
      h_{**}(D)=1+\max\left(3,
                      \left\lfloor\frac{N-4\ell}{8\ell}\right\rfloor\right).

   For every integer :math:`4\leq M\leq h_{**}(D)`,

   .. math::
      :label: risk-composite-joint-domain-error

      \left|\frac{D^{M+1}a_0(M,D)}{C_M}-1\right|
        \leq2^{-g(q)}D^{-3/4},

   with :math:`g(q)=0` for :math:`3\leq q\leq160`, and
   :math:`g(q)=\lfloor q/3\rfloor+2-\lceil\log_2q\rceil` for
   :math:`q\geq161`.

This domain strictly contains the one in
:doc:`skeleton_risk_vertex_joint_growth_refined`. At :math:`N=256`
it includes :math:`M=4`; at :math:`N=1024` its vertex ceiling is 12,
compared with the previous ceiling 8. Its growth is asymptotic to
:math:`\log D/(8\log\log D)`, rather than the previous denominator 12.
The degree start is still astronomical. The ceiling can decrease when
:math:`\ell` jumps, so no monotonicity of the ceiling is asserted.

.. prf:proof::

   **Retain all dependence on the vertex count in the single-log bound.**
   For :math:`q\geq4`, use the explicit constants of
   :doc:`skeleton_risk_vertex_refined_asymptotics`. To avoid confusing
   their :math:`B_r,G_r` with the leading term :math:`B_M`, call their
   elementary bounds :math:`P_r,G_r,A_r,F_r` here. The bounds already
   proved in :doc:`skeleton_risk_vertex_joint_growth` give, for
   :math:`1\leq r\leq q`,

   .. math::

      P_r\leq q^{2r},\quad G_r\leq(4q^2)^r,\quad
      A_r\leq(2q^2)^r,\quad F_r\leq(10q^2)^r.

   For every positive integer :math:`p`,

   .. math::

      I_{p,p+1}=\sum_{j=0}^p\frac{p!}{(p-j)!p^{j+1}}
          \leq\frac{p+1}{p}\leq2.

   Each summand is at most :math:`1/p`. Thus the first moments in
   the single-log proof satisfy :math:`S_k\leq3G_k` and
   :math:`T_r\leq3A_r`. In its diagonal constant, the two terms in
   :math:`q\widetilde U_{q-1}` are bounded by
   :math:`(20q^2)^q` and :math:`(6q^2)^q`: for the latter use
   :math:`56+3^{q-2}\leq3^q` for :math:`q\geq4`.
   Its two positive binomial sums are bounded by
   :math:`6(24q^2)^q` and :math:`5(10q^2)^q`, respectively.
   The first follows by extending the sum to both endpoints and
   expanding :math:`6(4+20)^qq^{2q}`; the second uses
   :math:`5(8+2)^qq^{2q}`.
   Also :math:`2P_q+A_q\leq3(2q^2)^q`,
   :math:`J_q\leq(2q^2)^q`, and
   :math:`8(1+q^2+J_q)\leq16(2q^2)^q`.
   Consequently

   .. math::

      \widetilde K_q\leq32(24q^2)^q\leq(96q^2)^q.

   The last step uses :math:`32\leq4^q` for :math:`q\geq4`.
   Divide the explicit :math:`\widetilde R_M` by
   :math:`C_{\rm arc}=M^2q!q/4`. Exactly as in the earlier budget
   proof, :math:`273+q(q+1)\leq32q^2`,
   :math:`96\leq5^q`, and :math:`32q^2\leq16^q` imply

   .. math::

      \frac{\widetilde R_M}{C_{\rm arc}}
       \leq96(96q^2)^q+32q^2(2q^2)^q
       \leq(512q^2)^q.

   The refined large-vertex lower bound from the preceding joint chapter
   gives :math:`C_{\rm arc}/C_M\leq2^{-g(q)}`, without a gap at
   :math:`q=160,161`. Hence for every :math:`M\geq5,D\geq256`,

   .. math::

      \left|\frac{D^{M+1}a_0(M,D)}{C_M}-1\right|
       \leq 2^{-g(q)}(512q^2)^q\frac{1+\log D}{D}.

   This is an inequality with explicit dependence on both parameters,
   rather than substitution into a fixed-:math:`M` limit.

   **Spend two logarithmic factors per outside coordinate.**
   If :math:`q\geq4` is allowed by the new ceiling, then
   :math:`8\ell q+4\ell\leq N`, and :math:`\ell\geq9`.
   Therefore :math:`512q^2\leq8N^2/\ell^2\leq(N+2)^2`, while
   :math:`1+\log D<N+2\leq2^\ell`. The last error bound is at most

   .. math::

      2^{-g(q)}\frac{2^{(2q+1)\ell}}D
          \leq2^{-g(q)}\frac{2^{N/4}}D
          \leq2^{-g(q)}D^{-3/4}.

   **Cover the critical four-vertex case separately.**
   For :math:`q=3`, keep the previous squared-log estimate. Its explicit
   arc budget satisfies :math:`(512\cdot9)^3<2^{37}`. Thus the error
   is at most :math:`2^{37+2\ell}/D`.
   For :math:`N\geq256`, :math:`\ell\leq N/32+2`.
   Indeed this holds directly at :math:`\ell=9`; at :math:`\ell\geq10`
   use :math:`N\geq2^{\ell-1}-1\geq32(\ell-2)`, with the last
   inequality following by induction from :math:`\ell=10`.
   Consequently :math:`37+2\ell\leq41+N/16\leq N/4`.
   This proves the same degree error for :math:`M=4`.

   **Inclusion and strict improvement.**
   An old admissible :math:`q\leq\lfloor N/(12\ell)\rfloor` with
   :math:`q\leq3` is automatically allowed. If :math:`q\geq4`, then
   :math:`N\geq48\ell`, and in particular
   :math:`N/(12\ell)\leq(N-4\ell)/(8\ell)`.
   This proves inclusion at every old pair, including ceiling jumps.
   The pairs :math:`(M,D)=(4,2^{256})` and :math:`(12,2^{1024})`
   show strict enlargement. Finally :math:`\ell\sim\log_2N`
   gives the stated asymptotic ceiling.

Centers that contain no unknown leading constant
-------------------------------------------------

Use the explicit coefficient :math:`\beta` in
:doc:`skeleton_risk_vertex_large_dimension_second_order`. Define

.. math::

   T_M=\begin{cases}
      B_M,&4\leq M<258,\\
      B_M(1+\beta/(M-2)),&M\geq258.
   \end{cases}

For :math:`4\leq M<162`, use the elementary local radius from
:doc:`skeleton_risk_vertex_large_dimension`:

.. math::

   r_M=\left(1+\frac1{2n}\right)
       \left(\frac{21}{n}+4\sqrt n(79/80)^{n-2}
                    +\frac{16}{3\sqrt n}(64/65)^n\right)+\frac1{2n}.

An outward rational enclosure of its square roots is used by the tool.
For :math:`162\leq M<258`, put :math:`r_M=6/n`; for
:math:`M\geq258`, put :math:`r_M=84/n^2`.
The three existing theorems prove
:math:`|C_M/T_M-1|\leq r_M` on their respective ranges. The last
radius is conservative because division by :math:`1+\beta/n>1`
could only decrease it. The small-:math:`M` fallback can be wide and
is not an improvement over the exact-series evaluator there.

.. prf:corollary:: A computable composite center
   :label: cor-risk-composite-vertex-center

   Throughout the new joint domain, let :math:`t_M=T_M/S_M`, and let
   :math:`[t_-,t_+]` be any positive rational enclosure of :math:`t_M`.
   Put

   .. math::

      \widehat t=(t_-+t_+)/2,\qquad
      u_M=\frac{t_+-t_-}{t_++t_-},\qquad
      b_M^*=(1+u_M)(1+r_M)-1.

   If :math:`\delta` bounds the degree error relative to :math:`C_M`,
   then

   .. math::
      :label: risk-composite-computable-center

      \left|\frac{D^{M+1}a_0(M,D)}{S_M\widehat t}-1\right|
         \leq b_M^*+(1+b_M^*)\delta.

   The exact center :math:`T_M` instead has radius
   :math:`r_M+(1+r_M)\delta`.

.. prf:proof::

   Factor the displayed quotient into the degree ratio
   :math:`D^{M+1}a_0/C_M`, the constant ratio :math:`C_M/T_M`,
   and the elementary-evaluation ratio :math:`t_M/\widehat t`.
   Their errors are bounded by :math:`\delta,r_M,u_M`.
   Expansion of the product and the triangle inequality give
   :math:`(1+\delta)(1+r_M)(1+u_M)-1`.
   This proves the formula, including every cross term, without requiring
   a small radius in the first finite range. In particular no unknown
   value :math:`C_M` remains in the approximation center.

Compressed certificates and independent checks
-----------------------------------------------

``tools.risk_research.vertex_composite_asymptotic`` evaluates
:math:`[t_-,t_+]` with the existing rational logarithm, arctangent and
integer-square-root enclosures. It retains :math:`S_M` symbolically;
neither :math:`q!`, :math:`2^N` nor a risk table is constructed.
For a full block :math:`2^N\leq D<2^{N+1}` it uses

.. math::

   E=\begin{cases}
       N-37-2\ell,&q=3,\\
       N-(2q+1)\ell+g(q),&q\geq4,
     \end{cases}\qquad \delta=2^{-E}.

The certificate records a rational base plus a rational multiple of
:math:`2^{-E}` rather than materializing this denominator. A target is
checked by exact integer comparison with a bounded shift. Thus even
an exponent with thousands of decimal digits does not cause a shift
by that value. The target concerns the stated block, not every later
block across ceiling jumps.

The separate constant, degree, elementary-evaluation and combined
budgets are saved, along with their center, denominator convention,
full domain witnesses and source/proof fingerprints. Verification
restores every recorded option and recomputes the entire record. It
rejects duplicate/unknown keys, changed types, bounds or sources.
Integer text conversion uses bounded chunks without modifying a
process-wide digit limit.

.. code-block:: console

   $ .venv/bin/python -m tools.risk_research.vertex_composite_asymptotic --vertices 258 --degree-floor-exponent 40000 --bits 47 --relative-target 1/100 --output .research/skeleton-risk/composite258.json
   $ .venv/bin/python -m tools.risk_research.vertex_composite_asymptotic --verify .research/skeleton-risk/composite258.json

Source/output aliases are rejected before calculation and again at
the resolved parent plus original leaf before atomic replacement.
Failures retain existing output. The directory is caller-controlled;
this is not a sandbox against replacement of that canonical directory
inode by another process.

Independent finite checks compare the newly majorized single-log
constants with their original exact formulas, the centers with the
original positive integral, and feasible coefficients with the original
scalar or transformed formulas. Huge-degree certificates check exact
compressed inequalities. These checks support the implementation;
the proof above supplies the full growing domain.
