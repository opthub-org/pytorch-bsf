A simultaneous vertex-count and degree regime
==============================================

Retain the uniform-simplex assumptions, the positive constant :math:`C_M`
of :doc:`skeleton_risk_vertex_general` and the explicit finite-degree
remainder from :doc:`skeleton_risk_vertex_general_asymptotics`.
Here both :math:`M` and :math:`D` may grow. The argument controls the
entire dependence on :math:`M`; it does not substitute a growing value
into a fixed-:math:`M` limit.

A constructive uniform regime
------------------------------

For integer :math:`D\geq2^{1024}`, put

.. math::

   N=\lfloor\log_2D\rfloor,\qquad
   \ell=\lceil\log_2(N+2)\rceil,\qquad
   h(D)=1+\left\lfloor\frac{N}{16\ell}\right\rfloor.

.. prf:theorem:: A growing-vertex vertex asymptotic
   :label: thm-risk-joint-vertex-growth

   For every pair of integers :math:`D\geq2^{1024}` and
   :math:`4\leq M\leq h(D)`,

   .. math::
      :label: risk-joint-vertex-error

      \left|\frac{D^{M+1}a_0(M,D)}{C_M}-1\right|
          \leq D^{-3/4}.

   In particular, :math:`h(D)\to\infty`, with
   :math:`h(D)\sim\log D/(16\log\log D)`, and the relative error
   tends to zero uniformly over all the indicated vertex counts.

The start :math:`2^{1024}` is astronomical. This is a conservative
theoretical regime, not a claim of useful finite-degree precision near
that scale. The integer ceiling can decrease when :math:`\ell` jumps;
no monotonicity of :math:`h` is needed or asserted.

.. prf:proof::

   **A uniform lower bound for the positive constant.** Put
   :math:`q=M-1\geq3` and :math:`L=\log2`. The positive integral for
   :math:`c_q` has kernel :math:`H(\theta)=2\operatorname{Re}g(e^{i\theta})`.
   Its monotonicity on :math:`[0,\pi]` follows by differentiating the
   positive integral with respect to :math:`c=\cos\theta`:

   .. math::

      \frac{\partial}{\partial c}
        \frac{(1+u)(1-c)}{1-2uc+u^2}
       =-\frac{(1+u)(1-u)^2}{(1-2uc+u^2)^2}<0\quad(0\leq u<1).

   At :math:`\theta=\pi/2`, the explicit logarithm gives
   :math:`H(\pi/2)=L+\pi/2>2`. On the two arcs
   :math:`|\theta|\in[\pi/2,\pi]`, of total length :math:`\pi`,
   also :math:`|1-e^{i\theta}|^2\leq4`. Therefore

   .. math::

      c_q>q\,2^{q-4},\qquad
      C_M=M^2q!2^{2-q}c_q>\frac{M^2q!q}{4}.

   This bound holds for every :math:`M\geq4`.

   **Control of every error-budget constant.** Use the notation of the
   finite-degree error proof, made explicit here. For :math:`1\leq r\leq q`,
   let

   .. math::

      P_r&=r!r^{r-1},\\
      B_r&=2^{r+1}r\,r!(r+1)^{r-1},\\
      A_r&=\sum_{a=1}^{r-1}\binom ra P_aP_{r-a},\\
      \Delta_r&=2^{r-2}\sum_{a=1}^{r-1}\binom ra
                      (B_aP_{r-a}+P_aB_{r-a}).

   The last two quantities are needed only for :math:`r\geq2`.
   The elementary estimates :math:`r!\leq q^r`, :math:`r\leq q` and
   :math:`r+1\leq2q` give

   .. math::

      P_r\leq q^{2r},\quad B_r\leq(4q^2)^r,\quad
      A_r\leq(2q^2)^r,\quad\Delta_r\leq(10q^2)^r.

   For :math:`A_r`, sum the binomial coefficients. For :math:`\Delta_r`,
   extend the positive sums to both endpoints and use
   :math:`2^{r-2}(2\,5^r)q^{2r}=(10q^2)^r/2`.

   The explicit limiting and diagonal constants are

   .. math::

      J_q={}&2qA_{q-1},\\
      K_q={}&P_q+A_q+q\bigl(2^q\Delta_{q-1}
                         +(20+3^{q-2})A_{q-1}\bigr)\\
           &+\sum_{k=2}^{q-2}\binom qk B_k
              \bigl(2^{q-k+2}\Delta_{q-k}+5\,2^kA_{q-k}\bigr)
              +8(1+q^2+J_q).

   Empty sums, including the sum at :math:`q=3`, are zero.
   Directly :math:`J_q\leq(2q^2)^q`.
   The two terms multiplied by :math:`q` in :math:`K_q` are at most
   :math:`(20q^2)^q` and :math:`(6q^2)^q`, respectively; use
   :math:`20+3^{q-2}\leq2\,3^q` for :math:`q\geq3`.
   The two binomial sums are at most :math:`4(24q^2)^q` and
   :math:`5(10q^2)^q`. The last term is at most
   :math:`16(2q^2)^q`, since :math:`1+q^2\leq(2q^2)^q`.
   It follows that

   .. math::

      K_q\leq29(24q^2)^q\leq(96q^2)^q,

   because :math:`29\leq4^q` for :math:`q\geq3`.

   The already proved finite-degree remainder is

   .. math::

      |a_0(M,D)-C_MD^{-M-1}|
        &\leq R_M(1+\log D)^{q-1}D^{-M-2},\qquad D\geq256,\\
      R_M&=\frac{M^2q!}{2^q}
                \bigl(96K_q+(273+q(q+1))J_q\bigr).

   Since :math:`273+q(q+1)\leq32q^2`, division by the lower bound for
   :math:`C_M` gives

   .. math::

      \frac{R_M}{C_M}
       &\leq96(96q^2)^q+32q^2(2q^2)^q\\
       &\leq(480q^2)^q+(32q^2)^q\leq(512q^2)^q.

   In the first line, :math:`2^{2-q}/q\leq1` was discarded.
   For the second line, :math:`96\leq5^q` and
   :math:`32q^2\leq16^q` for :math:`q\geq3`; the latter follows from
   :math:`q^2\leq4^q` and :math:`32\leq4^q`.
   This controls every constant in the old error theorem with a single
   explicit bound, uniformly in :math:`M`.

   **The growing range.** Here :math:`N\geq1024`, and
   :math:`q\leq N/(16\ell)\leq N+2`.
   Also :math:`512\leq N+2`, :math:`1+\log D<N+2`, and
   :math:`N+2\leq2^\ell`. Consequently

   .. math::

      \left|\frac{D^{M+1}a_0(M,D)}{C_M}-1\right|
       &\leq\frac{(512q^2)^q(N+2)^{q-1}}{D}\\
       &\leq\frac{(N+2)^{4q-1}}{D}
        \leq\frac{2^{4q\ell}}{D}
        \leq\frac{2^{N/4}}{D}\leq D^{-3/4}.

   The final step uses :math:`2^N\leq D`.
   The formula for :math:`h` gives its divergence and stated asymptotic,
   so the same bound tends to zero uniformly in its whole range.

Small exact certificates for enormous degree blocks
---------------------------------------------------

The standard-library tool ``tools.risk_research.vertex_joint_growth``
accepts the integer :math:`N`, rather than materializing :math:`2^N`.
A certificate covers **every integer degree**
:math:`2^N\leq D<2^{N+1}` at the specified vertex count.
It records :math:`\ell`, the maximum permitted vertex count and the
nonnegative integer witnesses

.. math::

   N-16\ell q,\qquad N+2-512,\qquad N+2-q.

The pointwise relative bound is :math:`D^{-3/4}`. A compressed exact
dyadic upper bound for the entire block is
:math:`2^{-\lfloor3N/4\rfloor}`; its denominator exponent is saved
as an integer, with no enormous numerator or denominator serialization.
The certificate stores the six proved budget multipliers and source hashes.
The checker recomputes the full domain, all witnesses and every field,
and rejects altered constants, omitted or additional fields,
bool/float replacements of exact integers, and changed sources.

For example:

.. code-block:: console

   $ .venv/bin/python -m tools.risk_research.vertex_joint_growth --vertices 6 --degree-floor-exponent 1024 --output .research/skeleton-risk/joint6.json
   $ .venv/bin/python -m tools.risk_research.vertex_joint_growth --check .research/skeleton-risk/joint6.json

The source hash policy canonicalizes CRLF pairs to LF, allowing a report
to be verified across platforms. The concurrent-edit guard compares
original raw bytes and therefore still rejects line-ending changes
during a calculation. Failures preserve an existing report.
No optional computer algebra system, numerical backend or runtime API
change is required.

Finite independent checks compare the displayed budget bounds with the
original exact error constants, and compare the new lower bound for
:math:`C_M` with the existing exact-series intervals. These support the
implementation; the all-:math:`M,D` assertion rests on the inequalities
proved above.
