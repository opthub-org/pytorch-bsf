A broader simultaneous vertex-count and degree regime
======================================================

Keep the uniform-simplex assumptions and normalization of
:doc:`skeleton_risk_vertex_general`. The new domain below strictly
contains the earlier domain in :doc:`skeleton_risk_vertex_joint_growth`.
This is a relative approximation to the vertex coefficient, distinct
from the risk comparison at :math:`D\geq8M^2`.

.. prf:theorem:: Broader growing-vertex approximation
   :label: thm-risk-refined-joint-vertex-growth

   For integer :math:`D\geq2^{384}`, put

   .. math::

      N=\lfloor\log_2D\rfloor,\qquad
      \ell=\lceil\log_2(N+2)\rceil,\qquad
      h_*(D)=1+\left\lfloor\frac{N}{12\ell}\right\rfloor.

   For every integer :math:`4\leq M\leq h_*(D)`, with :math:`q=M-1`,

   .. math::
      :label: risk-refined-joint-vertex-error

      \left|\frac{D^{M+1}a_0(M,D)}{C_M}-1\right|
        \leq 2^{-g(q)}D^{-3/4},

   where

   .. math::

      g(q)=\begin{cases}
        0,&3\leq q\leq160,\\
        \lfloor q/3\rfloor+2-\lceil\log_2q\rceil,&q\geq161.
      \end{cases}

   The integer :math:`g(q)` is nonnegative. In particular, the uniform
   error is at most :math:`D^{-3/4}` throughout the new domain.

At :math:`N=384`, the domain already includes :math:`M=4`.
At :math:`N=1024`, the vertex ceiling increases from 6 to 8.
The new ceiling is asymptotic to :math:`\log D/(12\log\log D)`,
compared with the earlier denominator 16. The degree threshold remains
astronomical; neither ceiling is a claim of practical finite-degree
accuracy. Ceilings may decrease at jumps of :math:`\ell`.

.. prf:proof::

   **Retain every vertex-count dependent error constant.** Write
   :math:`C_{\rm arc}=M^2q!q/4`. The elementary arc argument and the
   explicit budget estimates in :doc:`skeleton_risk_vertex_joint_growth`
   prove, separately and for every integer :math:`q\geq3`,

   .. math::

      C_M>C_{\rm arc},\qquad
      \frac{R_M}{C_{\rm arc}}\leq(512q^2)^q.

   Here the original all-degree error estimate is

   .. math::

      |a_0(M,D)-C_MD^{-M-1}|
        \leq R_M(1+\log D)^{q-1}D^{-M-2},\qquad D\geq256.

   In particular, the second inequality bounds the explicit
   :math:`R_M` after division by the arc bound, before replacing
   :math:`C_{\rm arc}` by :math:`C_M`. No growing value is substituted
   into a fixed-:math:`M` limit.

   **Use the sharper large-vertex lower bound without a gap.** For
   :math:`4\leq M\leq161`, keep the arc bound. For :math:`M\geq162`,
   :doc:`skeleton_risk_vertex_large_dimension_refined` proves
   :math:`C_M\geq(1-6/(M-2))B_M`. With :math:`L=\log2`, its exact
   normalization is

   .. math::

      \frac{B_M}{M^2q!}
        =\frac{(2L)^q\sqrt q}
                    {4\sqrt{\pi L(L-1/2)}}.

   The already proved bounds :math:`2/3<L<7/10`, together with
   :math:`\pi<4`, give
   :math:`16\pi L(L-1/2)<224/25<9`. Consequently

   .. math::

      \frac{C_M}{M^2q!}
        >\frac{1-6/(q-1)}3(4/3)^q\sqrt q
        \geq2^{\lfloor q/3\rfloor}\quad(q\geq161).

   Indeed, :math:`\sqrt q\geq12`,
   :math:`1-6/(q-1)\geq77/80`, and their product divided by 3 is at
   least :math:`77/20>1`. Also :math:`(4/3)^3>2`; grouping triples
   proves the last inequality, including remainders 0, 1 and 2.
   The bound :math:`\pi<4` follows directly from
   :math:`\pi=4\int_0^1(1+x^2)^{-1}dx<4`.

   Since :math:`q\leq2^{\lceil\log_2q\rceil}`, the ratio of the arc
   bound to this new bound is at most :math:`2^{-g(q)}`. To check that
   :math:`g(q)\geq0`, put :math:`k=\lceil\log_2q\rceil\geq8`.
   The elementary induction :math:`2^m\geq3m` for :math:`m\geq4`
   gives :math:`q>2^{k-1}\geq3(k-1)`, hence
   :math:`\lfloor q/3\rfloor+2\geq k+1`. Thus throughout both
   branches,

   .. math::

      \frac{R_M}{C_M}\leq2^{-g(q)}(512q^2)^q.

   This explicitly joins the finite range :math:`4\leq M\leq161`
   with the refined large-vertex range.

   **Spend three logarithmic factors rather than four.** Here
   :math:`N\geq384`, :math:`\ell\geq9` and
   :math:`q\leq N/(12\ell)`. In particular,

   .. math::

      512q^2
        \leq \frac{512N^2}{(12\ell)^2}
        \leq (N+2)^2,\qquad
      1+\log D<N+2\leq2^\ell.

   The middle inequality uses :math:`512<(12\ell)^2`. The logarithm
   bound uses :math:`D<2^{N+1}` and :math:`\log2<1`.
   Combining the explicit budgets,

   .. math::

      \left|\frac{D^{M+1}a_0(M,D)}{C_M}-1\right|
      &\leq2^{-g(q)}\frac{(512q^2)^q(1+\log D)^{q-1}}D\\
      &\leq2^{-g(q)}\frac{2^{(3q-1)\ell}}D
        \leq2^{-g(q)}\frac{2^{3q\ell}}D\\
      &\leq2^{-g(q)}\frac{2^{N/4}}D
        \leq2^{-g(q)}D^{-3/4}.

   All estimates apply to every integer degree in its entire dyadic
   block. For :math:`N\geq1024`,
   :math:`\lfloor N/(12\ell)\rfloor\geq
   \lfloor N/(16\ell)\rfloor` at that same :math:`\ell`. This proves
   inclusion of every old pair, including at ceiling jumps. The pair
   :math:`(M,D)=(8,2^{1024})` and the new block starting at
   :math:`2^{384}` show strict enlargement. Finally,
   :math:`\ell\sim\log_2N` proves the stated growth rate.

Compressed exact certificates and limitations
----------------------------------------------

``tools.risk_research.vertex_joint_growth_refined`` accepts :math:`M,N`,
without constructing :math:`2^N` or a risk coefficient. Its certificate
covers every integer :math:`2^N\leq D<2^{N+1}`. The constant lower
bound is encoded after dividing by :math:`M^2q!` as an exact rational
coefficient times an integer power of two; factorials and that power
are not materialized. The record keeps both the pointwise bound above
and the block bound :math:`2^{-\lfloor3N/4\rfloor-g(q)}`.

The integer margins :math:`N-12\ell q` and :math:`(12\ell)^2-512`
certify the degree budget. The checker recomputes the complete payload,
including domain, branch, gain, every margin, proof/source hashes and
compressed bounds. It rejects duplicate or unknown fields and bool,
float or string substitutions for integers. Very large integer text is
handled in local bounded chunks, without changing the interpreter's
integer conversion limit.

.. code-block:: console

   $ .venv/bin/python -m tools.risk_research.vertex_joint_growth_refined --vertices 8 --degree-floor-exponent 1024 --output .research/skeleton-risk/joint-refined8.json
   $ .venv/bin/python -m tools.risk_research.vertex_joint_growth_refined --verify .research/skeleton-risk/joint-refined8.json

Verification accepts no generation overrides and preserves its input.
Generation checks source collisions before calculation, then pins the
resolved parent while retaining the original leaf name, checks again
and atomically replaces output. Source hashes normalize CRLF to LF;
raw bytes are compared before saving to detect concurrent edits.
An existing output is retained on failure. Output directories are
controlled by the caller; this is not a sandbox against another
process replacing the resolved directory itself.

Independent finite comparisons use the original exact error constants,
old constant-series intervals and feasible scalar/Gram or positive
kernel evaluations. Huge-degree boundary checks use exact compressed
inequalities. These check normalization and implementation, rather
than evaluating a coefficient at an astronomical degree or replacing
the proof of the full growing domain. No finite-sample assertion,
higher-order degree expansion, or API change is made.
