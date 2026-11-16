Dyadic enclosures for multiple zeta values
=============================================

This chapter supplies a rational numerical evaluator for the finite
expressions in :doc:`skeleton_risk_leading_constant_mzv`. It retains the
strict descending-index convention. It does not change the expression
compiler, sampling assumptions, finite-degree risk, or production API.

A half-interval identity
-------------------------

For any positive index tuple :math:`v=(s_1,\ldots,s_d)`, including a first
index equal to one, put

.. math::

   L_v(z)=\sum_{n_1>\cdots>n_d\geq1}
            \frac{z^{n_1}}{n_1^{s_1}\cdots n_d^{s_d}},\qquad
   L_\varnothing(z)=1.

Encode the tuple by the binary word
:math:`w(v)=0^{s_1-1}1\cdots0^{s_d-1}1`. The word is read from the
outermost to innermost integral. Write
:math:`\omega_0(t)=dt/t`, :math:`\omega_1(t)=dt/(1-t)`.
For a word ending in one, its integral over
:math:`z>t_1>\cdots>t_W>0` is :math:`L_v(z)`.
This convention agrees with Definition 2.1, Lemma 2.2 and Proposition 2.3
on printed pages 45--47 of `Bachmann's lecture notes, version 24
<https://www.henrikbachmann.com/uploads/7/7/6/3/77634444/mzv_2025_lecture_notes_v24.pdf>`_.
The following proof also derives the representation used here.

.. prf:theorem:: Dyadic split for every admissible MZV
   :label: thm-risk-mzv-dyadic-split

   Let :math:`v` be any nonempty positive integer tuple with
   :math:`s_1\geq2`, and let :math:`w=w(v)` have length :math:`W`.
   For each cut :math:`w=p_j r_j`, :math:`|p_j|=j`, let
   :math:`p_j^*` be the reversed complement of :math:`p_j`.
   Decode each nonempty word ending in one back to its index tuple.
   Then

   .. math::

      \zeta(v)=\sum_{j=0}^{W} L_{p_j^*}(1/2)L_{r_j}(1/2).

   Empty factors denote one. Every nonempty factor is an absolutely
   convergent multiple polylogarithm at one half, even if its first
   index is one.

.. prf:proof::

   In :math:`|z|<1`, the defining series is absolutely convergent:
   ignoring all denominators bounds the coefficient at :math:`z^n`
   by :math:`\binom{n-1}{d-1}`, whose generating function is
   :math:`z^d/(1-z)^d`. Termwise differentiation gives
   :math:`L_{a,u}'(z)=L_{a-1,u}(z)/z` for :math:`a>1`.
   At :math:`a=1`, summing :math:`\sum_{n>m}z^{n-1}` gives
   :math:`L_{1,u}'(z)=L_u(z)/(1-z)`, including empty :math:`u`.
   The zero initial value of every nonempty tuple proves the stated
   iterated-integral formula by induction on the binary word length.
   At :math:`z=1`, monotone convergence and the convergent MZV sum
   identify the integral with :math:`\zeta(v)`; convergence also follows
   from the harmonic majorant in
   :doc:`skeleton_risk_leading_constant_mzv`.

   Partition the integration simplex by the number :math:`j` of its
   variables above :math:`1/2`. The boundaries where a variable equals
   :math:`1/2` have measure zero. The integrand is nonnegative, so
   Tonelli and the already finite total integral justify this partition
   and factorization. The lower variables give :math:`L_{r_j}(1/2)`.
   Substitute :math:`u_i=1-t_{j+1-i}` in the upper variables: their order
   reverses and each zero/one differential form becomes the other one.
   They therefore give :math:`L_{p_j^*}(1/2)`.
   The original word starts in zero and ends in one. Thus every nonempty
   reflected prefix and every nonempty suffix ends in one and has the
   required convergent series. This also covers both endpoint cuts.

A rational geometric tail
---------------------------

.. prf:theorem:: Polylogarithm tail at one half
   :label: thm-risk-mzv-dyadic-tail

   For every positive tuple :math:`v=(a,u)` of depth :math:`d\geq1`
   and every integer :math:`T\geq1`, define

   .. math::

      P_T(v)&=\sum_{n=1}^T\frac{H_{n-1}(u)}{2^n n^a},\\
      E_T(v)&=\frac{1}{2^T(T+1)^a}
                    \sum_{k=0}^{\min(d-1,T)}\binom Tk.

   Here the harmonic word uses strict descending indices and the empty
   harmonic word is one. Then

   .. math::

      P_T(v)\leq L_v(1/2)
        \leq\min\{1,P_T(v)+E_T(v)\}.

   For the empty word use the exact interval :math:`[1,1]` and zero tail.
   For each fixed tuple, the error is a polynomial in :math:`T` times
   :math:`2^{-T}`. No numerical logarithm or floating-point rounding
   enters this bound.

.. prf:proof::

   For each outer index :math:`n`, there are
   :math:`\binom{n-1}{d-1}` strict inner chains. Their reciprocal products
   are at most one. In the omitted tail, :math:`n^{-a}\leq(T+1)^{-a}`.
   The exact binomial identity

   .. math::

      \sum_{n>T}\frac{\binom{n-1}{d-1}}{2^n}
        =\frac1{2^T}\sum_{k=0}^{\min(d-1,T)}\binom Tk

   now proves the error bound. To see the identity without assuming an
   infinite rearrangement, put its proposed right-hand side equal to
   :math:`Q_T`. Pascal's formula gives
   :math:`Q_T-Q_{T+1}=2^{-T-1}\binom T{d-1}`. For fixed :math:`d`,
   :math:`Q_T\to0`; summing this difference to a finite upper endpoint
   and taking the limit proves the identity. The full counting sum is
   one, proving :math:`L_v(1/2)\leq1`. All omitted terms are positive.

Target widths and signed constants
------------------------------------

For a finite rational expression
:math:`X=c_\varnothing+\sum_v c_v\zeta(v)`, apply the dyadic identity
to every nonconstant word. Let :math:`[l_u,h_u]\subseteq[0,1]` be the
polylogarithm intervals. Products use :math:`[l_ul_v,h_uh_v]`.
Since

.. math::

   h_uh_v-l_ul_v
       =(h_u-l_u)h_v+l_u(h_v-l_v)
       \leq E_T(u)+E_T(v),

a computable upper bound for the final interval width is

.. math::

   B_T(X)=\sum_v|c_v|\sum_{j=0}^{|w(v)|}
                         (E_T(p_j^*)+E_T(r_j)).

Signs select the proper endpoints of each MZV interval. In particular,
the absolute coefficient sum, rather than the signed sum, controls
cancellation. For every fixed finite expression, :math:`B_T(X)\to0`.
Testing :math:`T=1,2,4,8,\ldots` therefore terminates for every requested
absolute width :math:`2^{-b}`, :math:`b\geq1`, if operational limits are
removed. Every accepted implementation result has actual rational width
at most its target. This is an absolute-width guarantee; no relative
accuracy is inferred near zero.

The method applies to every admissible MZV, hence to every fixed
:math:`c_q`, :math:`q\geq3`, from the existing compiler. It does not
require a classical-zeta reduction or a minimal basis. Multiplication by
:math:`(q+1)^2q!/2^{q-2}` gives the vertex leading-constant interval;
its width is scaled by that same positive factor. The requested width
in constant mode concerns :math:`c_q`, not the scaled vertex constant.
No finite-degree risk error or joint-limit assertion follows from this
numerical theorem.

Implementation and bounded resources
--------------------------------------

``tools.risk_research.mzv_dyadic`` exposes fixed-cutoff
``polylog_enclosure`` and ``mzv_enclosure`` and target-width
``mzv_precision_enclosure`` and ``constant_enclosure``. Its exact
harmonic table is updated in decreasing depth, so every suffix is still
:math:`H_{n-1}` during a strict-chain update. Polylogarithm contributions
are accumulated before that update. Equal suffixes are shared.

For the resulting set of nonempty suffixes :math:`S`, one cutoff uses
:math:`O(T\sum_{v\in S}(|v|+\mathrm{wt}(v)))` selected arithmetic units,
plus word-splitting, cutoff selection and expression construction.
There are at most :math:`W+1` cuts for a word of weight :math:`W`;
constructing all decoded cuts costs :math:`O((W+1)^2)` entries.
The implementation charges for these selected operations and rejects
work beyond the requested limit (default 200,000, maximum 1,000,000).
It checks predicted numerical table work before entering its loops.
The work counter is not a CPU/RSS or bit-complexity bound, nor does it
count every Python instruction. The mathematical arbitrary-precision
termination theorem is distinct from this bounded implementation, which
can refuse large indices, weights, precisions or cutoffs.

``build_record`` saves a direct MZV; ``build_constant_record`` saves a
compiled :math:`c_q` expression and its vertex scaling. Both preserve
all options, selected cutoff/work, exact endpoints/width, convention,
and canonical source/proof hashes. ``verify_record`` restores the
options and compares the whole strict JSON object. Raw source bytes
are checked before and after each generation, including direct API
calls; the independent CLI also checks around the builder. LF/CRLF
canonical hashes are portable, while a raw change during a calculation
is rejected. Generation cannot replace a bound evaluator or proof.
All numerical functions require only the Python standard library.

These are analytical error and termination results. Finite tests,
comparisons with the previous harmonic-tail evaluator, independent
original constant series/integrals, and measured resource comparisons
validate the implementation separately; they do not establish the
all-index theorem or a universal wall-time speedup.
