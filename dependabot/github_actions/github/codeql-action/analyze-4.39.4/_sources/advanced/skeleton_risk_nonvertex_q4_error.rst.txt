Finite-degree errors for every fixed nonvertex q4 face size
===========================================================

Keep the uniform-simplex design, evaluation and noise assumptions of
:doc:`skeleton_risk_theory`. Put :math:`M=s+4` for a fixed integer
:math:`s\geq2`. The existing positive-diagonal theorem gives

.. math::

   a_{s-1}(s+4,D)\sim A_{s,4}D^{s-6},\qquad
   A_{s,4}=\binom{s+4}s^2\frac{(s)_4}{16(s-2)!}S,\qquad
   S=\sum_{j\geq0}\delta_4(j)=\frac{c_5}{5}>0.

This chapter adds an explicit finite-degree error. It uses a cutoff
argument and does not assume the q3 discrete-derivative estimate for q4.
The rate is deliberately conservative: :math:`O(D^{-1/2})`, rather than
the :math:`O(\log D/D)` proved separately at q3. These are coefficients
of sample-size-asymptotic risk, not finite-sample risk.

.. prf:theorem:: A finite-D q4 error at every fixed source size
   :label: thm-risk-nonvertex-q4-finite-error

   Set :math:`Q=341944`, and define

   .. math::

      K_s=\frac{Q(s-2)}4,\qquad
      B_s=2s+3+\frac Q4\left(2+(s-1)(s-2)\right).

   For every pair of integers :math:`s\geq2,D\geq2(s-1)`,

   .. math::
      :label: risk-nonvertex-q4-finite-error

      \left|\frac{a_{s-1}(s+4,D)}{A_{s,4}D^{s-6}}-1\right|
      \leq \frac{29Q}{8\sqrt D}
          +\frac{K_s(1+\log D)+B_s}{D}.

   For every integer :math:`k\geq2` with
   :math:`2^k\geq2(s-1)`, the relative error at **every**
   :math:`D\geq2^k` is at most

   .. math::
      :label: risk-nonvertex-q4-dyadic-error

      \frac{29Q}{8\,2^{\lfloor k/2\rfloor}}
        +\frac{K_s(k+1)+B_s}{2^k}.

   In particular, both the edge :math:`s=2` and triangle :math:`s=3`
   have relative error below one percent at every :math:`D\geq2^{54}`.
   This is a sufficient threshold, not the smallest useful degree.
   No new growing-:math:`s,D` joint limit or q5 remainder is claimed.

A summable q4 diagonal bound
-----------------------------

The already proved single-log diagonal estimate in
:eq:`risk-refined-diagonal-explicit-error`, at :math:`q=4`, is

.. math::

   \left|\delta_4(j)-\frac{c_4}{(j+1)^2}\right|
     \leq341752\frac{1+\log(j+1)}{(j+1)^3},\qquad
   0<c_4\leq192.

These constants are obtained by substituting :math:`q=4` in
:eq:`risk-refined-vertex-error-constants`; they are not fitted to data.
Since :math:`1+\log x\leq x` for :math:`x\geq1`, positivity of the
transformed diagonals proves the global majorant

.. math::
   :label: risk-nonvertex-q4-diagonal-majorant

   0<d_j:=\delta_4(j)\leq\frac{Q}{(j+1)^2}\quad(j\geq0).

Direct rational coefficient extraction gives
:math:`d_0=1,d_1=2,d_2=37/24`. Thus
:math:`S>1+2+37/24=109/24>4`. The summability and identity
:math:`S=c_5/5` follow from
:prf:ref:`thm-risk-nonvertex-fixed-codimension-asymptotic`.

Two exact inverse-factorial moments
------------------------------------

Let :math:`0\leq L\leq D-s`, :math:`B=2D+s-L-1`,
:math:`T=L+B=2D+s-1`, and

.. math::

   \mathbb P(J=j)=\frac{\binom Lj\binom Bj}{\binom TL}.

Vandermonde's identity gives

.. math::
   :label: risk-nonvertex-q4-inverse-moments

   \mathbb E\frac1{(J+1)(J+2)}
     &=\frac{(T+1)(T+2)}{(L+1)(L+2)(B+1)(B+2)},\\
   \mathbb E\frac{L-J}{(J+1)(J+2)}
     &=\frac{L(T+1)}{(L+1)(B+1)(B+2)},\\
   \mathbb E(L-J)&=\frac{L^2}{T}.

For the second identity, use
:math:`(L-j)\binom Lj=L\binom{L-1}j` and
:math:`\binom Bj/((j+1)(j+2))=\binom{B+2}{j+2}/((B+1)(B+2))`.
At :math:`L=0` both sides are zero. The first identity uses the same
replacement without the deficit. The last is the hypergeometric mean.
Because :math:`B\geq L` and :math:`B+2\geq D+2s+1`,

.. math::

   \mathbb E\frac1{(J+1)^2}
      &\leq\frac8{(L+1)(L+2)},\\
   \mathbb E\frac{L-J}{(J+1)(J+2)}
      &\leq\frac2{D+2s+1}.

A cutoff averaging lemma
--------------------------

The following argument works for any nonnegative sequence satisfying
:eq:`risk-nonvertex-q4-diagonal-majorant`; it does not use monotonicity
or a difference bound for that sequence. On :math:`J=L` the difference
is zero, so the triangle inequality and
:math:`\mathbf1_{J<L}\leq L-J` give

.. math::

   |\mathbb E d_J-d_L|
     &\leq Q\mathbb E\frac{L-J}{(J+1)^2}
        +\frac{Q}{(L+1)^2}\mathbb E(L-J)\\
     &\leq\frac{4Q}{D+2s+1}+\frac QT
      \leq\frac{9Q}{2D}.

Alternatively, the preceding inverse-square moment and the bound for
:math:`d_L` give

.. math::

   |\mathbb E d_J-d_L|
      \leq\mathbb E d_J+d_L
      \leq\frac{10Q}{(L+1)(L+2)}.

For an arbitrary positive integer :math:`H`, use the first bound on
:math:`L<H` and sum the second on :math:`L\geq H`. Extending the latter
sum to infinity only increases its positive majorant. Even when
:math:`H>D-s+1` there are at most :math:`H` terms in the first group.
The telescoping reciprocal sum therefore yields

.. math::
   :label: risk-nonvertex-q4-cutoff-average

   \sum_{L=0}^{D-s}|\mathbb E d_J-d_L|
       \leq Q\left(\frac{9H}{2D}+\frac{10}{H+1}\right).

Choose :math:`H=\lfloor\sqrt D\rfloor`. Since
:math:`H\leq\sqrt D<H+1`, this is at most
:math:`29Q/(2\sqrt D)`.

Angular weights, tail and normalization
----------------------------------------

Write :math:`n=D-s` and

.. math::

   w_L=\prod_{r=2}^{s-1}\left(1-\frac{L+r}{D}\right),\qquad
   U_D=\sum_{L=0}^nw_L\mathbb E d_J.

For active degrees :math:`D\geq s`, all factors lie in :math:`[0,1]`.
The empty product at :math:`s=2` is one. The same finite-product
calculation used in :doc:`skeleton_risk_nonvertex_q3_error` gives

.. math::

   \sum_{L=0}^n(1-w_L)d_L
       &\leq\frac QD\left((s-2)H_{n+1}+(s-1)(s-2)\right),\\
   \sum_{j>n}d_j&\leq\frac Q{n+1}.

Together with :eq:`risk-nonvertex-q4-cutoff-average`, these bound
:math:`|U_D-S|` by

.. math::

   E_s(D)=Q\left(\frac{9H}{2D}+\frac{10}{H+1}
          +\frac1{D-s+1}
          +\frac{(s-2)H_{D-s+1}+(s-1)(s-2)}D\right).

The exact coefficient formula, without an asymptotic substitution, is

.. math::

   \frac{a_{s-1}(s+4,D)}{A_{s,4}D^{s-6}}
      =P_s(D)\frac{U_D}{S},\qquad
   P_s(D)=\prod_{i=0}^3\left(1+\frac{s+i}{2D}\right)^{-1}.

Since :math:`S>4` and :math:`0<P_s(D)\leq1`, its relative error is at
most :math:`1-P_s(D)+E_s(D)/4`. This sharper unsimplified bound is valid
at every active degree. For :math:`D\geq2(s-1)`, use
:math:`1/(D-s+1)\leq2/D`, :math:`H_{D-s+1}\leq1+\log D` and
:math:`1-P_s(D)\leq(2s+3)/D`. This proves
:eq:`risk-nonvertex-q4-finite-error`.

Both terms in that bound decrease on :math:`D\geq1`:
the logarithmic term has derivative
:math:`-(K_s\log D+B_s)/D^2<0`. Use :math:`\log2<1` and
:math:`2^{k/2}\geq2^{\lfloor k/2\rfloor}` to obtain
:eq:`risk-nonvertex-q4-dyadic-error`. Its right side decreases to zero,
so it supplies a terminating exact search for every positive rational
target. Direct rational substitution at :math:`k=54` proves the stated
one-percent sufficient threshold for :math:`s=2,3`.

Exact certificates and independent checks
-------------------------------------------

``tools.risk_research.nonvertex_q4_error`` uses ``isqrt(D)`` in the
cutoff estimate and integer bit length to majorize harmonic numbers.
Where applicable it takes the smaller of the unsimplified estimate and
the all-tail dyadic estimate. The inactive prefix :math:`D<s` is exactly
zero, with normalized relative error one. It builds no table proportional
to :math:`D`.

The positive constant interval uses the existing rational series for
:math:`C_6=540c_5`, scaled by the proved nonvertex normalization. It does
not require a special-value reduction of :math:`c_5`. ``constant_terms``
controls only that interval's truncation error. The reported finite-D
relative error concerns the exact :math:`A_{s,4}`, so those two error
budgets must remain separate.

.. code-block:: console

   $ python -m tools.risk_research.nonvertex_q4_error --face-size 2 --degree 18014398509481984 --constant-terms 64 --relative-target 1/100 --output .research/q4-edge-error.json
   $ python -m tools.risk_research.nonvertex_q4_error --verify .research/q4-edge-error.json

The saved record regenerates its complete contents, including nondefault
constant terms and target. Scalar types, unknown or changed fields,
duplicate keys and conflicting overrides are rejected. Source hashes
normalize CRLF to LF; the concurrent-edit guard compares original bytes.
The existing bounded decimal codec handles large rational and parameter
integers locally without changing process-wide conversion limits.
The new schema does not alter any existing saved-record schema.

The CLI rejects destinations resolving to the bound evaluator, dependency
or proof sources. The atomic writer receives the validated canonical
parent plus original leaf, so retargeting the original parent alias
cannot redirect its write. Output directories must be controlled by the
caller; this is not a sandbox against another process renaming the
canonical directory itself. Failures preserve any previous destination.

Tests check both inverse moments, cutoff bounds for unrelated finite
sequences, diagonal constants, all-tail behavior including odd dyadic
exponents, endpoint handling, original scalar/Gram values and post-freeze
original integer :math:`F^4` kernel holdouts. Optional native checks use
the original positive integral separately for the leading constant.
Finite checks validate these implementations and normalizations; the
displayed inequalities prove the infinite degree tail.
