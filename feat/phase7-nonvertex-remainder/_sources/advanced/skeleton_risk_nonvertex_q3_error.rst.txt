Finite-degree errors for every nonvertex q3 face
================================================

Keep the uniform-simplex design, evaluation and noise assumptions of
:doc:`skeleton_risk_theory`. Fix an integer :math:`s\geq2` and put
:math:`M=s+3`. The positive diagonal construction in
:doc:`skeleton_risk_codimension_three_exact` already proves

.. math::

   a_{s-1}(s+3,D)\sim A_s D^{s-5},\qquad
   A_s=\frac{s(s+1)^3(s+2)^3(s+3)^2}{24(s-2)!}
          (\zeta(2)-\zeta(3)),

and in particular

.. math::

   A_2=3600(\zeta(2)-\zeta(3)),\quad
   A_3=36000(\zeta(2)-\zeta(3)).

The following adds an explicit error valid beyond a stated degree. It
concerns sample-size-asymptotic risk coefficients, not finite-sample risk.

.. prf:theorem:: Finite-D relative error at every q3 source-face size
   :label: thm-risk-all-nonvertex-q3-finite-error

   Put :math:`K_s=(25s+358)/4` and
   :math:`B_s=(25s^2-69s+106)/4`. For every integer
   :math:`s\geq2,D\geq2(s-1)`,

   .. math::

      \left|\frac{a_{s-1}(s+3,D)}{A_sD^{s-5}}-1\right|
       \leq\frac{K_s(1+\log D)+B_s}{D}.

   For integers :math:`k\geq2` with :math:`2^k\geq2(s-1)`, the error
   is at most :math:`(K_s(k+1)+B_s)/2^k` at every :math:`D\geq2^k`.
   Thus each fixed source-face size has a certified
   :math:`O(\log D/D)` remainder. The displayed inequality explicitly
   records its dependence on :math:`s`; it does not assert a new joint
   asymptotic for growing :math:`s,D` or cover codimension four and above.

.. prf:theorem:: Finite-D relative error for the q3 edge and triangle
   :label: thm-risk-nonvertex-q3-finite-error

   For both :math:`s=2,3` and every integer :math:`D\geq4`,

   .. math::
      :label: risk-nonvertex-q3-finite-error

      \left|\frac{a_{s-1}(s+3,D)}{A_sD^{s-5}}-1\right|
       \leq\frac{109(1+\log D)+31}{D}.

   For every integer :math:`k\geq2`, the relative error is therefore at
   most :math:`(109(k+1)+31)/2^k` at **every** integer :math:`D\geq2^k`.
   In particular :math:`D\geq262144` suffices for relative error below
   one percent for both families. These are sufficient, not optimal,
   thresholds. This is a specialization of the preceding theorem.

The constants below are analytic majorants. They are not fitted to
computed risks or inferred from a finite degree scan.

Uniform bounds and one discrete derivative
-------------------------------------------

Write :math:`d_j=\delta_3(j)`, and use the harmonic expression in
:eq:`risk-nonvertex-q3-harmonic-diagonal`. For every integer :math:`j\geq0`,

.. math::
   :label: risk-nonvertex-q3-diagonal-lipschitz

   0<d_j\leq\frac{25}{(j+1)^2},\qquad
   |d_{j+1}-d_j|\leq\frac{102}{(j+1)(j+2)(j+3)}.

Here are bounds including :math:`j=0`, so no hidden small-index exception
is needed. Put :math:`f_j=j/(j+2)`, :math:`N_j=H_j^2-f_jH_j`, and
:math:`Q_j=(j+1)(j+2)(j+3)`. Cauchy--Schwarz and
:math:`\sum_{r\geq1}r^{-2}\leq2` give
:math:`0\leq N_j\leq H_j^2\leq2j`. Also :math:`H_j\leq j` and
:math:`f_{j+1}-f_j=2/((j+2)(j+3))`. Expanding the numerator difference
gives :math:`|N_{j+1}-N_j|\leq5`.

Split :math:`d_j/6=A_j+B_j+C_j` into the three displayed harmonic
terms. Since :math:`Q_j^{-1}-Q_{j+1}^{-1}=3/(Q_j(j+4))`,

.. math::

   |A_{j+1}-A_j|&\leq(5+6j/(j+4))/Q_j\leq11/Q_j,\\
   |B_{j+1}-B_j|&\leq
     \frac{1}{(j+1)^2(j+2)(j+4)}
     +\frac{2(2j+5)}{(j+1)(j+2)(j+3)(j+4)}
       \leq5/Q_j,\\
   |C_{j+1}-C_j|&\leq C_j\leq1/Q_j.

The last term decreases because all its positive denominator factors
increase. This proves the difference bound with :math:`6(11+5+1)=102`.
The three terms themselves are at most
:math:`2/(j+1)^2`, :math:`2/(j+1)^2`, and
:math:`1/(6(j+1)^2)`, respectively, proving the bound with 25.

An exact inverse-factorial deficit moment
------------------------------------------

Let :math:`0\leq L\leq D-s`, :math:`B=2D+s-L-1`,
:math:`T=L+B=2D+s-1`, and

.. math::

   \mathbb P(J=j)=\frac{\binom Lj\binom Bj}{\binom TL}.

For :math:`L\geq1`, replace
:math:`(L-j)\binom Lj=L\binom{L-1}j` and
:math:`\binom Bj/(j+1)_3=\binom{B+3}{j+3}/(B+1)_3`.
Vandermonde then proves the exact identity

.. math::
   :label: risk-nonvertex-q3-deficit-moment

   \mathbb E\frac{L-J}{(J+1)(J+2)(J+3)}
    =\frac{L(T+1)(T+2)}
       {(L+1)(L+2)(B+1)(B+2)(B+3)}.

At :math:`L=0` both sides are zero. Since :math:`B\geq L` and
:math:`B+3\geq D+2s+2`, the right side is at most
:math:`4L/((L+1)(L+2)(D+2s+2))`.
Telescoping the discrete derivative gives

.. math::

   |d_J-d_L|\leq
    \frac{102(L-J)}{(J+1)(J+2)(J+3)}.

Consequently, with :math:`n=D-s`,

.. math::
   :label: risk-nonvertex-q3-average-error

   \sum_{L=0}^n|\mathbb E d_J-d_L|
    \leq\frac{408}{D+2s+2}
       \left(H_{n+1}+\frac{2}{n+2}-2\right)
    \leq\frac{408H_{n+1}}{D+2s+2}.

The exact sum here uses
:math:`L/((L+1)(L+2))=-1/(L+1)+2/(L+2)`.

Angular weights, omitted tail, and normalization
------------------------------------------------

Put :math:`S=\sum_{j\geq0}d_j=12(\zeta(2)-\zeta(3))`.
It is finite by the positive diagonal theorem. Direct rational arithmetic
in the harmonic expression gives
:math:`\sum_{j=0}^7d_j=8568671/2116800>4`, hence :math:`S>4`.
The positive-square bound and the integral test give

.. math::

   \sum_{j>n}d_j\leq25/(n+1).

Normalize the angular weights by their limiting factor
:math:`D^{s-2}/(s-2)!`. The resulting reindexed weight is

.. math::

   w_L=\prod_{r=2}^{s-1}\left(1-\frac{L+r}{D}\right)\in[0,1],
   \qquad
   1-w_L\leq\frac{(s-2)L+s(s-1)/2-1}{D}.

The empty product is one at :math:`s=2`; at :math:`s=3` the weight is
:math:`(D-L-2)/D`. Every factor lies in :math:`[0,1]` because
:math:`L\leq D-s` and :math:`r\leq s-1`. The deficit inequality is the
finite product inequality :math:`1-\prod(1-x_r)\leq\sum x_r`.
Since
:math:`(s-2)L+s(s-1)/2-1=(s-2)(L+1)+(s-1)(s-2)/2`,

.. math::

   \sum_{L=0}^n(1-w_L)d_L
     \leq\frac{25}{D}\left((s-2)H_{n+1}+(s-1)(s-2)\right).

Here the finite inverse-square sum is at most two. Define
:math:`U_D=\sum_{L=0}^nw_L\mathbb E d_J`. The triangle inequality
and :eq:`risk-nonvertex-q3-average-error` show

.. math::

   |U_D-S|\leq E_s(D):=
      \frac{408H_{D-s+1}}{D+2s+2}+\frac{25}{D-s+1}
      +\frac{25((s-2)H_{D-s+1}+(s-1)(s-2))}D.

The exact positive coefficient formula is

.. math::

   \frac{a_{s-1}(s+3,D)}{A_sD^{s-5}}
       =P_s(D)\frac{U_D}{S},\qquad
   P_s(D)=\prod_{i=0}^2\left(1+\frac{s+i}{2D}\right)^{-1}.

Thus its relative error is at most :math:`1-P_s(D)+E_s(D)/4`.
For :math:`D\geq2(s-1)`, use :math:`H_{D-s+1}\leq1+\log D`,
:math:`1/(D-s+1)\leq2/D`, and
:math:`1-P_s(D)\leq3(s+1)/(2D)`.
These give precisely the stated :math:`K_s,B_s`, proving the general
theorem. For :math:`s=2` the resulting bound is
:math:`(102(1+\log D)+17)/D`; for :math:`s=3` it is
:math:`((433/4)(1+\log D)+31)/D`.
Both are bounded by :eq:`risk-nonvertex-q3-finite-error`.
Finally :math:`(109(1+\log x)+31)/x` decreases on :math:`x\geq1`,
and :math:`\log2<1`, proving the dyadic all-tail statement.

Exact rational certification and independent checks
----------------------------------------------------

``tools.risk_research.nonvertex_q3_error`` supports every integer
:math:`s\geq2` and evaluates these bounds without
generating the double sum. It uses the sharper harmonic-sum expression
in :eq:`risk-nonvertex-q3-average-error`, replacing
:math:`H_m` by :math:`1+\lceil\log_2m\rceil`, evaluated by integer
bit length. It also applies the dyadic bound at the largest power of two
below the requested degree when it is at least :math:`2(s-1)`.
The smaller of these two valid bounds is
still a rigorous bound. The inactive prefix :math:`D<s` is exactly zero;
its normalized relative error is one. Active degrees below the dyadic domain use
the unsimplified bound above.

For the leading constant the summand :math:`(n-1)/n^3` is positive and
decreasing on :math:`n\geq2`. The tail after :math:`n=N` lies between
the integrals from :math:`N+1` and from :math:`N`, whose values are
:math:`1/t-1/(2t^2)`. Rational interval multiplication with
:math:`\max(0,1-r)` and :math:`1+r` then encloses the finite-D
coefficient. The reported ``relative_error_bound`` concerns deviation
from the exact leading term; ``constant_terms`` separately controls the
uncertainty of the leading constant's displayed interval.

.. code-block:: console

   $ .venv/bin/python -m tools.risk_research.nonvertex_q3_error --face-size 2 --degree 262144 --constant-terms 64 --relative-target 1/100 --output .research/skeleton-risk/q3-edge-error.json
   $ .venv/bin/python -m tools.risk_research.nonvertex_q3_error --verify .research/skeleton-risk/q3-edge-error.json

The target selects an all-tail threshold, not an exact crossover or the
smallest degree at which an individual coefficient is accurate. Saved
records bind the source and proof files; verification reconstructs the
complete record using its own saved options, including nondefault terms
and target. Verification can optionally write an identical checked record
with ``--output``; conflicting generation overrides are rejected. Scalar JSON
types and duplicate keys are checked. Hashes normalize CRLF to LF;
the before/after source guard compares raw bytes. Failed computation or
verification preserves any previous destination through atomic output.

The tests independently check the deficit moment, diagonal majorants,
normalization, endpoint behavior, original scalar/Gram coefficients and
held-out integer-kernel values. Optional native FLINT/Arb checks evaluate
the original positive double sum and known zeta constant separately.
These finite checks validate the implementation; the displayed
inequalities prove the infinite degree tail.

The CLI rejects outputs resolving to evaluator or proof sources, including
symlink aliases, before calculation and immediately before replacement.
Large rational integers and JSON integer parameters use local bounded
decimal chunks; parameter fields remain JSON numbers. Replay uses the
same local decoder, and no process-wide integer conversion limit is changed.
Ordinary decimal or rational ``--relative-target`` syntax remains supported.
