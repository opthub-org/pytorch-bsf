A logarithmic finite-degree remainder for nonvertex q4 risks
============================================================

Keep the uniform-simplex design, evaluation and noise assumptions of
:doc:`skeleton_risk_theory`. For each fixed integer :math:`s\geq2`, set
:math:`M=s+4` and use the existing positive leading constant

.. math::

   A_{s,4}=\binom{s+4}s^2\frac{(s)_4}{16(s-2)!}S,
   \qquad S=\sum_{j\geq0}\delta_4(j)=\frac{c_5}{5}>4.

The previous cutoff argument gives :math:`O(D^{-1/2})` in
:doc:`skeleton_risk_nonvertex_q4_error`. Here a new, explicit q4 discrete
derivative proves :math:`O(\log D/D)`. No q3 difference estimate is
assumed for q4. The coefficients concern sample-size-asymptotic risk;
they do not supply finite-sample risk guarantees.

.. prf:theorem:: A logarithmic q4 remainder at every fixed source size
   :label: thm-risk-nonvertex-q4-logarithmic-error

   Put :math:`Q=5400`, :math:`V=179568`, and

   .. math::

      K_s=V+\frac{Q(s-2)}4,\qquad
      B_s=2s+3+\frac Q4\left(2+(s-1)(s-2)\right).

   For every pair of integers :math:`s\geq2,D\geq2(s-1)`,

   .. math::
      :label: risk-nonvertex-q4-logarithmic-error

      \left|\frac{a_{s-1}(s+4,D)}{A_{s,4}D^{s-6}}-1\right|
       \leq\frac{K_s(1+\log D)+B_s}{D}.

   For every integer :math:`k\geq1` with :math:`2^k\geq2(s-1)`, the
   relative error at every :math:`D\geq2^k` is at most

   .. math::
      :label: risk-nonvertex-q4-refined-dyadic

      \frac{K_s(k+1)+B_s}{2^k}.

   Both :math:`s=2` and :math:`s=3` therefore have relative error below
   one percent for every :math:`D\geq2^{30}`. The previous sufficient
   threshold was :math:`2^{54}`. Neither threshold is claimed to be the
   smallest useful degree. No growing-:math:`s,D` joint limit or q5
   remainder is asserted.

A convolution estimate including the first index
------------------------------------------------

Write :math:`\Delta a_j=a_{j+1}-a_j`. Suppose two sequences satisfy

.. math::

   |a_j|\leq\frac A{(j+1)^2},\quad
   |\Delta a_j|\leq\frac U{(j+1)^3},\qquad
   |b_j|\leq\frac B{(j+1)^2},\quad
   |\Delta b_j|\leq\frac W{(j+1)^3}.

No positivity, monotonicity or zero total sum is required. For ordinary
convolution :math:`c_n=\sum_{j=0}^na_jb_{n-j}`,

.. math::
   :label: risk-nonvertex-q4-convolution-norms

   |c_n|\leq\frac{16AB}{(n+1)^2},\qquad
   |\Delta c_n|\leq\frac{16(AW+BU+AB)}{(n+1)^3}.

Indeed, :math:`\sum_{j\geq0}(j+1)^{-2}\leq2`. Splitting the convolution
at :math:`h=\lfloor n/2\rfloor` leaves a denominator at least
:math:`(n+1)^2/4` on each half, proving the first bound. An exact
split of the difference, using the same :math:`h` in both sums, is

.. math::

   \Delta c_n={}&\sum_{j=0}^ha_j\Delta b_{n-j}
       +\sum_{l=0}^{n-h-1}b_l\Delta a_{n-l}
       +a_{h+1}b_{n-h}.

The two derivative denominators are at least :math:`(n+1)^3/8`.
The last product is at most :math:`16AB/(n+1)^4`, hence also at most
:math:`16AB/(n+1)^3`. This proves the second bound, including
:math:`n=0`, where the second sum is empty.

Explicit coefficient norms for q4
---------------------------------

Use the sequences :math:`\phi_r,b_r,e_r` defined in
:doc:`skeleton_risk_vertex_general`. Put :math:`x=j+1` and let
:math:`E_l^{(r)}(j)` denote the elementary symmetric polynomial of
order :math:`l` in the signed reciprocal list in
:eq:`risk-general-phi-elementary`. With

.. math::

   p_r(j)=\frac{r!(r-1)!j!}{(j+r)!},

adding :math:`-1/(j+1)` to that list gives the exact recurrence

.. math::

   \Delta\phi_r(j)
     =-\frac{r\phi_r(j)+p_r(j)E_{r-2}^{(r)}(j)}{j+r+1}
       \quad(r\geq2).

The absolute reciprocal sum is at most :math:`2+\log x` for
:math:`r=2` and at most :math:`3+\log x` for :math:`r=3,4`.
Together with :math:`|E_l|\leq(\sum |u_i|)^l/l!`, this proves

.. math::

   |\phi_2(j)|&\leq\frac{2(2+\log x)}{x^2},&
   |\Delta\phi_2(j)|&\leq\frac{10+4\log x}{x^3},\\
   |\phi_3(j)|&\leq\frac{6(3+\log x)^2}{x^3},&
   |\Delta\phi_3(j)|&\leq
        \frac{18(3+\log x)^2+12(3+\log x)}{x^4},\\
   |\phi_4(j)|&\leq\frac{24(3+\log x)^3}{x^4},&
   |\Delta\phi_4(j)|&\leq
        \frac{96(3+\log x)^3+72(3+\log x)^2}{x^5}.

For :math:`x\geq1`, the elementary functions used on these right sides
decrease after multiplication by the powers of :math:`x` indicated
below. For example, :math:`(a+\log x)^b/x^c` decreases whenever
:math:`b/a\leq c`. Since :math:`\phi_1(j)=1/(j+1)`, it follows that

.. math::

   |\phi_1(j)|&\leq1/x,& |\Delta\phi_1(j)|&\leq1/x^2,\\
   |\phi_2(j)|&\leq4/x,& |\Delta\phi_2(j)|&\leq10/x^2,\\
   |\phi_3(j)|&\leq54/x,& |\Delta\phi_3(j)|&\leq198/x^2,\\
   |\phi_4(j)|&\leq648/x^2,& |\Delta\phi_4(j)|&\leq3240/x^3.

The last line retains stronger decay for the solitary
:math:`\phi_4` term. The exact identities

.. math::

   e_2=2\phi_1^2,\qquad e_3=6\phi_1\phi_2,\qquad
   e_4=8\phi_1\phi_3+6\phi_2^2

and the discrete product rule give the following absolute norms.

.. list-table:: Bounds valid at every integer j at least zero
   :header-rows: 1

   * - Sequence a
     - A in :math:`|a_j|\leq A/x^2`
     - U in :math:`|\Delta a_j|\leq U/x^3`
   * - :math:`e_2`
     - 2
     - 4
   * - :math:`e_3`
     - 24
     - 84
   * - :math:`e_4`
     - 528
     - 2496
   * - :math:`b_1`
     - 2
     - 3
   * - :math:`b_2`
     - 6
     - 12

For the last two rows, :math:`b_1(0)=b_2(0)=1` and, at :math:`j\geq1`,

.. math::

   b_1(j)&=-\frac1{j(j+1)},&
   \Delta b_1(j)&=\frac2{j(j+1)(j+2)},\\
   b_2(j)&=\frac{2(2H_{j-1}-3)}{j(j+1)(j+2)},&
   \Delta b_2(j)&=\frac{2(11-6H_{j-1})}{j(j+1)(j+2)(j+3)}.

At :math:`j=0`, their differences are :math:`-3/2` and :math:`-2`.
For positive indices use :math:`H_{j-1}\leq j-1` and
:math:`(j+1)^2\leq j(j+3)` for the :math:`b_2` row. For :math:`b_1`,
:math:`2(j+1)^2/(j(j+2))\leq8/3<3`. Thus the table has no omitted
small-index exception. These are absolute bounds: the individual
:math:`e_r` terms need not be nonnegative.

The general transformed-diagonal identity specializes exactly to

.. math::

   \delta_4=-\phi_4+e_4-4(b_1*e_3)+6(b_2*e_2).

Applying :eq:`risk-nonvertex-q4-convolution-norms` and the table gives

.. math::

   |\delta_4(j)|&\leq
     \frac{648+528+4(768)+6(192)}{(j+1)^2}
       =\frac{5400}{(j+1)^2},\\
   |\Delta\delta_4(j)|&\leq
     \frac{3240+2496+4(4608)+6(960)}{(j+1)^3}
       =\frac{29928}{(j+1)^3}.

The existing positive Gram-diagonal theorem supplies strict positivity.
Since :math:`(j+2)(j+3)\leq6(j+1)^2`, the second estimate also proves

.. math::
   :label: risk-nonvertex-q4-refined-diagonal

   0<d_j:=\delta_4(j)\leq\frac Q{(j+1)^2},\qquad
   |d_{j+1}-d_j|\leq\frac V{(j+1)(j+2)(j+3)}.

The constants :math:`Q,V` therefore come from explicit coefficient
inequalities, not from a fit or a finite scan.

Averaging, angular weights and the infinite tail
------------------------------------------------

Use the exact positive-diagonal averages from
:doc:`skeleton_risk_codimension_three_exact`. Set :math:`n=D-s`,
:math:`0\leq L\leq n`, :math:`B=2D+s-L-1`, :math:`T=L+B`, and

.. math::

   \mathbb P(J=j)=\frac{\binom Lj\binom Bj}{\binom TL}.

The inverse-factorial deficit identity is a distributional identity,
independent of the q3 or q4 sequence. Vandermonde's identity gives

.. math::

   \mathbb E\frac{L-J}{(J+1)(J+2)(J+3)}
    =\frac{L(T+1)(T+2)}
      {(L+1)(L+2)(B+1)(B+2)(B+3)}
    \leq\frac{4L}{(L+1)(L+2)(D+2s+2)}.

Indeed, replace :math:`(L-j)\binom Lj=L\binom{L-1}j` and
:math:`\binom Bj/(j+1)_3=\binom{B+3}{j+3}/(B+1)_3`, then apply
Vandermonde. At :math:`L=0` both sides vanish. Since :math:`B\geq L`,
the remaining upper bound uses :math:`T+1\leq2(B+1)`,
:math:`T+2\leq2(B+2)`, and :math:`B+3\geq D+2s+2`.

Telescoping the newly proved q4 derivative and summing over :math:`L`
therefore bounds the averaging error by

.. math::

   \sum_{L=0}^n|\mathbb E d_J-d_L|
     &\leq \frac{4V}{D+2s+2}
         \left(H_{n+1}+\frac2{n+2}-2\right)\\
     &\leq\frac{4VH_{n+1}}{D+2s+2}.

For active degrees :math:`D\geq s`, put

.. math::

   w_L=\prod_{r=2}^{s-1}\left(1-\frac{L+r}{D}\right),\qquad
   U_D=\sum_{L=0}^nw_L\mathbb E d_J,\qquad
   P_s(D)=\prod_{i=0}^3\left(1+\frac{s+i}{2D}\right)^{-1}.

The exact normalization is
:math:`a_{s-1}(s+4,D)/(A_{s,4}D^{s-6})=P_s(D)U_D/S`.
The product factors :math:`w_L` lie in :math:`[0,1]`, including the
empty product at :math:`s=2`. The same finite-product calculation as
in :doc:`skeleton_risk_nonvertex_q4_error`, now with the proved smaller
:math:`Q`, yields

.. math::

   \sum_{L=0}^n(1-w_L)d_L
     &\leq\frac QD\bigl((s-2)H_{n+1}+(s-1)(s-2)\bigr),\\
   \sum_{j>n}d_j&\leq\frac Q{n+1}.

The exact initial values :math:`d_0=1,d_1=2,d_2=37/24` imply
:math:`S>109/24>4`. Hence the relative error is at most

.. math::
   :label: risk-nonvertex-q4-refined-unsimplified

   1-P_s(D)+\frac14\left[
      \frac{4V}{D+2s+2}\left(H_{n+1}+\frac2{n+2}-2\right)
      +\frac Q{n+1}
      +\frac QD\bigl((s-2)H_{n+1}+(s-1)(s-2)\bigr)\right].

At :math:`D\geq2(s-1)`, use :math:`(n+1)^{-1}\leq2/D`,
:math:`H_{n+1}\leq1+\log D`, and
:math:`1-P_s(D)\leq(2s+3)/D`. This proves
:eq:`risk-nonvertex-q4-logarithmic-error`. Its right side decreases on
:math:`D\geq1`, since its derivative is
:math:`-(K_s\log D+B_s)/D^2<0`. Together with :math:`\log2<1`, this
proves :eq:`risk-nonvertex-q4-refined-dyadic`. Its right side decreases
to zero, giving a terminating rational threshold search for every
positive rational target. At :math:`s=2,3,k=30`, the numerator
:math:`K_s(k+1)+B_s` is respectively :math:`5569315` and :math:`5613867`,
both strictly below :math:`2^{30}/100`.

Certificates, scope and independent validation
----------------------------------------------

``tools.risk_research.nonvertex_q4_refined`` evaluates the displayed
bounds by exact fractions. It uses integer bit lengths to majorize
harmonic numbers, and the smaller of
:eq:`risk-nonvertex-q4-refined-unsimplified` and the available dyadic
bound. It constructs no degree-sized risk or harmonic table. On the
inactive prefix :math:`D<s`, the coefficient is exactly zero and its
normalized relative error is one.

The leading-constant interval still uses the independent existing
rational series for :math:`C_6=540c_5`. ``constant_terms`` changes that
interval's truncation width, while the new finite-D bound concerns the
exact :math:`A_{s,4}`. The two uncertainties remain separate. No finite
special-value reduction or new growing-source asymptotic is needed.

.. code-block:: console

   $ python -m tools.risk_research.nonvertex_q4_refined --face-size 2 --degree 1073741824 --constant-terms 64 --relative-target 1/100 --output .research/q4-refined-edge.json
   $ python -m tools.risk_research.nonvertex_q4_refined --verify .research/q4-refined-edge.json

The new source-bound schema leaves existing saved-record schemas
unchanged. Reverification restores the saved nondefault parameters and
regenerates the complete record, rejecting changed types, duplicate or
unknown fields, modified proof constants and conflicting overrides.
Hashes normalize CRLF to LF; the concurrent-edit guard compares original
bytes. Large decimal integers use the existing local chunked codec,
without changing process-wide limits.

The CLI rejects destinations resolving to a bound evaluator, dependency
or proof source. Its atomic writer uses the validated canonical parent
and original leaf. The directory must be controlled by the caller;
this is not a sandbox against concurrent replacement of the canonical
directory inode itself. Failures preserve previous output contents.

Finite checks cover the exact coefficient and convolution identities,
small-index norms, inverse-factorial moments, scalar/Gram coefficients,
frozen original integer F4-kernel holdouts, saved options and failure
preservation. Optional native checks independently integrate the original
positive leading-constant integral. These finite checks validate code
and normalization; the inequalities above prove the entire degree tail.
