Further evaluations of the skeleton risk
==========================================

This chapter continues :doc:`skeleton_risk_theory` under exactly its
uniform-simplex design, evaluation and noise assumptions. Here ``M`` is
the number of vertices and ``D`` is the degree. It separates proved
identities from finite computational checks and remaining evaluations.
The optional reference implementations live in ``tools/risk_research``.
They add no runtime dependencies. An accompanying numerical guard rejects
nonfinite assembled-risk propagation before its condition estimate.

.. list-table:: Results and limitations
   :header-rows: 1

   * - Result
     - Scope
   * - :prf:ref:`thm-risk-fixed-dimension-holonomic`
     - Every fixed ``M,m`` admits a polynomial-coefficient degree recurrence;
       this existence statement does not supply a small operator uniformly in ``M``.
   * - :prf:ref:`lem-risk-two-coordinate-kernel`
     - The entire two-coordinate kernel is evaluated with harmonic numbers.
   * - :prf:ref:`thm-risk-nonvertex-boundary-transfer`
     - Every nonvertex source has a trace-correct boundary-transfer inequality.
       In particular the five-vertex edge has an all-degree rational upper bound;
       this does not provide an exact codimension-three formula.
   * - :prf:ref:`thm-risk-codimension-two-modes`
     - Every codimension-two coefficient reduces to one angular sum.
   * - :prf:ref:`thm-risk-tetrahedron-edge-representation`
     - Every tetrahedron edge coefficient is evaluated using harmonic numbers
       and two explicitly specified auxiliary recurrences.
   * - :prf:ref:`thm-risk-five-vertex-triangle-state`
     - Every triangular-face coefficient on a five-vertex simplex has an
       eight-state rational representation and a certified order-eight recurrence.
   * - :prf:ref:`thm-risk-six-vertex-tetrahedral-state`
     - Every tetrahedral-face coefficient on a six-vertex simplex also has
       an eight-state formula and a certified order-eight recurrence.
   * - :prf:ref:`thm-risk-tetrahedron-vertex-recurrence`
     - The tetrahedron vertex has an explicit order-ten degree recurrence
       certified by four rational telescoping identities.
   * - :prf:ref:`cor-risk-codimension-two-asymptotic`
     - The leading large-degree constant for every fixed codimension-two face
       with at least two vertices.
   * - :prf:ref:`thm-risk-vertex-boundary-pair`
     - A vertex norm identity reduces the largest outside-kernel power by two;
       it is not a completed tetrahedron vertex closed form.

Fixed-dimension degree recurrences exist
----------------------------------------

.. prf:theorem:: Holonomicity at fixed dimension
   :label: thm-risk-fixed-dimension-holonomic

   Fix integers :math:`M\geq2` and :math:`0\leq m<M`. There exist an
   integer :math:`r\geq0` and polynomials
   :math:`p_0,\ldots,p_r\in\mathbb Q[D]`, not all zero, such that

   .. math::

      \sum_{j=0}^r p_j(D)a_m(M,D+j)=0

   for all sufficiently large integer :math:`D`. A polynomial factor can
   include any exceptional finite prefix. The order and degrees may depend
   on :math:`M,m`.

.. prf:proof::

   Put :math:`s=m+1`, :math:`q=M-s`. The explicit logarithmic expression
   for :math:`F(x,y)` in :prf:ref:`lem-skeleton-kernel` is multivariate
   differential D-finite. For fixed nonnegative integer :math:`q`, the
   derivatives of :math:`F^q` lie in the rational-function span of monomials
   of total degree at most :math:`q` in
   :math:`\log(1-x-y),\log(1-x),\log(1-y)`. This is a finite-dimensional
   span, including the constant function. Thus :math:`F^q` is D-finite.

   The coefficient dictionary from holonomic power series to holonomic
   discrete sequences makes :math:`K_q(h,l)=[x^hy^l]F^q` holonomic.
   This uses differential D-finiteness of the generating series, rather
   than a claim about arbitrary multivariate shift-D-finite sequences.
   The distinction and coefficient dictionary are discussed in
   :cite:t:`kauers2023dfinite`, Section 4.5, printed page 368.

   In :prf:ref:`thm-skeleton-scalar-sum`, all remaining factors are
   factorial ratios of affine integer expressions, signs and polynomial
   factors, hence holonomic on the summation domain. Numerator factorials
   can be extended by zero to negative arguments and their boundary
   recurrences cleared; denominator factorials use reciprocal-factorial
   zero extensions. The case :math:`s=1` uses the holonomic multiplier
   :math:`\mathbf1_{k=0}`. Multiply by the indicators of

   .. math::

      k,h,l\geq0,\qquad k+h\leq D-s,\qquad k+l\leq D-s.

   All factorial arguments in the original summand are nonnegative on
   this domain. Holonomic sequences are closed under pointwise products
   and finite summation over such rational polyhedral ranges
   (:cite:t:`kauers2023dfinite`, Theorems 5.24 and 5.33 and printed
   pages 442--443). Consequently the scalar sum is holonomic in :math:`D`.
   Its remaining prefactor is hypergeometric in :math:`D`. The inactive
   degrees form only a finite prefix, which preserves P-recursiveness.

Finite products of such fixed-dimension sequences and finite weighted
sums preserve P-recursiveness. Therefore the product-cell formulas and
regular-complex incidence formulas inherit the conclusion when their
combinatorics are fixed and their factor degrees follow fixed affine
integer functions of ``D`` that remain positive for all sufficiently large
``D``. This statement concerns risk coefficients;
it does not assert closure for the square roots in optimal allocation.

Two outside coordinates
------------------------

Write :math:`H_n=\sum_{j=1}^n1/j`, with :math:`H_0=0`.

.. prf:lemma:: Harmonic evaluation of the two-coordinate kernel
   :label: lem-risk-two-coordinate-kernel

   For all nonnegative integers :math:`h,l`,

   .. math::
      :label: risk-two-coordinate-kernel

      K_2(h,l)={}&
      \frac{2(h+l+1)!}{(h+2)!(l+2)!}
      \left((h+l+2)H_{h+l+1}-1-hH_h-lH_l\right)\\
      &+\frac{2}{(h+1)(h+2)(l+1)(l+2)}.

.. prf:proof::

   Put :math:`U=(1-x-y)\log(1-x-y)`,
   :math:`A=(1-x)\log(1-x)`, :math:`B=(1-y)\log(1-y)`.
   Then :math:`x^2y^2F^2=U^2-2UA-2UB+2AB+A^2+B^2`.
   Set :math:`a=h+2`, :math:`b=l+2`, :math:`n=a+b`, and
   :math:`C=(n-3)!/(a!b!)`. The last two terms have zero mixed coefficient,
   and :math:`[x^ay^b]AB=1/(a(a-1)b(b-1))`.

   Differentiating the finite binomial products in

   .. math::

      [x^ay^b](1-x-y)^\alpha(1-x)^\beta
      =(-1)^{a+b}\binom\alpha b\binom{\alpha+\beta-b}{a}

   once in each parameter at :math:`\alpha=\beta=1` gives

   .. math::

      [x^ay^b]UA=-C\left(1+(b-2)(H_{n-3}-H_{b-2})\right).

   For :math:`b=2` this follows directly from the derivative at zero;
   for :math:`b\geq3`, use the logarithmic derivative of the binomial
   product at :math:`2-b`. The displayed expression includes both cases
   without a negative-index harmonic number. Interchange :math:`a,b`
   for :math:`UB`. Twice differentiating
   :math:`(-1)^n\binom\alpha n` at :math:`\alpha=2` gives
   :math:`[x^ay^b]U^2=4C(H_{n-3}-3/2)`. Combining the terms proves the formula.

A codimension-two face
-----------------------

.. prf:theorem:: Harmonic norm of every codimension-two mode
   :label: thm-risk-codimension-two-modes

   Let :math:`M=s+2`, :math:`s\geq1`, :math:`D\geq s`,
   :math:`0\leq k\leq D-s`. Put

   .. math::

      L=D-s-k,\quad \beta=3s+2k-1,\quad N=2L+\beta+2=2D+s+1,\\
      \Delta=(L+1)(L+2)(L+\beta+1)(L+\beta+2),\\
      U=(L+1)(L+\beta+1)-1,\quad c=(N+1)U/\Delta,
      \quad W=\binom NL.

   The full-simplex squared norm of a propagated angular mode, divided
   by its source-face squared norm, equals

   .. math::
      :label: risk-codimension-two-mode

      \rho_{s,k,D}=\frac{2s(s+1)}\Delta
      \left(H_{L+2}+H_{L+\beta+2}-H_N-c
       -\frac1{(L+\beta+1)(L+\beta+2)W}\right).

   With :math:`d_s(k)` as in :prf:ref:`thm-skeleton-scalar-sum`,

   .. math::
      :label: risk-codimension-two-coefficient

      a_{s-1}(s+2,D)=\binom{s+2}s^2
      \sum_{k=0}^{D-s}d_s(k)\rho_{s,k,D}.

   If :math:`D<s`, the coefficient is zero.

.. prf:proof::

   Let :math:`p` be a propagated mode, with outside coordinates :math:`y,z`.
   Let :math:`g` be its codimension-one extension with the outside
   coordinate replaced by :math:`y+z`. On each facet the traces agree;
   thus :math:`p-g` is a full-simplex bubble. Propagation orthogonality
   gives :math:`\langle p,p\rangle=\langle p,g\rangle`.

   The common angular factor cancels. The radial factor of :math:`g` is
   :math:`P_L^{(1,\beta)}(2r-1)/(L+1)`, while the conditional average of
   :math:`p` has coefficients
   :math:`A_h=(-1)^h\binom Lh\binom{L+\beta}h` multiplied by
   :math:`2H_{h+1}/((h+1)(h+2))`. The latter is the uniform outside-edge
   moment of :math:`T_h` in the scalar-sum proof.

   For

   .. math::

      I_h=\frac1{L+1}\int_0^1 r^\beta(1-r)
      P_L^{(1,\beta)}(2r-1)r^{L-h}(1-r)^h\,dr,

   Jacobi orthogonality gives :math:`I_h=(-1)^hI_0`.
   Rodrigues' formula and :math:`L` integrations by parts give
   :math:`I_0=L!(L+\beta)!/N!`; all boundary terms vanish.
   The beta density of :math:`r` supplies :math:`s(s+1)`. Hence

   .. math::

      \rho=2s(s+1)I_0\sum_{h=0}^L
      \binom Lh\binom{L+\beta}h
      \frac{H_{h+1}}{(h+1)(h+2)}.

   Set :math:`w_h=\binom{L+2}{h+2}\binom{L+\beta}h`.
   Vandermonde gives :math:`\sum w_h=W`; differentiating its first
   upper parameter gives
   :math:`\sum w_hH_{h+2}=W(H_{L+2}+H_{L+\beta+2}-H_N)`.
   The rational certificate

   .. math::

      R_h=\frac{(h+2)(Uh-1)}{\Delta(h+1)},\qquad
      w_h\left(\frac1{h+2}-c\right)
       =w_{h+1}R_{h+1}-w_hR_h

   includes both endpoints: :math:`w_{L+1}=0` and
   :math:`w_0R_0=-1/((L+\beta+1)(L+\beta+2))`.
   Subtract the telescoped sum from the harmonic Vandermonde identity.
   Finally use :math:`I_0W=1/((L+\beta+1)(L+\beta+2))` and the
   factor :math:`1/((L+1)(L+2))` from rewriting the binomial coefficient.

Tetrahedron edges at every degree
----------------------------------

The remaining angular sum for :math:`s=2` can itself be evaluated with
harmonic numbers and two explicitly specified auxiliary recurrences.
This is a fixed-order recurrence representation; it is not a claim that
only ordinary harmonic numbers are needed.

Define :math:`F_0=G_0=0` and, for :math:`n\geq1`,

.. math::
   :label: risk-edge-auxiliary-recurrences

   G_n=\tfrac12G_{n-1}+\frac1n,\qquad
   F_n=F_{n-1}-\frac{G_{n-1}}n+\frac1{n^2}.

Equivalently,
:math:`F_n=\sum_{j=1}^n1/(j^2\binom nj)` and
:math:`G_n=\sum_{j=1}^n1/(j\binom nj)`.
To see the recurrences directly, put
:math:`S_n=\sum_{j=0}^n1/\binom nj`.
Symmetry and Pascal's identity give
:math:`S_n=(n+1)S_{n-1}/(2n)+1` and :math:`G_n=S_{n-1}/n`.
The same identity applied to the sum defining :math:`F_n` gives its
displayed recurrence. In particular,
:math:`G_n=2^{-n}\sum_{j=1}^n2^j/j`.

.. prf:theorem:: Tetrahedron edge representation
   :label: thm-risk-tetrahedron-edge-representation

   For :math:`D\geq1`, set :math:`N=2D+3` and define

   .. math::

      Q_N={}&\frac{H_N^2}{N(N+1)(N+2)}
      +\frac{(N^2+2)H_N^{(2)}}{N^2(N+1)(N+2)}\\
      &-\frac{2(N-2)H_N}{N^3(N+2)}
       -\frac{N^2-3N+6}{N^4(N+2)},\\
      V_N={}&\frac{2(N^2+N+2)F_N}{N^2(N+1)(N+2)}
       -\frac{4(N-2)G_N}{N^3(N+2)}\\
      &-\frac{2(N^3-2N^2+3N+6)}{N^4(N+1)(N+2)}.

   For :math:`1\leq j<N`, put

   .. math::

      \Delta_j=j(j+1)(N-j)(N-j+1),\quad
      c_j=\frac{(N+1)(j(N-j)-1)}{\Delta_j},\\
      f_N(j)=\frac{H_{j+1}+H_{N-j+1}-H_N-c_j}{\Delta_j},\\
      v_N(j)=\frac1{j^2(j+1)(N-j)^2(N-j+1)\binom Nj},
      \qquad g_N(j)=f_N(j)-v_N(j).

   Then

   .. math::
      :label: risk-tetrahedron-edge-representation

      a_1(4,D)=432\left(Q_N-\frac12V_N-g_N(D)-g_N(D+1)\right).

   All denominators are nonzero on this range. The same formula gives
   :math:`a_1(4,1)=0`; at :math:`D=2,3,4,5` it gives
   :math:`36/7,33/7,169/44,34/11`.

.. prf:proof::

   Reindex :eq:`risk-codimension-two-coefficient` by :math:`j=L+1`.
   Its summand is :math:`432g_N(j)`. Both :math:`f_N` and :math:`v_N`
   are symmetric under :math:`j\mapsto N-j`. Since :math:`N=2D+3`,
   half the full range :math:`1\leq j<N` is :math:`1\leq j\leq D+1`.
   Remove the central terms :math:`j=D,D+1`.

   To evaluate the harmonic range, use

   .. math::

      \frac1{\Delta_j}=\frac1{N(N+1)}
       \left(\frac1j+\frac1{N-j}\right)
       -\frac1{(N+1)(N+2)}
       \left(\frac1{j+1}+\frac1{N-j+1}\right).

   The usual harmonic convolution identities give

   .. math::

      \sum_{j=1}^{N-1}\frac{H_{j+1}+H_{N-j+1}-H_N}{j}
       =\frac{H_N^2-H_N^{(2)}}2+\frac{2H_N}{N+1}+1-\frac2N,\\
      \sum_{j=1}^{N-1}\frac{H_{j+1}+H_{N-j+1}-H_N}{j+1}
       =\frac{H_N^2-H_N^{(2)}}2
        +2H_N\left(\frac1{N+1}+\frac1{N+2}\right)-1-\frac2{N+2}.

   For the rational correction, use

   .. math::

      \frac{c_j}{\Delta_j}={}&A_1\left(\frac1j+\frac1{N-j}\right)
       -B_1\left(\frac1{j+1}+\frac1{N-j+1}\right)\\
       &-A_2\left(\frac1{j^2}+\frac1{(N-j)^2}\right)
       -B_2\left(\frac1{(j+1)^2}+\frac1{(N-j+1)^2}\right),\\
      A_1={}&\frac{(N-1)(N^2+4N+2)}{N^3(N+1)^2},\qquad
      B_1=\frac{N^2+7N+8}{(N+1)^2(N+2)^2},\\
      A_2={}&\frac1{N^2(N+1)},\qquad
      B_2=\frac1{(N+1)(N+2)}.

   Summation of these rational terms and the two harmonic identities
   yields :math:`\frac12\sum f_N(j)=Q_N`.

   For the reciprocal range, write :math:`w_j=1/\binom Nj`,

   .. math::

      P(j)={}&\frac1{N+2}-\frac{(6N+4)j}{N^3(N+1)(N+2)}
        -\frac{6(N-2)j^2}{N^3(N+2)}
        +\frac{4(N-2)j^3}{N^3(N+1)(N+2)},\\
      R(j)={}&\frac{P(j)}{j^2(N-j+1)},\quad
      b_0=-\frac{8(N-2)}{N^3(N+1)(N+2)},\quad
      b_1=\frac{N^2+N+2}{N^2(N+1)(N+2)}.

   Direct multiplication verifies the rational identity

   .. math::

      \frac{v_N(j)}{w_j}
       =\frac{j+1}{N-j}R(j+1)-R(j)
         +b_0+b_1\left(\frac1{j^2}+\frac1{(N-j)^2}\right).

   The telescoping boundary is
   :math:`R(N)-R(1)/N=-2(N-1)(N^2+2N-4)/(N^4(N+1)(N+2))`.
   Finally
   :math:`\sum_{j=1}^{N-1}w_j=(N+1)G_N/2-1` and
   :math:`\sum_{j=1}^{N-1}w_j/j^2=F_N-1/N^2`.
   Substitution gives :math:`\sum v_N(j)=V_N` and proves the result.

Explicit scalar degree recurrences
-------------------------------------

.. prf:corollary:: Certified tetrahedron edge recurrences
   :label: cor-risk-tetrahedron-edge-scalar-recurrences

   The sequence :math:`a_1(4,D)` satisfies both of the following
   polynomial-coefficient recurrences for every integer :math:`D\geq1`:

   .. math::

      \sum_{j=0}^r p_j(D)a_1(4,D+j)=0.

   One operator has :math:`r=7` and maximum polynomial degree 67; another
   has :math:`r=8` and maximum polynomial degree 40. Their complete
   integer coefficients, in ascending powers of :math:`D`, are stored in
   ``tools/risk_research/examples/tetrahedron_edge_recurrence_order7.json``
   and ``tetrahedron_edge_recurrence_order8.json``.

   The leading polynomial of each operator has no positive integer root.
   Their initial values are the corresponding prefix of

   .. math::

      (a_1(4,1),\ldots,a_1(4,8))=
       \left(0,\frac{36}7,\frac{33}7,\frac{169}{44},\frac{34}{11},
       \frac{4708411}{1876875},\frac{250779}{121550},
       \frac{24981497}{14523600}\right).

.. prf:proof::

   This is a computer-assisted proof by exact rational identities,
   independently of the finite data that suggested the order-eight
   operator. Put :math:`N=2D+3` and

   .. math::

      v(D)=\left(H_N^2,H_N^{(2)},H_N,H_D,F_N,G_N,1,
        \binom{N}{D}^{-1}\right)^\mathsf T.

   The finite representation gives a rational row :math:`r(D)` with
   :math:`a_1(4,D)=432r(D)v(D)`. The state satisfies
   :math:`v(D+1)=T(D)v(D)` over :math:`\mathbb Q(D)`.
   In detail, writing :math:`a=1/(N+1)+1/(N+2)` and
   :math:`b=1/(N+1)^2+1/(N+2)^2`, its nonconstant updates are

   .. math::

      H_{N+2}^2&=H_N^2+2aH_N+a^2,&
      H_{N+2}^{(2)}&=H_N^{(2)}+b,\\
      H_{N+2}&=H_N+a,&H_{D+1}&=H_D+1/(D+1),\\
      G_{N+2}&=\tfrac14G_N+\frac1{2(N+1)}+\frac1{N+2},\\
      F_{N+2}&=F_N-\left(\frac1{N+1}+\frac1{2(N+2)}\right)G_N
       +b-\frac1{(N+1)(N+2)},\\
      \binom{2D+5}{D+1}^{-1}
       &=\frac{(D+1)(D+4)}{(2D+4)(2D+5)}\binom{2D+3}{D}^{-1}.

   For each supplied operator, exact multiplication verifies the eight
   component identities

   .. math::

      \sum_{j=0}^r p_j(D)r(D+j)
       T(D+j-1)\cdots T(D)=0,

   where the empty product is the identity matrix. All state and output
   denominators are nonzero for :math:`D\geq1`; the verifier checks that
   these nonzero polynomials have nonnegative coefficients. Thus these are
   identities for all
   stated degrees, rather than a finite list of zero residuals.
   The order-seven operator is obtained directly from a left kernel of
   these state rows, without numerical fitting.

   Every coefficient of the order-seven leading polynomial is strictly
   positive. Exact rational-root determination also excludes positive
   integer roots for the supplied operators. The finite representation
   supplies the initial values above, completing forward determination.
   ``tools/risk_research/edge_recurrence.py`` rebuilds the rational state
   and verifies the operators and initial values from their coefficient
   files; its reports include file hashes and the eight zero identities.

Triangular faces on a five-vertex simplex
-----------------------------------------

The angular multiplicity for :math:`s=3` is :math:`d_3(k)=k+1`.
One additional auxiliary sequence evaluates this weighted angular sum.
Use the same :math:`F_n,G_n,Q_N,V_N,g_N(j)` as in the tetrahedron-edge
representation; their displayed definitions make sense for even
:math:`N` as well as odd :math:`N`.

Define :math:`J_1=3/4` and, for :math:`n\geq1`,

.. math::
   :label: risk-triangle-auxiliary-state

   J_{n+1}=J_n+\frac1{(n+1)^2}
    -\frac{(3n+2)^2}{2n^2(2n+1)(2n+2)\binom{2n}{n}}.

.. prf:theorem:: Eight-state triangular-face representation
   :label: thm-risk-five-vertex-triangle-state

   For every integer :math:`D\geq1`, put :math:`n=D+2`, :math:`N=2n`,
   :math:`w_n=\binom{2n}{n}^{-1}` and

   .. math::

      A_n={}&\frac{n+1}{2n(n+2)}
       -\frac{H_n+H_{n+2}-H_N}{2n(n+1)},\\
      E_n={}&A_n+\frac{J_n}{2n}
       -\frac{(3n+1)w_n}{4n^3(n+1)}
       -\frac{2n^4+2n^3-3n^2+1}{4n^3(n+1)^2}.

   Then

   .. math::
      :label: risk-five-vertex-triangle-state

      a_2(5,D)=2400\left[
       -3\left(Q_N-\frac{V_N}2-\frac{g_N(n)}2\right)
       +E_n+g_N(n-2)+2g_N(n-1)\right].

   Every denominator is nonzero on this domain. In particular the same
   formula gives the inactive values :math:`a_2(5,1)=a_2(5,2)=0` and

   .. math::

      (a_2(5,3),a_2(5,4),a_2(5,5))
        =\left(\frac{40}3,\frac{255}{11},\frac{2675}{91}\right).

   The expression is a rational row on the eight-state vector

   .. math::

      v_n=\left(H_N^2-H_N^{(2)},H_N,H_n,
        F_N-H_N^{(2)},G_N,J_n,w_n,1\right)^{\mathsf T}.

   It therefore evaluates each new degree with a fixed number of rational
   operations on its state. This concerns arithmetic operation counts;
   integer bit lengths still grow with the degree.

.. prf:proof::

   Reindex :eq:`risk-codimension-two-coefficient` by :math:`j=L+1`.
   Its normalized summand is :math:`(n-3-j)g_N(j)`, so

   .. math::

      \frac{a_2(5,D)}{2400}
       =-3\sum_{j=1}^{n-1}g_N(j)
        +\sum_{j=1}^{n-1}(n-j)g_N(j)
        +g_N(n-2)+2g_N(n-1).

   This also gives zero for :math:`D=1,2` by cancellation. Symmetry gives
   :math:`\sum_{j=1}^{n-1}g_N(j)=Q_N-V_N/2-g_N(n)/2`.
   It remains to evaluate the weighted half range as :math:`E_n`.

   Write :math:`B_j=H_{j+1}+H_{N-j+1}-H_N` and
   :math:`\Delta_j=j(j+1)(N-j)(N-j+1)`. For
   :math:`r_j=1/(j(N-j+1))`, :math:`t_j=1/(j(N-j+2))`, direct
   rational identities give

   .. math::

      \frac{n-j}{\Delta_j}=\frac{r_j-r_{j+1}}2,
      \qquad r_j(B_j-B_{j-1})=t_j-t_{j+1}.

   Summation by parts, including both endpoints, yields
   :math:`\sum_{j=1}^{n-1}(n-j)B_j/\Delta_j=A_n`.
   The rational partial fractions used in the edge proof similarly give

   .. math::

      \sum_{j=1}^{n-1}\frac{(n-j)c_j}{\Delta_j}
       =\frac{H_N^{(2)}-2H_n^{(2)}}{4n}
        +\frac{(2n+1)(4n^3+3n^2-6n+3)}{16n^3(n+1)^2}.

   For the reciprocal term put :math:`b_j=\binom Nj^{-1}` and
   :math:`R_j=1/(2j^2(N-j+1))`. Exact multiplication verifies

   .. math::

      (n-j)v_N(j)=b_{j+1}R_{j+1}-b_jR_j
       +\frac{b_j}{2N}\left(\frac1{j^2}-\frac1{(N-j)^2}\right).

   Define :math:`T_n=\sum_{j=1}^{n-1}b_j/j^2`. The boundary is
   :math:`b_nR_n-b_1R_1=w_n/(2n^2(n+1))-1/(2N^2)` and symmetry gives

   .. math::

      \sum_{j=1}^{n-1}b_j\left(\frac1{j^2}-\frac1{(N-j)^2}\right)
        =2T_n-F_N+\frac{w_n}{n^2}+\frac1{N^2}.

   The partial first and zeroth moments are

   .. math::

      \sum_{j=1}^{n-1}\frac{b_j}{j}=\frac{G_N}2-\frac{w_n}{n},
      \qquad
      \sum_{j=1}^{n-1}b_j=\frac{N+1}{4}G_N-\frac12-\frac{w_n}2.

   The first follows by telescoping
   :math:`b_j(1/j-1/(N-j))`; the second follows from symmetry and the
   full reciprocal sum in the edge proof. The ratio
   :math:`\binom{N+2}{j}^{-1}/b_j=(N+2-j)(N+1-j)/((N+1)(N+2))`
   and the added endpoint :math:`j=n` now give

   .. math::

      T_{n+1}-T_n=-\frac{(6n+5)G_N}{4(N+1)(N+2)}
        -\frac1{2(N+1)(N+2)}
        +\frac{(3n+2)^2w_n}{2n^2(N+1)(N+2)}.

   Consequently :math:`J_n=H_n^{(2)}-T_n+(F_N-H_N^{(2)})/2`
   has initial value :math:`3/4` and satisfies
   :eq:`risk-triangle-auxiliary-state`. Combining the three weighted
   ranges gives exactly :math:`E_n`.

   Finally :math:`w_{n+1}=(n+1)w_n/(2(2n+1))`. The other state
   transitions follow from harmonic increments and the displayed
   :math:`F,G` recurrences. In :math:`Q_N-V_N/2`, the coefficients of
   :math:`H_N^2,H_N^{(2)},F_N` sum to zero, so the two differences in
   :math:`v_n` suffice. Each central term uses
   :math:`H_{n+r}=H_n+\sum_{i=1}^r1/(n+i)` and
   :math:`\binom{N}{n-r}^{-1}=w_n\prod_{i=1}^r(n+i)/(n-i+1)`
   for :math:`r=0,1,2`; only a fixed number of shifts occurs.
   Thus all output coefficients and all state transitions are rational
   in :math:`n`, proving the asserted fixed-state representation.

.. prf:corollary:: Certified triangular-face degree recurrence
   :label: cor-risk-five-vertex-triangle-recurrence

   The sequence :math:`a_2(5,D)` satisfies

   .. math::

      \sum_{j=0}^8 p_j(D)a_2(5,D+j)=0,\qquad D\geq1,

   where the complete integer coefficient lists are stored in
   ``tools/risk_research/examples/five_vertex_triangle_recurrence_order8.json``.
   The order is eight and the maximum polynomial degree is 76. The leading
   coefficient is strictly positive for every real :math:`D\geq1`, so
   there are no positive-integer exceptional indices. Its initial values are

   .. math::

      (a_2(5,1),\ldots,a_2(5,8))=
        \left(0,0,\frac{40}3,\frac{255}{11},\frac{2675}{91},
          \frac{96345}{2912},\frac{2934923}{83538},
          \frac{84006889}{2325600}\right).

.. prf:proof::

   This is an exact rational-state certificate, with the same construction
   as :prf:ref:`cor-risk-tetrahedron-edge-scalar-recurrences`.
   ``tools/risk_research/triangle_recurrence.py`` reconstructs the eight
   state transitions and rational output row of
   :prf:ref:`thm-risk-five-vertex-triangle-state` over :math:`\mathbb Q(D)`.
   Exact elimination supplies the displayed operator; multiplication of
   its coefficient row by the shifted state-output rows gives eight zero
   rational functions. After :math:`n=D+2`, every output and transition
   denominator is a nonzero polynomial with nonnegative coefficients,
   proving validity for :math:`D\geq1`. Every coefficient of
   :math:`p_8(1+z)` is strictly positive, proving the leading-coefficient
   claim. The fixed-state representation supplies the eight initial values.

Large-degree constants
------------------------

.. prf:corollary:: Codimension-two large-degree risk
   :label: cor-risk-codimension-two-asymptotic

   For every fixed integer :math:`s\geq2`,

   .. math::
      :label: risk-codimension-two-asymptotic

      a_{s-1}(s+2,D)\sim
       \frac{s(s+1)^3(s+2)^2\pi^2}{48(s-2)!}\,D^{s-4}.

   In particular,

   .. math::

      a_1(4,D)=\frac{18\pi^2}{D^2}
        +O\!\left(\frac{\log^2D}{D^3}\right).

.. prf:proof::

   In :eq:`risk-codimension-two-coefficient`, put :math:`j=L+1`.
   For fixed :math:`j`, the multiplicity divided by :math:`D^{s-2}`
   tends to :math:`1/(s-2)!`, and

   .. math::

      D^2\rho_{s,k,D}\longrightarrow
       \frac{s(s+1)H_j}{2j(j+1)}.

   The bracket in :eq:`risk-codimension-two-mode` is nonnegative because
   it is a squared norm. It is at most :math:`H_{j+1}`: the other
   harmonic difference is nonpositive, and both subtracted terms are
   nonnegative. Also :math:`(N-j)(N-j+1)\geq D^2`, and the scaled
   multiplicities are bounded by :math:`1/(s-2)!`. A constant multiple
   of the summable sequence :math:`H_{j+1}/(j(j+1))` dominates the
   scaled summands. Dominated convergence applies, with

   .. math::

      \sum_{j\geq1}\frac{H_j}{j(j+1)}
       =\sum_{i\geq1}\frac1i\sum_{j\geq i}\frac1{j(j+1)}
       =\frac{\pi^2}6.

   Multiplication by :math:`\binom{s+2}s^2` proves the first result.
   For the sharper tetrahedron error, :math:`Q_N=\pi^2/(6N^2)
   +O(\log^2N/N^3)`, each central :math:`f_N` is
   :math:`O(\log N/N^4)`, and
   :math:`0\leq\sum_{j=1}^{D-1}v_N(j)\leq16/N^4`.
   Substitute :math:`N=2D+3` in the finite formula.

Vertex boundary-pair reduction
-------------------------------

Let :math:`q=M-1`, :math:`f(x)=-\log(1-x)/x`, and

.. math::

   A_h=(-1)^h\binom{D-1}h\binom{D+1}h,\qquad
   \mathcal L_D(G)=\sum_{h,l=0}^{D-1}
     A_hA_lh!l!(2D-h-l)!\,[x^hy^l]G.

Then :math:`a_0(M,D)=M^2\mathcal L_D(F^q)/(M)_{2D}` by
:prf:ref:`thm-skeleton-scalar-sum`.

.. prf:theorem:: Two fewer outside-kernel factors for vertex risk
   :label: thm-risk-vertex-boundary-pair

   For all :math:`M\geq2,D\geq1`,

   .. math::
      :label: risk-vertex-boundary-pair

      \mathcal L_D(F^q)=(-1)^{q+1}\mathcal L_D(f(x)^q)
       +\sum_{\substack{a,b\geq1\\a+b\leq q}}
        (-1)^{a+b}\frac{q!}{a!b!(q-a-b)!}
        \mathcal L_D(F^{q-a-b}f(x)^af(y)^b).

   For :math:`q=1` the sum is empty. In particular, symmetry gives

   .. math::

      \mathcal L_D(F^3)=6\mathcal L_D(Ff(x)f(y))
       -6\mathcal L_D(f(x)^2f(y))+\mathcal L_D(f(x)^3).

.. prf:proof::

   Write :math:`x` for the source-vertex coordinate. For a subset
   :math:`S` of the :math:`q` outside coordinates, let :math:`p_S` be
   the homogeneous degree-:math:`D` lift of the vertex error on its face
   with :math:`x`. The scalar-mode extension has coefficients independent
   of the ambient number of coordinates. Dirichlet moments therefore give

   .. math::

      \langle p_{[q]},p_S\rangle
       =\frac{\mathcal L_D(F^{|S|}f(x)^{q-|S|})}{(q+1)_{2D}}.

   The inclusion-exclusion lift
   :math:`b=\sum_{S\subsetneq[q]}(-1)^{q-|S|+1}p_S` agrees with
   :math:`p_{[q]}` on every facet. Their difference is a full bubble,
   so :math:`\langle p_{[q]},p_{[q]}\rangle=\langle p_{[q]},b\rangle`.
   Grouping subsets by size yields

   .. math::

      \mathcal L_D(F^q)=\sum_{r=0}^{q-1}
       (-1)^{q-r+1}\binom qr\mathcal L_D(F^rf(x)^{q-r}).

   For :math:`1\leq r<q`, condition on the sum of the coordinates of
   the corresponding face. Its normalized angular coordinates are uniform
   and independent of that sum and the remaining coordinates. The terms
   of :math:`p_{[q]}` positive in all :math:`r` selected outside
   coordinates also have positive source-coordinate :math:`x` exponent.
   They are face bubbles of angular degree at most :math:`D`;
   they are orthogonal to :math:`p_S`. Removing the zero first-exponent
   axis in each coordinate kernel therefore gives

   .. math::

      \mathcal L_D((F-f(y))^rf(x)^{q-r})=0.

   If :math:`D<r+1`, the same statement holds because the relevant
   monomials do not exist. Expand this identity in the preceding subset
   sum and put :math:`a=q-r`, :math:`b=r-j`. The resulting signs and
   binomial factors are exactly :eq:`risk-vertex-boundary-pair`.

Verification and remaining questions
--------------------------------------

``tools/risk_research/kernel_identities.py`` independently evaluates
:eq:`risk-two-coordinate-kernel`; its tests compare exact polynomial
convolution and the axis boundary through degree 100.
``tools/risk_research/edge_closed.py`` evaluates
:eq:`risk-tetrahedron-edge-representation` and the general codimension-two
mode formula. Tests compare the pre-existing scalar sum, independent Gram
propagation and direct reciprocal-binomial sums, including degree one.
``vertex_boundary.py`` checks the reduced vertex formula with exact
coefficient arithmetic. Finite checks detect errors; the all-degree
claims rely on the proofs above.
``triangle_faces.py`` evaluates :eq:`risk-five-vertex-triangle-state`
using a fixed state and checks independent scalar and Gram references.
``triangle_recurrence.py`` verifies its complete scalar operator by eight
exact rational identities, including the initial values and denominator
conditions, independently of finite sequence residual checks.

:doc:`skeleton_risk_vertex_recurrence` supplies a certified order-ten
tetrahedron vertex recurrence at every positive integer degree. A bounded
classical-special-function expression remains open.
The boundary-pair identity reduces its mixed part to
:math:`\mathcal L_D(Ff(x)f(y))`; :doc:`skeleton_risk_vertex` further
evaluates the complete vertex coefficient by one positive harmonic sum
and proves its leading large-degree constant. Bounded polynomial/harmonic ansatz spaces
can be excluded by exact modular rank, but such exclusions say nothing
about formulas outside the recorded spaces. A general elementary or
classical-special-function evaluation of the remaining codimension-two
angular sums is also still open beyond the fixed-state evaluations above
and the common fifteen-state construction for every nonvertex codimension-two
face in :doc:`codimension_two_family`. That construction has a state bound
independent of face size; it does not assert a bounded classical-special-function
expression or a minimal scalar order.
:doc:`skeleton_risk_vertex_general_asymptotics` supplies an explicit
finite-degree remainder for every fixed vertex count at least four, with
a computable constant and the common degree domain :math:`D\geq256`.
It distinguishes the analytic error from truncating the leading constant
and from arithmetic rounding.
:doc:`skeleton_risk_comparisons` settles the complete comparison at every
fixed degree through thirty. A general crossover law and the nonlinear
self-identification questions of :doc:`skeleton_risk_theory` remain.
The fixed-tetrahedron comparison is completely classified over all degrees
in :doc:`skeleton_risk_tetrahedron_comparison`.
