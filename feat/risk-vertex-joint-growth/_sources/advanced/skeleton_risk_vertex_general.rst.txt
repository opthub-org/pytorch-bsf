Vertex coefficients at every fixed vertex count
================================================

This chapter keeps the uniform-simplex assumptions of
:doc:`skeleton_risk_theory`. It gives a constructive one-dimensional
convolution formula at every vertex count and proves the leading vertex
constant for **each fixed** :math:`M\geq4`. A quantified regime with
both :math:`M` and the degree growing is proved separately in
:doc:`skeleton_risk_vertex_joint_growth`.

A rational univariate construction
-----------------------------------

Put :math:`g(t)=-(1-t)\log(1-t)/t`, with :math:`g(0)=1`, and set

.. math::

   \phi_r(j)=[t^j]\frac{g(t)^r}{1-t},\qquad
   e_r(j)=\sum_{a=1}^{r-1}\binom ra\phi_a(j)\phi_{r-a}(j).

Thus :math:`e_0=e_1=0`. Let :math:`b_k(j)=[t^j]g(t)^k` and use ordinary
sequence convolution. In particular, :math:`b_0` is the unit sequence
and :math:`b_k(j)=\phi_k(j)-\phi_k(j-1)`, where
:math:`\phi_k(-1)=0`. Define

.. math::
   :label: risk-general-transformed-diagonal

   \delta_q(j)=(-1)^{q+1}\phi_q(j)
       +\sum_{k=0}^{q-2}(-1)^k\binom qk(b_k*e_{q-k})(j).

An empty sum is zero. These sequences require only rational arithmetic.
For example, writing :math:`E_l` for an elementary symmetric polynomial,

.. math::
   :label: risk-general-phi-elementary

   \phi_r(j)=\frac{r!(r-1)!j!}{(j+r)!}
    E_{r-1}\left(1,\frac12,\ldots,\frac1{r-1},
                    -1,-\frac12,\ldots,-\frac1j\right).

The positive list is empty at :math:`r=1`; the negative list is empty
at :math:`j=0`. Adding :math:`-1/(j+1)` updates the finite symmetric
polynomial state, while the factorial ratio has a rational update.

.. prf:theorem:: Constructive vertex formula at every vertex count
   :label: thm-risk-general-vertex-construction

   For integers :math:`M\geq2`, :math:`D\geq1`, let
   :math:`W_{D,j}=\binom{D-1}j\binom{D+1}j` and
   :math:`Z_D=\binom{2D}{D-1}`. Then

   .. math::
      :label: risk-general-vertex-single

      a_0(M,D)=\frac{M^2(M-1)!}
        {Z_D\prod_{h=1}^{M-1}(2D+h)}
          \sum_{j=0}^{D-1}W_{D,j}\delta_{M-1}(j).

   Every :math:`\delta_q(j)` is strictly positive. Formula
   :eq:`risk-general-transformed-diagonal` evaluates it by univariate
   convolutions, without constructing a bivariate outside kernel.

.. prf:proof::

   The transformed-kernel identity in :doc:`skeleton_risk_vertex` gives

   .. math::

      \delta_q(j)=[x^jy^j]
        \frac{(g(x)+g(y)-g(xy))^q}{(1-x)(1-y)}.

   Expanding by powers of :math:`g(xy)` gives
   :math:`\sum_{k=0}^q(-1)^k\binom qk(b_k*d_{q-k})(j)`, where
   :math:`d_r=2\phi_r+e_r` for :math:`r\geq1` and :math:`d_0=1`.
   Since :math:`b_k*\phi_{q-k}=\phi_q`, the edge terms combine to
   :math:`(-1)^{q+1}\phi_q`. This proves
   :eq:`risk-general-transformed-diagonal`.
   The previously proved binomial Gram diagonalization supplies
   :eq:`risk-general-vertex-single`; positivity is
   :prf:ref:`prop-risk-positive-transformed-kernel`.

   To prove :eq:`risk-general-phi-elementary`, extract the coefficient
   of :math:`t^{j+r}` after differentiating
   :math:`(1-t)^\alpha` exactly :math:`r` times at
   :math:`\alpha=r-1`. The generalized binomial coefficient has one
   vanishing factor there. Expanding its remaining factors yields the
   displayed elementary symmetric polynomial, including both empty-list
   cases. Thus the evaluator does not require symbolic differentiation.

A positive integral for the leading constant
--------------------------------------------

.. prf:theorem:: Vertex leading constant for every fixed M at least four
   :label: thm-risk-general-vertex-asymptotic

   For each fixed integer :math:`M\geq4`, put :math:`q=M-1`. Then

   .. math::

      a_0(M,D)\sim\frac{C_M}{D^{M+1}},\qquad
      C_M=M^2(M-1)!\,2^{3-M}c_q>0,

   where the following absolutely convergent expressions agree:

   .. math::
      :label: risk-general-vertex-constant

      c_q=q\sum_{j=0}^{\infty}e_{q-1}(j)
         =\frac{q}{2\pi}\int_{-\pi}^{\pi}
          \frac{\left(2\operatorname{Re}g(e^{it})\right)^{q-1}}
               {|1-e^{it}|^2}\,dt.

   At :math:`M=4,5`, this recovers
   :math:`C_4=48\pi^2` and
   :math:`C_5=7200(\zeta(2)-\zeta(3))`.

.. prf:proof::

   **Coefficient bounds.** For each fixed :math:`r\geq1`, the elementary
   symmetric formula and
   :math:`|E_l(x)|\leq(\sum|x_i|)^l/l!` give

   .. math::
      :label: risk-general-phi-bound

      |\phi_r(j)|\leq B_r
        \frac{(1+\log(j+1))^{r-1}}{(j+1)^r},
      \qquad B_r=r!r^{r-1}.

   Indeed, the reciprocal sum is at most
   :math:`H_{r-1}+H_j\leq r+\log(j+1)` and
   :math:`j!/(j+r)!\leq(j+1)^{-r}`.
   Differentiating the same binomial product at :math:`\alpha=r`
   instead proves

   .. math::

      b_r(j)=O_r\!\left(\frac{(1+\log(j+1))^{r-1}}
                                {(j+1)^{r+1}}\right).

   Consequently :math:`\sum_jb_r(j)=g(1)^r=0` for :math:`r\geq1`:
   the coefficients are absolutely summable, so Abel's limit is the
   ordinary sum. For :math:`r\geq2`, :math:`b_r` also has an absolutely
   summable first absolute moment. For :math:`r=1`,
   :math:`b_1(j)=-1/(j(j+1))` at positive indices, and its first absolute
   moment through :math:`n` is :math:`O(1+\log n)`.

   Products and differences of :eq:`risk-general-phi-bound`, using
   :math:`\phi_r(j+1)-\phi_r(j)=b_r(j+1)`, give, for :math:`r\geq2`,

   .. math::

      e_r(j)&=O_r((1+\log(j+1))^{r-2}(j+1)^{-r}),\\
      e_r(j+1)-e_r(j)
        &=O_r((1+\log(j+1))^{r-2}(j+1)^{-r-1}).

   In particular, :math:`e_r` is absolutely summable.

   **The convolution limit.** Fix :math:`k\geq1`, :math:`r\geq2`.
   Split :math:`(b_k*e_r)(n)` at :math:`l=\lfloor n/2\rfloor` in
   the index of :math:`b_k(l)`. On the first part, subtract
   :math:`e_r(n)\sum_{l\geq0}b_k(l)=0`. The difference estimate bounds
   this part by

   .. math::

      O_{k,r}\!\left(\frac{(1+\log n)^{r-2}}{n^{r+1}}
                   \sum_{l\leq n/2}l|b_k(l)|\right)
       +|e_r(n)|\sum_{l>n/2}|b_k(l)|=o(n^{-2}).

   For the second part set :math:`m=n-l<n/2`. The coefficients satisfy
   :math:`n^2b_1(n-m)\to-1` and
   :math:`n^2b_k(n-m)\to0` for :math:`k\geq2`, at every fixed :math:`m`.
   They are uniformly bounded over :math:`m<n/2`, by the coefficient
   bounds above. Absolute summability of :math:`e_r` permits dominated
   convergence. Hence

   .. math::

      n^2(b_1*e_r)(n)\longrightarrow-\sum_me_r(m),\qquad
      n^2(b_k*e_r)(n)\longrightarrow0\quad(k\geq2).

   In :eq:`risk-general-transformed-diagonal`, both
   :math:`\phi_q(n)` and the :math:`k=0` term :math:`e_q(n)` are
   :math:`o(n^{-2})` when :math:`q\geq3`. Its :math:`k=1` term proves
   :math:`n^2\delta_q(n)\to c_q` with the displayed series.

   **Boundary integrals and positivity.** The sequences
   :math:`\phi_a`, :math:`a\geq1`, are square summable. Parseval, or
   finite polynomial Parseval followed by their Abel limits, gives

   .. math::

      \sum_j\phi_a(j)\phi_b(j)
       =\frac1{2\pi}\int_{-\pi}^{\pi}
           \frac{g(e^{it})^a\overline{g(e^{it})}^{\,b}}
                {|1-e^{it}|^2}\,dt\quad(a,b\geq1).

   All these boundary products are integrable: near zero they are bounded
   by a constant times
   :math:`|t|^{a+b-2}(1+|\log|t||)^{a+b}`.
   The analytic terms absent from :math:`e_r` have zero integral. Indeed,
   on :math:`|z|=1`,

   .. math::

      \frac{g(z)^r}{|1-z|^2}=-\frac{zg(z)^r}{(1-z)^2}.

   The right side is analytic in the disk with zero constant coefficient.
   Its circle integrals vanish at every radius less than one. For radii
   at least one half its absolute value is dominated, uniformly near
   :math:`t=0`, by
   :math:`O_r((1+|\log|t||)^r)` for :math:`r\geq2`; use
   :math:`|1-\rho e^{it}|\geq c|t|` and the explicit formula for
   :math:`g`. Dominated convergence therefore proves that its boundary
   integral vanishes too. Summing the mixed Parseval identities gives
   :eq:`risk-general-vertex-constant`.

   Finally, for :math:`z=e^{it}\ne1`,

   .. math::

      g(z)=\int_0^1\frac{1-z}{1-uz}\,du,\qquad
      \operatorname{Re}\frac{1-z}{1-uz}
        =\frac{(1+u)(1-\cos t)}{|1-uz|^2}>0.

   The integral for :math:`c_q` is thus strictly positive. Its absolute
   convergence also follows from the preceding boundary bound.

   **The weighted vertex limit.** Under probabilities
   :math:`W_{D,j}/Z_D`, the random variable :math:`J` satisfies
   :math:`J/D\to1/2` in probability and
   :math:`\mathbb P(J\leq\lfloor D/4\rfloor)\leq4\,2^{-D/12}`,
   as proved in :doc:`skeleton_risk_vertex`. The diagonal limit implies
   a bound :math:`|\delta_q(j)|\leq K_q/(j+1)^2`. On the complementary
   central event, :math:`D^2\delta_q(J)` is bounded and converges in
   probability to :math:`4c_q`; its expectations therefore converge.
   The small-index event contributes at most
   :math:`4K_qD^2 2^{-D/12}\to0`. Multiplying the outer prefactor in
   :eq:`risk-general-vertex-single` proves the theorem.

   For the known constants, :math:`e_2(j)=2/(j+1)^2` gives
   :math:`c_3=6\zeta(2)`. At :math:`M=5`, the same limit agrees with
   :prf:ref:`cor-risk-five-vertex-asymptotic`, giving
   :math:`c_4=48(\zeta(2)-\zeta(3))`.

The critical triangle case is separate: :math:`q=2` leaves
:math:`\delta_2=-\phi_2+e_2`, whose first term has a logarithm at order
:math:`j^{-2}`. The known triangle formula gives
:math:`a_0(3,D)\sim36\log D/D^4`, rather than a finite constant at that
normalization.

Certified constant enclosures
------------------------------

The infinite constant does not require unverified numerical integration.
For :math:`r\geq2`, put

.. math::

   A_r=\sum_{a=1}^{r-1}\binom ra B_aB_{r-a},\qquad
   m=r-2,\qquad L_T=1+\lceil\log_2T\rceil.

For every integer :math:`T\geq1`, :eq:`risk-general-phi-bound` and the
decreasing function :math:`(1+\log x)^m/x^r` give

.. math::
   :label: risk-general-constant-tail

   \left|c_q-q\sum_{j=0}^{T-1}e_{q-1}(j)\right|
    \leq\frac{qA_r}{T^{r-1}}
       \sum_{k=0}^{m}\frac{m!L_T^{m-k}}
              {(m-k)!(r-1)^{k+1}},\qquad r=q-1.

To check the integral exactly, substitute :math:`u=\log x` in
:math:`\int_T^\infty(1+\log x)^m x^{-r}\,dx` and integrate by parts
:math:`m` times. Replacing :math:`1+\log T` by the integer
:math:`L_T` increases every positive term. The tool computes this
rational bound and the partial sum exactly, then scales by
:math:`M^2(M-1)!2^{3-M}`. The known positivity permits clipping a
negative lower bound to zero. Increasing :math:`T` converges to the
true constant for each fixed :math:`M`; no claim of uniform cost in
:math:`M` is made.

From a source checkout, for example:

.. code-block:: console

   $ .venv/bin/python -m tools.risk_research.vertex_general --vertices 6 --constant-terms 128 --output .research/skeleton-risk/vertex6-constant.json
   $ .venv/bin/python -m tools.risk_research.vertex_general --vertices 6 --degree 32 --output .research/skeleton-risk/vertex6-exact.json

``vertex_general.vertex_coefficient`` implements the all-degree formula;
``leading_constant_enclosure`` returns two exact fractions. These are
checkout-only research tools, with no additional runtime dependency or
public sampling API. This chapter certifies the constant's truncation error.
:doc:`skeleton_risk_vertex_general_asymptotics` separately proves an explicit
finite-degree error for every fixed :math:`M\geq4` and :math:`D\geq256`,
combining the constant interval with an analytic remainder. The sharper
five-vertex finite-degree error remains a separate result.

Independent implementation checks compare the elementary rows with
logarithmic-series expansion, the convolutions with the original bivariate
kernel and signed transform, and the coefficient with the original scalar
and rational Gram oracles. After freezing the candidate, a separate
integer-polynomial computation of :math:`F^5` at degrees 67, 79 and 97
agreed exactly; the committed artifact records both source hashes.
Native Arb checks place the independently known constants at M4/M5
inside the exact rational intervals. These finite checks support the
implementation; the general claims use the proofs above.
