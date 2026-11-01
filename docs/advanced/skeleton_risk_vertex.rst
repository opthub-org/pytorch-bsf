Tetrahedron vertex risk
========================

Under the uniform-simplex assumptions of :doc:`skeleton_risk_theory`, the
tetrahedron vertex coefficient has a positive single sum involving only
binomial coefficients and harmonic numbers. This eliminates the bivariate
kernel convolution. The sum still has :math:`D` terms. An explicit certified
degree recurrence is proved in :doc:`skeleton_risk_vertex_recurrence`;
a bounded classical-special-function expression remains open.
The next asymptotic term and a fully explicit error bound are proved in
:doc:`skeleton_risk_vertex_asymptotics`.

A binomial-transform diagonalization
--------------------------------------

Put :math:`A_h=(-1)^h\binom{D-1}h\binom{D+1}h`,
:math:`C_D=(D-1)!(D+1)!`, and
:math:`W_{D,j}=\binom{D-1}j\binom{D+1}j`. As in
:doc:`skeleton_risk_extensions`, define

.. math::

   \mathcal L_D(G)=\sum_{h,l=0}^{D-1}
    A_hA_lh!l!(2D-h-l)!\,[x^hy^l]G.

.. prf:theorem:: Diagonalization by signed binomial transforms
   :label: thm-risk-binomial-gram-diagonal

   For every integer :math:`D\geq1` and arrays :math:`u,v` of length
   :math:`D`, set

   .. math::

      \widehat u_j=\sum_{h=0}^j(-1)^h\binom jh u_h,
      \qquad
      \widehat v_j=\sum_{l=0}^j(-1)^l\binom jl v_l.

   Then

   .. math::
      :label: risk-binomial-gram-diagonal

      \sum_{h,l=0}^{D-1}A_hA_lh!l!(2D-h-l)!u_hv_l
       =C_D\sum_{j=0}^{D-1}W_{D,j}\widehat u_j\widehat v_j.

   Consequently, if

   .. math::

      \mathscr T(G)(x,y)=
       \frac{G(-x/(1-x),-y/(1-y))}{(1-x)(1-y)},

   then

   .. math::
      :label: risk-transformed-vertex-functional

      \mathcal L_D(G)=C_D\sum_{j=0}^{D-1}
        W_{D,j}[x^jy^j]\mathscr T(G).

.. prf:proof::

   Write :math:`n=D-1` and
   :math:`P_u(t)=\sum_{h=0}^n A_hh!u_ht^{D-h}`. The Laguerre expansion

   .. math::

      L_k^{(2)}(t)=\sum_{r=0}^k
       (-1)^r\binom{k+2}{k-r}\frac{t^r}{r!}

   and binomial inversion give

   .. math::

      P_u(t)=(-1)^n n!t\sum_{j=0}^n
       (-1)^j\binom{n+2}j\widehat u_jL_{n-j}^{(2)}(t).

   Equality of the coefficients of :math:`t^{r+1}` is exactly ordinary
   binomial inversion for :math:`u_{n-r}`. Integrate
   :math:`P_u(t)P_v(t)e^{-t}` on :math:`[0,\infty)`. The monomial moments
   are :math:`(2D-h-l)!`, while Laguerre orthogonality gives

   .. math::

      \int_0^\infty t^2e^{-t}L_k^{(2)}(t)L_l^{(2)}(t)\,dt
       =\mathbf1_{k=l}\frac{(k+2)!}{k!}.

   Its surviving coefficient satisfies

   .. math::

      n!^2\binom{n+2}j^2\frac{(n+2-j)!}{(n-j)!}
       =C_DW_{D,j},

   proving :eq:`risk-binomial-gram-diagonal`. Apply the identity linearly
   to each coefficient of :math:`G`; the generating-function identity for
   signed binomial transforms is precisely :math:`\mathscr T`.
   All relevant arrays and polynomials are finite.

The logarithmic formula for the outside kernel simplifies under this
substitution. With :math:`f(t)=-\log(1-t)/t` and
:math:`g(t)=(1-t)f(t)`, direct collection of logarithms gives

.. math::
   :label: risk-transformed-outside-kernel

   F\!\left(-\frac{x}{1-x},-\frac{y}{1-y}\right)
     =g(x)+g(y)-g(xy).

Indeed, if :math:`X=-x/(1-x)`, :math:`Y=-y/(1-y)`, then
:math:`1-X-Y=(1-xy)/((1-x)(1-y))`. Substitution into the numerator
:math:`(1-X-Y)\log(1-X-Y)-(1-X)\log(1-X)-(1-Y)\log(1-Y)`
of :math:`F(X,Y)` proves :eq:`risk-transformed-outside-kernel`.
For every fixed vertex count, this transforms
:math:`\mathcal L_D(F^{M-1})` into a weighted diagonal of the corresponding
power of an additive kernel. No evaluation of all those diagonals is claimed.

The positive single sum
-------------------------

Write :math:`H_j=\sum_{r=1}^j1/r`,
:math:`H_j^{(2)}=\sum_{r=1}^j1/r^2`, with both values zero at :math:`j=0`.

.. prf:theorem:: Tetrahedron vertex coefficient at every degree
   :label: thm-risk-tetrahedron-vertex-single

   For every integer :math:`D\geq1`,

   .. math::
      :label: risk-tetrahedron-vertex-single

      a_0(4,D)=\frac{96(D-1)!(D+1)!}{(4)_{2D}}
       \sum_{j=0}^{D-1}\binom{D-1}j\binom{D+1}j G_j,

   where

   .. math::
      :label: risk-tetrahedron-vertex-weight

      G_j={}&\frac{H_j^2}{(j+1)(j+2)(j+3)}
        +\frac{H_j^{(2)}}{(j+1)(j+3)}\\
        &-\frac{jH_j}{(j+1)(j+2)^2(j+3)}
        +\frac{2}{(j+1)^2(j+2)^2(j+3)}.

   Every weight :math:`G_j` is strictly positive, including
   :math:`G_0=1/6`. All denominators are nonzero on the stated domain.

.. prf:proof::

   Set :math:`\phi_r(j)=[t^j]g(t)^r/(1-t)`. Elementary logarithmic
   coefficient identities give

   .. math::

      \phi_1(j)&=\frac1{j+1},\\
      \phi_2(j)&=\frac{2(1-H_j)}{(j+1)(j+2)},\\
      \phi_3(j)&=\frac{6(H_j^2-H_j^{(2)}-3H_j+1)}
                       {(j+1)(j+2)(j+3)}.

   For example, differentiate the finite product
   :math:`\binom\alpha{j+r}` :math:`r` times at :math:`\alpha=r-1`,
   using
   :math:`g(t)^r/(1-t)=(1-t)^{r-1}(-\log(1-t))^r/t^r`.
   The product has a simple zero there; the logarithmic derivatives of its
   remaining factors give the three formulas, including :math:`j=0`.

   Expand :math:`(g(x)+g(y)-g(xy))^3` and divide by
   :math:`(1-x)(1-y)`. The pure powers contribute :math:`\phi_3(j)`.
   The terms :math:`g(x)^2g(y)` and its symmetric counterpart contribute
   :math:`6\phi_1(j)\phi_2(j)`. The diagonal contributions of
   :math:`g(x)^2g(xy)` and :math:`g(x)g(xy)^2`, together with their
   symmetric counterparts, cancel by ordinary convolution. Thus

   .. math::

      [x^jy^j]\mathscr T(F^3)
       =\phi_3(j)+6\phi_1(j)\phi_2(j)
        -6\sum_{k=0}^j[t^k]g(t)\phi_1(j-k)^2.

   Since :math:`[t^0]g=1` and
   :math:`[t^k]g=-1/(k(k+1))` for :math:`k\geq1`, the convolution equals
   :math:`1/(j+1)^2-B_j`, where

   .. math::

      B_j={}&\sum_{r=0}^{j-1}
        \frac1{(r+1)(r+2)(j-r)^2}\\
       ={}&\frac{H_j^{(2)}}{(j+1)(j+2)}
          +\frac{2(2j+3)H_j}{(j+1)^2(j+2)^2}
          +\frac{j}{(j+1)(j+2)^2}.

   The last equality follows by partial fractions and includes the empty
   sum :math:`B_0=0`. Substituting the three :math:`\phi_r` formulas and
   simplifying gives :math:`[x^jy^j]\mathscr T(F^3)=6G_j`. Apply
   :eq:`risk-transformed-vertex-functional` and the scalar risk identity
   :math:`a_0(4,D)=16\mathcal L_D(F^3)/(4)_{2D}`.

   For positivity, group the first and third terms of :math:`G_j` as

   .. math::

      \frac{H_j(H_j-j/(j+2))}{(j+1)(j+2)(j+3)}.

   At :math:`j\geq1`, :math:`H_j\geq1>j/(j+2)`; at zero the last
   rational term is positive. The second and last terms are nonnegative.

Large-degree behavior
-----------------------

.. prf:corollary:: Tetrahedron vertex asymptotic
   :label: cor-risk-tetrahedron-vertex-asymptotic

   As :math:`D\to\infty`,

   .. math::
      :label: risk-tetrahedron-vertex-asymptotic

      a_0(4,D)=\frac{48\pi^2}{D^5}
        +O\!\left(\frac{\log^2D}{D^6}\right).

.. prf:proof::

   Vandermonde's identity gives
   :math:`Z_D=\sum_jW_{D,j}=\binom{2D}{D-1}`. Let :math:`J` have
   probabilities :math:`W_{D,j}/Z_D`. Since :math:`C_DZ_D=(2D)!`,

   .. math::

      a_0(4,D)=\frac{576}{(2D+1)(2D+2)(2D+3)}\,\mathbb E[G_J].

   The first two factorial moments, obtained from Vandermonde, give

   .. math::

      \mu=\mathbb E[J]=\frac{D^2-1}{2D},\qquad
      \operatorname{Var}(J)=\frac{(D^2-1)^2}{4D^2(2D-1)}=O(D).

   An elementary weight ratio controls the potentially small indices.
   For :math:`D\geq12`, put :math:`a=\lfloor D/4\rfloor`,
   :math:`b=\lfloor D/3\rfloor`. For :math:`0\leq j<b`,

   .. math::

      \frac{W_{D,j+1}}{W_{D,j}}
       =\frac{(D-1-j)(D+1-j)}{(j+1)^2}\geq2.

   The numerator is at least :math:`4D^2/9-1`, the denominator is at
   most :math:`(D/3+1)^2`, and their ratio exceeds two on this range.
   Consequently

   .. math::

      \mathbb P(J\leq a)
       \leq\sum_{j=0}^a2^{j-b}\frac{W_{D,b}}{Z_D}
       \leq2^{a+1-b}\leq4\,2^{-D/12}.

   On :math:`j>a`, harmonic-number bounds in
   :eq:`risk-tetrahedron-vertex-weight` give uniformly

   .. math::

      G_j=\frac{\zeta(2)}{(j+1)(j+3)}
        +O(\log^2D/D^3).

   Put :math:`r(t)=1/((t+1)(t+3))`. Between :math:`\mu` and any
   :math:`j>a`, :math:`r''(t)=O(D^{-4})`. Taylor's theorem, the
   mean-zero linear term, the variance bound, and the exponentially small
   omitted tail therefore give

   .. math::

      \mathbb E[r(J)]=r(\mu)+O(D^{-3})
        =4/D^2+O(D^{-3}).

   Both :math:`G_j` and :math:`r(j)` are bounded by an absolute constant
   for all :math:`j\geq0`; the lower tail contributes exponentially little.
   Hence :math:`\mathbb E[G_J]=2\pi^2/(3D^2)+O(\log^2D/D^3)`.
   The outer prefactor is :math:`72/D^3+O(D^{-4})`, proving the result.

Verification and scope
------------------------

``tools/risk_research/vertex_single.py`` evaluates
:eq:`risk-tetrahedron-vertex-single` with exact rational arithmetic and
incremental binomial weights. Tests compare the pre-existing scalar sum
through degree 20 and independent full Gram propagation through degree 6.
After fixing the formula, the original integer-kernel implementation generated
new held-out values at degrees 241, 247 and 260, beyond the previous
exploration through degree 240. All three exact fractions agree; degree 260
is retained as an independent regression value. Finite checks verify the
implementation; the all-degree claims use the proofs above.

The formula and its positivity avoid the severe alternating cancellation of
the original scalar sum. :doc:`skeleton_risk_vertex_recurrence` supplies the
explicit scalar degree recurrence, while
:doc:`skeleton_risk_tetrahedron_comparison` classifies the optimal tetrahedron
comparison at every degree. The nonlinear self-identification question
remains separate, and the generic P-recursiveness theorem in
:doc:`skeleton_risk_extensions` continues to apply independently.
