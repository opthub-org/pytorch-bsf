Quadratic codimension-two multiplicities
========================================

The next case after edges on four vertices and triangular faces on five
vertices is a tetrahedral face on six vertices. Its angular multiplicity
is quadratic. The following result removes that degree-dependent angular
sum and gives an eight-dimensional rational state for every degree.

These are exact results about the risk coefficient :math:`a_3(6,D)`.
They do not extend the public sampling API or assert a state bound for
arbitrary face size.
The separate construction in :doc:`codimension_two_family` now proves a
common fifteen-state bound for arbitrary nonvertex codimension-two face
size. The eight-state formula here remains a smaller representation for
the six-vertex case.

The odd half-range state
------------------------

Write :math:`H_n^{(2)}=\sum_{j=1}^n j^{-2}` and use the previously proved
inverse-binomial sequences

.. math::

   F_n=\sum_{j=1}^n\frac1{j^2\binom nj},\qquad
   G_n=\sum_{j=1}^n\frac1{j\binom nj},

with :math:`F_0=G_0=0` and

.. math::

   G_{n+1}=\frac{G_n}{2}+\frac1{n+1},\qquad
   F_{n+1}=F_n-\frac{G_n}{n+1}+\frac1{(n+1)^2}.

For :math:`m\ge1`, set :math:`N=2m+1` and

.. math::

   C_m&=H_N^2-H_N^{(2)},& U_m&=F_N-H_N^{(2)},\\
   w_m&=\binom{2m+1}{m}^{-1},&
   T_m&=\sum_{j=1}^{m}\frac1{j^2\binom Nj},\\
   J_m&=H_m^{(2)}-T_m+\frac{U_m}{2}.

The defining sum for :math:`T_m` is used in the proof and independent
checks. Evaluation uses the first-order rule

.. math::
   :label: risk-odd-half-state

   J_{m+1}=J_m+\frac{1-3w_m/4}{(m+1)^2},\qquad
   w_{m+1}=\frac{m+2}{2(2m+3)}w_m,
   \qquad J_1=\frac14,\quad w_1=\frac13.

With :math:`a=(N+1)^{-1}+(N+2)^{-1}`,
:math:`b=(N+1)^{-2}+(N+2)^{-2}` and :math:`q=(N+1)(N+2)`, the full state

.. math::

   v_m=(C_m,H_N,H_m,U_m,G_N,J_m,w_m,1)^{\mathsf T}

has the remaining transitions

.. math::

   C_{m+1}&=C_m+2aH_N+a^2-b,\\
   H_{N+2}&=H_N+a,\qquad H_{m+1}=H_m+\frac1{m+1},\\
   U_{m+1}&=U_m-\frac{3N+5}{2q}G_N-\frac1q,\\
   G_{N+2}&=\frac{G_N}{4}+\frac1{2(N+1)}+\frac1{N+2}.

Its initial vector at :math:`m=1` is

.. math::

   v_1=\left(2,\frac{11}{6},1,-\frac56,\frac56,
                   \frac14,\frac13,1\right)^{\mathsf T}.

All denominators in these transitions are positive for :math:`m\ge1`.

An all-degree coefficient formula
---------------------------------

Use the universal codimension-two summand from
:prf:ref:`thm-risk-codimension-two-modes`. For :math:`1\le j<N`, define

.. math::

   \Delta_j&=j(j+1)(N-j)(N-j+1),\\
   B_j&=H_{j+1}+H_{N-j+1}-H_N,\\
   c_j&=\frac{(N+1)(j(N-j)-1)}{\Delta_j},\\
   f_N(j)&=\frac{B_j-c_j}{\Delta_j},\\
   v_N(j)&=\frac1{j^2(j+1)(N-j)^2(N-j+1)\binom Nj},\\
   g_N(j)&=f_N(j)-v_N(j).

Here :math:`v_N(j)` is a scalar summand, distinct from the state vector
:math:`v_m`. The summands are symmetric under :math:`j\mapsto N-j`.

.. prf:theorem:: Eight states for a tetrahedral face on six vertices
   :label: thm-risk-six-vertex-tetrahedral-state

   For every integer :math:`D\ge1`, put :math:`m=D+2`, :math:`N=2m+1`
   and evaluate the state above. Define

   .. math::
      :label: risk-odd-half-moments

      S_m&=\frac{C_m}{N(N+1)(N+2)}
       -\frac{N^2+N+2}{N^2(N+1)(N+2)}U_m
       -\frac{2(N-2)}{N^3(N+2)}(H_N-G_N),\\
      X_m&=\frac{C_m+2U_m+4(H_N-G_N)/N}{2(N+2)},\\
      E_m&=\frac{H_N-2H_m}{2(m+1)^2}
                 +\frac{J_m}{N}-\frac{w_m}{2(m+1)^3}.

   Then

   .. math::
      :label: risk-six-vertex-tetrahedral-state

      a_3(6,D)=9000\left[
          \frac{N^2+63}{8}S_m-\frac{X_m}{2}-4E_m
          -g_N(m-2)-3g_N(m-1)-6g_N(m)\right].

   Every boundary term is a rational linear combination of the same eight
   states. The expression is nonsingular for every :math:`D\ge1`; it
   returns zero at the inactive degrees :math:`D=1,2,3`.

The first positive values, at :math:`D=4,5,6,7,8`, are

.. math::

   \frac{375}{13},\quad \frac{6900}{91},\quad
   \frac{7375}{56},\quad\frac{493125}{2584},\quad
   \frac{81423005}{325584}.

.. prf:proof::

   Reindex :eq:`risk-codimension-two-coefficient` with source-face size
   :math:`s=4` by :math:`j=L+1`. This gives, for :math:`D\ge4`,

   .. math::

      a_3(6,D)=9000\sum_{j=1}^{D-3}
         \frac{(D-2-j)(D-1-j)}2\,g_N(j).

   Let :math:`x=j(N-j)` and :math:`p=N/2-j`. The quadratic weight has the
   exact decomposition

   .. math::

      \frac{(D-2-j)(D-1-j)}2
        =\frac{N^2+63}{8}-\frac{x}{2}-4p.

   Extending the sum to :math:`j=m` adds only the three nonzero terms
   with weights :math:`1,3,6` at :math:`m-2,m-1,m`; the intervening two
   weights vanish. Thus it suffices to prove

   .. math::

      S_m=\sum_{j=1}^m g_N(j),\quad
      X_m=\sum_{j=1}^m j(N-j)g_N(j),\quad
      E_m=\sum_{j=1}^m (N/2-j)g_N(j).

   The already proved full-range formulas for :math:`f_N` and :math:`v_N`
   in :doc:`skeleton_risk_extensions` give
   :math:`S_m=Q_N-V_N/2` by symmetry; collecting
   :math:`H_N^2-H_N^{(2)}` and :math:`F_N-H_N^{(2)}` gives precisely the
   first line of :eq:`risk-odd-half-moments`. The rational constants cancel.

   The two new weighted moments are proved below. For a boundary index
   :math:`j=m-r`, :math:`r=0,1,2`, the harmonic part is :math:`2H_m-H_N`
   plus a fixed number of rational increments. Also

   .. math::

      \binom{N}{m-r}^{-1}
        =w_m\prod_{i=1}^{r}\frac{m+i+1}{m-i+1}.

   Hence these boundary values use the stated state. Since :math:`m\ge3`,
   all their denominators are positive. At the inactive degrees the
   half-range polynomial has support exactly on the three subtracted
   terms, so the same formula gives zero directly.

The symmetric quadratic moment
------------------------------

Multiplying :math:`f_N(j)` by :math:`x=j(N-j)` gives the harmonic factor
:math:`1/((j+1)(N-j+1))`. The elementary convolution identities

.. math::

   \sum_{k=1}^{r}\frac{H_k}{k}
       =\frac{H_r^2+H_r^{(2)}}2,\qquad
   \sum_{k=1}^{r-1}\frac{H_{r-k}}{k}
       =H_r^2-H_r^{(2)}

evaluate its full-range sum. The rational part is reduced by

.. math::

   \frac{x c_j}{\Delta_j}
   ={}&\frac1{N+2}\left[\frac1{(j+1)^2}
                            +\frac1{(N-j+1)^2}\right]\\
   &+\frac{3N+4}{(N+1)(N+2)^2}
             \left[\frac1{j+1}+\frac1{N-j+1}\right]\\
   &-\frac1{N(N+1)}\left[\frac1j+\frac1{N-j}\right].

Consequently

.. math::

   \sum_{j=1}^{N-1}x f_N(j)
     =\frac{H_N^2-3H_N^{(2)}+4H_N/N+2(N-2)/N^2}{N+2}.

For the inverse-binomial part put :math:`b_j=\binom Nj^{-1}` and

.. math::

   P(j)&=-\frac{N+1}{N+2}-\frac{2j}{N(N+2)}
       +\frac{6j^2}{N(N+2)}-\frac{4j^3}{N(N+1)(N+2)},\\
   R(j)&=\frac{P(j)}{j^2(N-j+1)}.

The rational identity

.. math::
   :label: risk-quadratic-weight-gosper

   \frac1{\Delta_j}
      =\frac{j+1}{N-j}R(j+1)-R(j)
       +\frac8{N(N+1)(N+2)}
       -\frac1{N+2}\left[\frac1{j^2}+\frac1{(N-j)^2}\right]

can be checked by multiplying denominators. After multiplication by
:math:`b_j`, the first two terms telescope because
:math:`b_{j+1}/b_j=(j+1)/(N-j)`. Using
:math:`\sum_{j=1}^{N-1} b_j=(N+1)G_N/2-1` gives

.. math::

   \sum_{j=1}^{N-1}x v_N(j)
      =\frac{-2F_N+4G_N/N+2(N-2)/N^2}{N+2}.

The difference of these full-range formulas is twice :math:`X_m`, since
:math:`N` is odd and the summands are symmetric. This proves its stated
formula without introducing another inverse-binomial state.

The antisymmetric moment
------------------------

For the harmonic term put
:math:`r_j=1/[j(N-j+1)]` and :math:`t_j=1/[j(N-j+2)]`. The identities

.. math::

   \frac{N/2-j}{\Delta_j}=\frac{r_j-r_{j+1}}2,\qquad
   r_j(B_j-B_{j-1})=t_j-t_{j+1}

give by summation by parts

.. math::

   \sum_{j=1}^m\frac{(N/2-j)B_j}{\Delta_j}
     =\frac1N-\frac{B_m}{2(m+1)^2}
                  -\frac1{2(m+1)(m+2)}.

For the rational part one has

.. math::

   \frac{(N/2-j)c_j}{\Delta_j}
   ={}&\frac{N+3}{2(N+1)^2}
       \left[\frac1j-\frac1{N-j}-\frac1{j+1}+\frac1{N-j+1}\right]\\
   &-\frac1{2N(N+1)}\left[\frac1{j^2}-\frac1{(N-j)^2}\right]\\
   &-\frac1{2(N+1)}
       \left[\frac1{(j+1)^2}-\frac1{(N-j+1)^2}\right].

Summing these differences of first- and second-order harmonic terms and
using :math:`N=2m+1` yields

.. math::

   \sum_{j=1}^m(N/2-j)f_N(j)
      =\frac{H_N-2H_m}{2(m+1)^2}
       +\frac{H_m^{(2)}-H_N^{(2)}/2}{N}
       -\frac{N-1}{2N^3}.

The inverse-binomial telescope, valid for either parity of :math:`N`, is

.. math::

   (N/2-j)v_N(j)
    =\frac{b_{j+1}}{2(j+1)^2(N-j)}
       -\frac{b_j}{2j^2(N-j+1)}
       +\frac{b_j}{2N}\left[\frac1{j^2}-\frac1{(N-j)^2}\right].

The odd half-range has :math:`b_m=b_{m+1}=w_m`. Its sum is

.. math::

   \sum_{j=1}^m(N/2-j)v_N(j)
      =\frac{T_m-F_N/2}{N}+\frac{w_m}{2(m+1)^3}
                                    -\frac{N-1}{2N^3}.

Subtracting proves the formula for :math:`E_m`.

Proof of the new state transition
---------------------------------

The full inverse-binomial moment and the first-order telescope give

.. math::

   \sum_{j=1}^{m}b_j&=\frac{N+1}{4}G_N-\frac12,\\
   \sum_{j=1}^{m}\frac{b_j}{j}
       &=\frac{G_N}{2}-\frac{w_m}{2(m+1)}.

For the second identity, telescope
:math:`b_j/j-b_j/(N-j)=b_j/j-b_{j+1}/(j+1)` and use symmetry. The change
of total degree satisfies

.. math::

   \frac{\binom{N+2}{j}^{-1}}{\binom Nj^{-1}}
       =\frac{(N+2-j)(N+1-j)}{(N+1)(N+2)}.

Divide by :math:`j^2`, sum to :math:`m`, and include the new endpoint
:math:`j=m+1`. Substitution of the preceding moments gives

.. math::

   T_{m+1}-T_m
     =-\frac{3N+5}{4(N+1)(N+2)}G_N
       -\frac1{2(N+1)(N+2)}+\frac{3w_m}{4(m+1)^2}.

Half the increment of :math:`U_m` is exactly the first two terms. They
therefore cancel in :math:`J_m=H_m^{(2)}-T_m+U_m/2`, proving
:eq:`risk-odd-half-state`. This also proves the rational state system and
completes the all-degree construction.

A scalar recurrence certificate
-------------------------------

``tools/risk_research/codimension_two_states.py`` evaluates the exact
coefficients without Sage. Its optional symbolic commands eliminate the
eight states over :math:`\mathbb Q(D)` and independently rebuild the state
rows to verify a supplied operator. The checked-in result is
``tools/risk_research/examples/six_vertex_tetrahedral_face_recurrence_order8.json``.
It has order eight and maximum polynomial degree 83:

.. math::

   \sum_{k=0}^8 p_k(D)a_3(6,D+k)=0\qquad(D\ge1).

The leading polynomial has no positive integer zero. The verifier computes
all rational roots exactly and filters positive integers, checks positive
state denominators, and verifies the eight rational identities and initial
values. Thus the operator and degrees one through eight determine the
entire positive-degree sequence. No claim of minimal scalar order is made.

To regenerate and verify it in a Sage Python environment:

.. code-block:: console

   python -m tools.risk_research.codimension_two_states --output /tmp/six-vertex.json
   python -m tools.risk_research.codimension_two_states --verify /tmp/six-vertex.json

The tests compare the coefficient with the original scalar kernel sum,
independent rational Gram propagation, and the angular-mode sum at held-out
degrees through 300. Separate symbolic checks verify the new partial
fractions, binomial-ratio reduction and Gosper certificate. These finite
checks guard the implementation; the identities and state proof establish
the all-degree statements.

For larger fixed source-face sizes, the angular multiplicity has higher
polynomial degree. Its symmetric and antisymmetric decomposition is a
useful next reduction, but closure in these same eight states has not been
proved here.
