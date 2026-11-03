A classical leading constant for seven vertices
===============================================

Keep the uniform-simplex assumptions of :doc:`skeleton_risk_theory`.
This chapter evaluates the next constant from
:doc:`skeleton_risk_vertex_general`. It extends the finite harmonic
telescope of :doc:`skeleton_risk_c5_closed_form` to weight five.

.. prf:theorem:: The sixth constant
   :label: thm-risk-c6-closed-form

   .. math::

      c_6=600\zeta(2)+2520\zeta(3)+13980\zeta(4)
            -4320\zeta(2)\zeta(3)-10080\zeta(5)>0.

   Consequently, for the fixed seven-vertex simplex,

   .. math::

      a_0(7,D)\sim\frac{C_7}{D^8},\qquad C_7=2205c_6.

   This is a leading constant as the degree tends to infinity. It is
   neither a finite-degree error estimate nor an all-degree comparison.

.. prf:proof::

   Put :math:`H=H_{n-1}`, :math:`G=H_{n-1}^{(2)}` and
   :math:`J=H_{n-1}^{(3)}`, with all three zero at :math:`n=1`.
   The existing elementary-symmetric coefficient formula gives

   .. math::

      \phi_1(n-1)&=\frac1n,\\
      \phi_2(n-1)&=\frac{2(1-H)}{n(n+1)},\\
      \phi_3(n-1)&=\frac{6(H^2-G-3H+1)}{n(n+1)(n+2)},\\
      \phi_4(n-1)&=\frac{12P(H,G,J)}{n(n+1)(n+2)(n+3)},\\
      P(H,G,J)&=-2H^3+6HG-4J+11H^2-11G-12H+2.

   Thus :math:`e_5(n-1)=10\phi_1\phi_4+20\phi_2\phi_3`, and
   :math:`t_n=e_5(n-1)/120` is

   .. math::

      t_n=\frac{P(H,G,J)}{n^2(n+1)(n+2)(n+3)}
          +\frac{2(1-H)(H^2-G-3H+1)}{n^2(n+1)^2(n+2)}.

   Define :math:`U(n,H,G,J)=V(n,H,G,J)/(36n^2(n+1)(n+2))`, where

   .. math::

      V={}&-(120n^3+420n^2+440n+144)H^3\\
          &+(144n^3+504n^2+528n+144)HG
             -(24n^3+84n^2+88n)J\\
          &+(-78n^3-183n^2+54n+144)H^2\\
          &+(42n^3+111n^2+18n)G
             -(30n^3+84n^2+72n)H.

   Direct rational simplification proves the finite identity

   .. math::
      :label: risk-c6-finite-telescope

      t_n={}&U(n+1,H+1/n,G+1/n^2,J+1/n^3)-U(n,H,G,J)\\
             &+\frac{-20H^3+24HG-4J+47H^2-17G+21H+5}{6n^2}.

   This is an identity in the independent variables :math:`n,H,G,J`:
   after multiplying by :math:`36n^3(n+1)^2(n+2)(n+3)`, its numerator
   has every polynomial coefficient zero. Hence it holds at every
   positive integer n; it is not inferred from sampled harmonic values.

   The first boundary is :math:`U(1,0,0,0)=0`. The other boundary tends
   to zero, because :math:`H_{n-1}\leq1+\log n`,
   :math:`G,J\leq2` and :math:`U=O((1+\log n)^3/n)`.
   The displayed residual series converges absolutely by comparison
   with :math:`(1+\log n)^3/n^2`. The original e5 series is absolutely
   convergent by the general coefficient bound. Summing the finite
   identity and only then taking its limit is therefore justified.
   No divergent harmonic series is assigned a value.

   Use descending indices
   :math:`\zeta(k_1,\ldots,k_r)=\sum_{n_1>\cdots>n_r\geq1}
   n_1^{-k_1}\cdots n_r^{-k_r}`. All values below have
   :math:`k_1=2`, hence converge. Finite ordered-index decompositions give

   .. math::

      \sum_{n\geq1}\frac{H^3}{n^2}
         &=6\zeta(2,1,1,1)+3\zeta(2,1,2)
              +3\zeta(2,2,1)+\zeta(2,3),\\
      \sum_{n\geq1}\frac{HG}{n^2}
         &=\zeta(2,1,2)+\zeta(2,2,1)+\zeta(2,3),\\
      \sum_{n\geq1}\frac{J}{n^2}&=\zeta(2,3),\\
      \sum_{n\geq1}\frac{H^2}{n^2}
         &=2\zeta(2,1,1)+\zeta(2,2),\\
      \sum_{n\geq1}\frac{G}{n^2}&=\zeta(2,2),\qquad
      \sum_{n\geq1}\frac{H}{n^2}=\zeta(2,1).

   Since :math:`c_6=6\sum e_5(j)=720\sum t_n`, this proves the finite
   convergent MZV expression

   .. math::

      c_6=120\bigl(&-120\zeta(2,1,1,1)-36\zeta(2,1,2)
                    -36\zeta(2,2,1)\\
                  &+94\zeta(2,1,1)+30\zeta(2,2)
                    +21\zeta(2,1)+5\zeta(2)\bigr).

   The iterated-integral duality proved in
   :doc:`skeleton_risk_c5_closed_form` reverses the binary word and
   interchanges 0 and 1. Here it sends 01111 to 00001, 01101 to
   01001, and 01011 to 00101, proving respectively

   .. math::

      \zeta(2,1,1,1)=\zeta(5),\quad
      \zeta(2,1,2)=\zeta(2,3),\quad
      \zeta(2,2,1)=\zeta(3,2).

   Splitting the absolutely convergent product into the cases where
   its two summation indices are greater, smaller or equal gives
   :math:`\zeta(2,3)+\zeta(3,2)=\zeta(2)\zeta(3)-\zeta(5)`.
   The same earlier chapter proves
   :math:`\zeta(2,1,1)=\zeta(4)`,
   :math:`\zeta(2,1)=\zeta(3)` and
   :math:`\zeta(2,2)=3\zeta(4)/4`. Substitution proves the displayed
   classical formula. These are finite identities among convergent
   sums and integrals; no numerical integer-relation fit is used.

   The original positive integral proves :math:`c_6>0`, independently
   of cancellation in the final expression. Substituting q=6 in the
   existing normalization gives
   :math:`C_7=7^2\,6!\,2^{-4}c_6=2205c_6`, proving the consequence.

Rational evaluation and saved records
-------------------------------------

The evaluator reuses the positive Euler-transformed ordinary zeta
intervals proved in :doc:`skeleton_risk_c5_closed_form`. Their tail after
N terms is at most :math:`2^{-N}/(1-2^{1-k})` for order k. All orders
needed here are fixed between 2 and 5. The rational-operation count is
linear in N, while the integer bit lengths grow.

For positive intervals :math:`l_i\leq\zeta(i)\leq u_i`, the product is
between :math:`l_2l_3` and :math:`u_2u_3`. The coefficient -4320
therefore reverses those endpoints, as does the coefficient -10080
of zeta(5). The linear positive terms use their ordinary endpoints.
The old positive-integral theorem permits clamping the lower c6
endpoint to zero. No floating-point approximation is required.

.. code-block:: console

   $ .venv/bin/python -m tools.risk_research.c6_closed_form --terms 64 --output .research/skeleton-risk/c6.json
   $ .venv/bin/python -m tools.risk_research.c6_closed_form --verify .research/skeleton-risk/c6.json

The new record contains exact rational c6/C7 intervals, the finite
classical and descending-index MZV coefficients, saved terms and source
hashes. Verification restores nondefault terms and compares the complete
strictly typed record. Source hashes normalize CRLF to LF; raw bytes
still guard against concurrent source changes. Failed generation or
verification preserves existing output. Protected evaluator/proof paths
cannot be overwritten. Output uses a checked canonical parent and
original leaf in a directory controlled by the caller, not a sandbox
against another process replacing that directory inode.

The finite identity and source-independent elementary-symmetric rows
are checked exactly. After freezing the theorem, independent finite
checks compare the original general-vertex rational series and original
positive Arb integral with this new expression. These checks validate
implementation and normalization; the finite telescope and convergent
identities above supply the proof. This chapter does not assert an
arbitrary-q classical reduction, an optimal MZV basis, or M7's
all-degree comparison classification.
