Six-vertex risk comparison at every degree
==========================================

Under the uniform-simplex design, evaluation and noise assumptions of
:doc:`skeleton_risk_theory`, put

.. math::

   R_{\rm ind}(D)=\left(\sum_{m=0}^5\sqrt{a_m(6,D)}\right)^2,
   \qquad R_{\rm all}(D)=\binom{D+5}5.

Inactive coefficients are zero. Common positive noise and output factors
cancel. These are asymptotic statistical risks, not finite-sample risks.

.. prf:theorem:: Complete six-vertex comparison
   :label: thm-risk-six-vertex-comparison

   For every integer :math:`D\geq1`, optimal inductive skeleton fitting
   on six vertices has strictly smaller asymptotic risk exactly when

   .. math::

      D\in\{1,2\}\quad\text{or}\quad D\geq17.

   All-at-once fitting strictly wins at :math:`3\leq D\leq16`.
   There are no ties. The finite prefix is computer-assisted.

.. prf:proof::

   We prove the entire tail :math:`D\geq128` analytically, and certify
   every smaller degree over exact rational square-root intervals.
   No monotonicity or single-crossing assumption is made.

   **Vertex.** Write :math:`G(x,y)=g(x)+g(y)-g(xy)`, where
   :math:`g(t)=-(1-t)\log(1-t)/t`. Its constant is one, its two axes
   have coefficients :math:`-1/(j(j+1))`, and its positive diagonal
   has coefficients :math:`1/(j(j+1))`. Thus its absolute coefficient
   sum is four. The transformed diagonal :math:`\delta_5(j)` is a
   rectangular partial sum of :math:`G^5`, so
   :math:`0<\delta_5(j)\leq4^5`. The positive normalized binomial
   average in :doc:`skeleton_risk_vertex_general` gives

   .. math::

      a_0(6,D)\leq\frac{36\cdot120\cdot4^5}
         {\prod_{h=1}^5(2D+h)}
         <\frac{138240}{D^5}<\frac{372^2}{D^5}.

   **Codimension-two bounds used below.** In the exact mode formula
   of :doc:`skeleton_risk_extensions`, put :math:`j=L+1` and
   :math:`N=2D+s+1`. Its nonnegative bracket is at most
   :math:`H_{j+1}`, and its denominator is
   :math:`j(j+1)(N-j)(N-j+1)\geq j(j+1)D^2`.
   The exact telescoping sum

   .. math::

      \sum_{j\geq1}\frac{H_{j+1}}{j(j+1)}=2

   is proved in :doc:`skeleton_risk_codimension_three`.
   At :math:`s=3`, the angular multiplicity is at most :math:`D`;
   at :math:`s=4`, it is at most :math:`D^2/2`. Including allocation
   factors gives, on their active domains,

   .. math::

      a_2(5,D)<\frac{100\cdot24\cdot2}{D}
                   =\frac{4800}D,\qquad
      a_3(6,D)<225\cdot40\cdot\frac12\cdot2=9000<95^2.

   **Edge.** Apply the nonvertex boundary-transfer inequality with
   :math:`s=2,q=4`. The embedding ratios from source sizes
   :math:`2,3,4,5` into six vertices are at most, respectively,

   .. math::

      \frac{15}{2D^4},\quad\frac{15}{2D^3},\quad
      \frac5{D^2},\quad\frac5{2D}.

   The source coefficients are bounded by
   :math:`D`, :math:`9`, :math:`864/D^2`, :math:`30000/D^2`.
   The triangle bound and the two latter bounds are proved in
   :doc:`skeleton_risk_codimension_three` and its cited tetrahedron
   comparison. Before the common allocation factor fifteen, the
   subset-size weights are :math:`1,4/3,1,2/5`.
   The four root constants are thus bounded by
   :math:`3,11,66,110`, with the third having exponent :math:`D^{-2}`
   and the others :math:`D^{-3/2}`. Therefore

   .. math::

      \sqrt{a_1(6,D)}
         <\frac{1860}{D^{3/2}}+\frac{990}{D^2}.

   **Triangle.** Apply the same transfer with :math:`s=3,q=3`.
   Its common allocation factor is twenty and its size weights are
   :math:`1,3/4,3/10`. The embedding bounds are
   :math:`15/(2D^3),5/D^2,5/(2D)`.
   The top coefficient on three vertices is at most :math:`D^2/2`.
   The tetrahedron facet is at most :math:`24(1+\log D)`:
   its exact sum has numerator :math:`D-j-1`, denominator at least
   :math:`2j(D-j)`, and common factor forty-eight.
   Use the preceding :math:`4800/D` bound for the third source.
   Rounding the root constants upward yields

   .. math::

      \sqrt{a_2(6,D)}<\frac{39}{\sqrt D}
           +\frac{165\sqrt{1+\log D}}D+\frac{658}D.

   Indeed, the three constants before rounding are
   :math:`10\sqrt{15}`, :math:`15\sqrt{120}` and
   :math:`6\sqrt{12000}`.

   **Facet.** Reindex the exact codimension-one formula by
   :math:`j=D-4-k`. For :math:`D\geq5`,

   .. math::

      a_4(6,D)=180\sum_{j=1}^{D-4}
         \frac{\binom{D-j-1}3}{j(2D+6-j)}
         \leq15D^2(1+\log D).

   Bound the numerator by :math:`(D-j)^3/6` and the second
   denominator factor below by :math:`2(D-j)`.

   **The entire tail.** Let :math:`L(D)=1+\log D`,
   :math:`b=a_5(6,D)=\binom{D-1}5` and
   :math:`X=\sum_{m=0}^4\sqrt{a_m(6,D)}`. Then

   .. math::

      \frac X{D^{3/2}}<
        \sqrt{\frac{15L(D)}D}+\frac{95}{D^{3/2}}
        +\frac{39}{D^2}+\frac{165\sqrt{L(D)}+658}{D^{5/2}}
        +\frac{1860}{D^3}+\frac{990}{D^{7/2}}+\frac{372}{D^4}.

   Every term on the right decreases for :math:`D\geq128`;
   for the logarithmic terms this follows by differentiating
   :math:`L(D)/D` and :math:`L(D)/D^5`.
   The elementary bound :math:`\log2<3/4` gives
   :math:`L(128)<25/4`, so :math:`\sqrt{L(128)}<5/2`.
   Also :math:`\sqrt{128}>11` and
   :math:`\sqrt{15L(128)/128}<7/8`.
   Thus :math:`X/D^{3/2}<Q`, where the following number is rational:

   .. math::

      Q=\frac78+\frac{95}{128\cdot11}+\frac{39}{128^2}
        +\frac{165\cdot5}{2\cdot128^2\cdot11}
        +\frac{658}{128^2\cdot11}+\frac{1860}{128^3}
        +\frac{990}{128^3\cdot11}+\frac{372}{128^4}.

   Since :math:`1/\sqrt{120}<11/120`, one has
   :math:`\sqrt b<(11/120)D^{5/2}`. Exact rational arithmetic gives

   .. math::

      \frac{R_{\rm ind}-b}{D^4}
         <\frac{11}{60}Q+\frac{Q^2}{128}<\frac14.

   In contrast, direct expansion gives

   .. math::

      R_{\rm all}-b
        =\binom{D+5}5-\binom{D-1}5
        =\frac{D^4}4+\frac{15D^2}4+2>\frac{D^4}4.

   This proves strict inductive victory at every :math:`D\geq128`.

   **Finite prefix.** At every :math:`1\leq D\leq127`, evaluate the
   general positive vertex sum, positive diagonal edge and triangle sums,
   codimension-two angular sum, facet sum and top coefficient exactly.
   Integer square-root intervals separate all comparisons. The entire
   loss set is :math:`\{3,4,\ldots,16\}`, with no possible equality.
   This completes the classification.

Reproduction
--------------

The optional checkout tool uses standard-library positive diagonal
averages by default; see :doc:`skeleton_risk_codimension_three_exact`.
The optional ``--backend flint`` independently evaluates the original
scalar edge/triangle sums using integer kernels. Both paths reproduce
all 127 exact coefficient rows.

.. code-block:: console

   $ python -m tools.risk_research.six_vertex_comparison --output .research/skeleton-risk/six-vertex-comparison.json
   $ python -m tools.risk_research.six_vertex_comparison --verify .research/skeleton-risk/six-vertex-comparison.json

The certificate records all 127 coefficient rows, radical intervals,
strict margins, the separate analytic tail and all evaluation-source
fingerprints. Verification uses the largest recorded per-row bit count
as its effective precision budget and recomputes every field. Certificates
generated with a nondefault ``--max-bits`` use the same plain ``--verify``
command; precision overrides are generation options only. Source hashes
canonicalize CRLF pairs to LF, while concurrent edits are detected by
comparing original raw bytes. Partial coverage, overlapping intervals,
altered sources or a possible equality cause failure before replacing
an existing output. The result classifies six vertices only.
