Five-vertex risk comparison at every degree
===========================================

Under the uniform-simplex design, evaluation and noise assumptions of
:doc:`skeleton_risk_theory`, the nonvertex boundary-lift bound from
:doc:`skeleton_risk_codimension_three` closes the five-vertex comparison.
Put

.. math::

   R_{\mathrm{ind}}(D)=\left(\sum_{m=0}^4\sqrt{a_m(5,D)}\right)^2,
   \qquad R_{\mathrm{all}}(D)=\binom{D+4}4.

Inactive coefficients are zero. The common positive noise and output
factors cancel. The conclusion concerns asymptotic statistical risk;
it is not a finite-sample risk-ordering claim.

.. prf:theorem:: Complete five-vertex comparison
   :label: thm-risk-five-vertex-comparison

   For every integer :math:`D\geq1`, optimal inductive skeleton fitting
   on five vertices has strictly smaller asymptotic risk precisely when

   .. math::

      D\in\{1,2\}\quad\text{or}\quad D\geq12.

   All-at-once fitting strictly wins at :math:`3\leq D\leq11`.
   Equality never occurs. The finite prefix is computer-assisted.

.. prf:proof::

   We first prove a strict inductive win for every :math:`D\geq64`,
   then certify every smaller degree with exact rational radical bounds.
   We make no monotonicity or single-crossing assumption.

   **Vertex.** The pointwise estimate in the proof of
   :prf:ref:`thm-risk-five-vertex-explicit-error` writes, at
   :math:`n=j+1`,

   .. math::

      \Psi_j=\frac{48(H_j^{(2)}-A_j)}{n(n+2)}+R_j,\qquad
      |R_j|\leq\frac{192H_j+229}{n^3}.

   Use :math:`H_j\leq n`, :math:`H_j^{(2)}<2` and :math:`A_j\geq0`
   to obtain :math:`0<\Psi_j\leq517/n^2\leq517`.
   Vandermonde's identity normalizes the positive weights in
   :prf:ref:`thm-risk-five-vertex-single`. Hence

   .. math::

      a_0(5,D)\leq\frac{600\cdot517}
         {(2D+1)(2D+2)(2D+3)(2D+4)}
         <\frac{140^2}{D^4}.

   This estimate is valid at every positive degree.

   **Edge.** The all-facet inclusion-exclusion lift in
   :prf:ref:`thm-risk-five-vertex-edge-upper` gives

   .. math::

      a_1(5,D)<\frac{30000}{D^2}<\frac{174^2}{D^2}
          \qquad(D\geq2).

   The inactive degree-one coefficient is zero.

   **Triangle face.** In :eq:`risk-codimension-two-mode`, put
   :math:`s=3`, :math:`j=L+1` and :math:`N=2D+4`.
   The nonnegative bracket is at most :math:`H_{j+1}` and
   :math:`(N-j)(N-j+1)\geq D^2`. The multiplicity :math:`k+1`
   is at most :math:`D`. Since

   .. math::

      \sum_{j\geq1}\frac{H_{j+1}}{j(j+1)}=2,

   summing the mode bound and multiplying by :math:`\binom53^2`
   proves

   .. math::

      a_2(5,D)\leq\frac{4800}{D}<\frac{70^2}{D}.

   To verify the infinite sum, write
   :math:`H_{j+1}=H_j+1/(j+1)`.
   The first resulting sum is :math:`\zeta(2)` by exchanging the
   positive sums; the second is :math:`2-\zeta(2)` by partial fractions.

   **Facet.** Reindex :prf:ref:`thm-skeleton-codimension-one` with
   :math:`s=4` and :math:`j=D-3-k`. For :math:`D\geq4`,

   .. math::

      a_3(5,D)=50\sum_{j=1}^{D-3}
        \frac{(D-j-1)(D-j-2)}{j(2D+5-j)}
        \leq25\sum_{j=1}^{D-3}\frac{D-j}{j}
        \leq25D(1+\log D).

   Indeed, :math:`2D+5-j\geq2(D-j)` and both numerator factors
   are positive and at most :math:`D-j`.
   The function :math:`2t^{1/4}-1-\log t` has its minimum at
   :math:`t=16`, where it equals :math:`3-4\log2>0`.
   Here :math:`\log2<3/4` follows by integrating the strict secant
   bound for :math:`1/t` on :math:`[1,2]`. Therefore

   .. math::

      \sqrt{a_3(5,D)}<\sqrt{50}\,D^{5/8}
                         <\frac{15}2D^{5/8}.

   **Infinite tail.** Set :math:`b=a_4(5,D)=\binom{D-1}4` and
   :math:`X=\sum_{m=0}^3\sqrt{a_m(5,D)}`. For :math:`D\geq64`,
   :math:`D^{3/8}>19/4`, since :math:`64^3\cdot4^8>19^8`,
   and :math:`\sqrt D\geq8`. The four preceding bounds give

   .. math::

      \frac XD
        &<\frac{15}2D^{-3/8}+174D^{-2}
                   +70D^{-3/2}+140D^{-3}\\
        &\leq Q=\frac{30}{19}+\frac{174}{64^2}
                   +\frac{70}{64\cdot8}+\frac{140}{64^3}
            =\frac{2189881}{1245184}.

   Also :math:`\sqrt b\leq D^2/\sqrt{24}<5D^2/24`.
   Consequently

   .. math::

      \frac{R_{\mathrm{ind}}-b}{D^3}
        =\frac{2\sqrt bX+X^2}{D^3}
        <\frac5{12}Q+\frac{Q^2}{64}
        =\frac{232531119030803}{297692773220352}<\frac56.

   In contrast, direct polynomial expansion gives

   .. math::

      R_{\mathrm{all}}-b
        =\binom{D+4}4-\binom{D-1}4
        =\frac56D^3+\frac{25}6D>\frac56D^3.

   Thus inductive fitting strictly wins for every :math:`D\geq64`.

   **Finite prefix.** For each :math:`1\leq D\leq63`, evaluate
   the positive vertex sum, original scalar edge formula, single angular
   triangle-face sum, facet sum and top coefficient over the rationals.
   The integer square-root enclosures from :doc:`skeleton_risk_comparisons`
   separate every comparison at 32 fractional bits. The complete loss set
   is :math:`\{3,4,\ldots,11\}`, proving the theorem.

Reproduction
-------------

The optional checkout tool needs only the standard library by default:

.. code-block:: console

   $ python -m tools.risk_research.five_vertex_comparison --output .research/skeleton-risk/five-vertex-comparison.json

Pass ``--backend flint`` to use an already installed ``python-flint``.
Both backends clear the kernel denominators before exact integer
convolution. The certificate includes all 63 finite coefficient rows,
strict radical margins, the rational tail margin and hashes of all
evaluation sources. Overlap, a possible equality, or changed sources
raises an error and preserves any existing output file.
