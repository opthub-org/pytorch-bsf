# Skeleton-risk research tools

This checkout contains optional tools for generating exact risk coefficients,
searching for recurrences, and checking rational telescoping certificates.
The existing mathematical references in `tests/sampling.py` remain the two
independent scalar-sum and Gram-propagation oracles. These tools add no runtime
dependencies. The accompanying sampling fix rejects nonfinite propagation
matrices before estimating their condition number.

Run commands from the repository root. Use the project's development Python
environment for the exact runner, SymPy, and tests. Use the separate SageMath
environment for Ore algebra. No Mathematica installation is needed.

## Install the symbolic environment

With Conda already installed:

```sh
.venv/bin/python -m tools.risk_research.setup_environment
```

Pass `--conda /path/to/conda` if it is not on `PATH`. The default installation
root is `.research/skeleton-risk`; `--root PATH` selects another location.
The installer uses conda-forge, pins SageMath 10.6, Python 3.12, and Cython
3.0.12, and builds
Ore algebra at commit `afbe4dad50bd63f5b2ce112193555d6f9740e0d3` from its
[official repository](https://github.com/mkauers/ore_algebra). This combination
retains the Arb Cython headers used by that Ore algebra revision. An initial
trial with SageMath 10.9 failed because those headers had moved.
The Ore revision also precedes an upstream multivariate-ring regression
that prevented creative telescoping in the tested newer revision, and a
FLINT 3.4-only change incompatible with SageMath 10.6's FLINT 3.2 headers.
Cython matches the host version in the Sage binary's build recipe; newer
Cython produced incompatible native-extension type signatures in testing.

The installer writes `conda-explicit.txt` with the exact platform-specific
package URLs, `ore-source.json` with the upstream revision, and
`sage-smoke.json` with the actual versions and mathematical check results.
It keeps the Conda cache and Sage state under the installation root.
On the same operating system and architecture, restore the Conda packages
with `conda create --prefix NEW_PREFIX --file conda-explicit.txt`, then
install the recorded Ore algebra revision in that environment.

The smoke checks cover exact arithmetic, recurrence guessing with unseen
central-binomial values, creative telescoping of the binomial theorem,
128-bit real-ball containment, all four compiled extension imports, and
the enclosed solution of `y' = y, y(0) = 1` at one. They verify the tools, rather than establish
a new risk theorem. Rerun them directly with:

```sh
DOT_SAGE="$PWD/.research/skeleton-risk/sage-state" \
  conda run --no-capture-output -p .research/skeleton-risk/sage-env \
  python tools/risk_research/sage_smoke.py \
  --output .research/skeleton-risk/sage-smoke.json
```

## Acquire searchable references

```sh
.venv/bin/python -m tools.risk_research.fetch_resources
```

The catalogue in `resources.json` identifies the authors' or publishers'
public URLs and distinguishes full books, preprints, and samples. The
downloaded PDFs, extracted text, and acquisition manifest stay under
`.research/skeleton-risk/resources`, outside version control. Each manifest
entry records its URL, SHA256, size, and available page/text metadata.
Downloads are atomic and resumed by checksum. Poppler's `pdfinfo` and
`pdftotext`, when available, inspect the PDFs and produce searchable text.

The acquisition obtained these fifteen searchable public PDFs:

| Reference | Scope | Pages |
| --- | --- | ---: |
| Petkovsek, Wilf, Zeilberger, *A=B* | Complete author-hosted book | 217 |
| *A=B* reprint errata, March 2006 | Author-hosted errata | 5 |
| Koutschan, *Creative Telescoping for Holonomic Functions* | Author preprint | 24 |
| Kauers, Jaroschek, Johansson, *Ore Polynomials in Sage* | Author preprint/tutorial | 16 |
| Xu, *Orthogonal polynomials of several variables* | Survey; separate from the monograph | 60 |
| Flajolet, Sedgewick, *Analytic Combinatorics* | Complete author-hosted book | 826 |
| Dunkl, Xu, *Orthogonal Polynomials of Several Variables*, second edition | Publisher frontmatter and contents | 17 |
| Kauers, *Algorithms for D-Finite Functions* (2015) | Prose course notes | 30 |
| *Creative Telescoping* long course (2023): Tuesday, Sister Celine's Method | Lecture slides | 173 |
| Same course: Wednesday, Creative Telescoping | Lecture slides | 143 |
| Same course: Thursday, Advanced Closure Properties | Lecture slides | 191 |
| Same course: Friday, Conclusion | Lecture slides | 79 |
| Kauers, *Guessing with Little Data* (2022) | Talk slides | 111 |
| Kauers, *Desingularization of Ore Operators* (2014) | Talk slides | 109 |
| Kauers, *Inequalities* (2010) | Talk slides | 844 |

The seven new slide PDFs include incremental Beamer overlays; their PDF
page counts are not counts of distinct slides. The Friday file is Kauers's
conclusion and extensions, rather than every lecture delivered on Friday.

The *A=B* errata required Playwright with headed Google Chrome using the
`browser_url` recorded in `resources.json`. Direct HTTP clients and headless
browsers returned 403. The PDF was saved through the browser PDF viewer's
download button. The downloader reuses its verified local checksum but does
not launch a browser automatically if that file is missing.

The JNCF lecture notes and selected additional materials are linked from
[Kauers's talks page](http://www.kauers.de/talks.html). The working course-notes
URL is `http://www.algebra.uni-linz.ac.at/people/mkauers/publications/kauers15n.pdf`.
The older JNCF conference endpoint timed out. These author links use HTTP;
the affected catalogue entries explicitly set `allow_http: true`. Other
entries continue to require HTTPS, including redirects. The downloader
exits nonzero if an entry remains unavailable; verified PDFs remain usable.

For the current risk questions, start with the 2015 prose notes. The 2023
Tuesday and Wednesday slides supply recurrence-elimination and telescoping
methods for the remaining scalar sums; Thursday covers useful closure
operations. *Guessing with Little Data* supports exploratory recurrence
searches, while *Desingularization* addresses leading-coefficient zeros and
exceptional indices. *Inequalities* is relevant to risk comparisons after
the coefficient formulas are obtained. These methods are research tools;
they do not resolve the open risk questions by themselves.

The catalogue does not download the complete Kauers *D-Finite Functions*
or Dunkl-Xu monograph. Publisher links and access status are recorded in
`resources.json`; samples are not presented as full books. User-supplied
copies stay separately in these local, ignored paths:

- `.research/skeleton-risk/resources/user-supplied/kauers-d-finite-functions.pdf`
- `.research/skeleton-risk/resources/user-supplied/dunkl-xu-orthogonal-polynomials-2nd-edition.pdf`

Search the acquired material, for example:

```sh
rg -n 'telescop|Jacobi|simplex' .research/skeleton-risk/resources/*.txt
```

## Generate exact data and resume experiments

```sh
.venv/bin/python -m tools.risk_research generate \
  --vertices 4 --degrees 1:12 --skeletons 0,1 --oracle scalar \
  --timeout 90 --output .research/skeleton-risk/results/tetrahedron-scalar.json

.venv/bin/python -m tools.risk_research generate \
  --vertices 4 --degrees 1:4 --skeletons 0,1 --oracle gram \
  --role validation --timeout 120 \
  --output .research/skeleton-risk/results/tetrahedron-gram.json
```

Degree ranges are inclusive. Fractions are stored as decimal numerator and
denominator strings. Every case runs in a subprocess with a wall-clock limit;
`--memory-mib N` additionally requests the OS address-space limit. Some OSes
do not support that limit or need considerable virtual memory for PyTorch;
such a worker failure is recorded as an error, not as a computed value.

Only successful exact results are cached. Source-content and environment
fingerprints invalidate stale entries, including uncommitted source changes.
Workers check the fingerprint before and after calculation, so an edit during
a calculation makes that case retryable. Records include the Git revision,
dirty state, fingerprint, oracle, runtime, and exploration/validation role.
The output is saved after each case. Rerun the same command to resume.

## Search for a recurrence

For a small known-family demonstration, generate curve vertex coefficients
and fit a first-order recurrence with quadratic polynomial coefficients:

```sh
.venv/bin/python -m tools.risk_research generate \
  --vertices 2 --degrees 1:20 --skeletons 0 --oracle scalar \
  --output .research/skeleton-risk/results/curve-scalar.json

.venv/bin/python -m tools.risk_research guess \
  .research/skeleton-risk/results/curve-scalar.json \
  --skeleton 0 --order 1 --poly-degree 2 --train-through 12 \
  --output .research/skeleton-risk/results/curve-recurrence.json
```

The result describes `sum_j p_j(D) a(D+j) = 0`, with polynomial coefficients
in ascending powers of `D`. Training and held-out windows use disjoint
degree sets. Singular indices, initial values, and missing restart values
are explicit. A candidate with successful finite checks is still labeled
`conjecture`; a failed held-out check is `rejected`. Neither label supplies
an all-degree proof. The manifest's role field is descriptive; recurrence
validation is determined by the explicitly disjoint index sets.

## Check a symbolic certificate

```sh
.venv/bin/python -m tools.risk_research.certificates \
  tools/risk_research/examples/triangle_vertex_certificate.json
```

This independently verifies the existing triangle-vertex telescoping step:
the weight shift ratio, the normalized rational identity, denominator
nonvanishing, and both boundary values, including degree one. It returns
`certified_identity` only for those claims; it does not certify the entire
risk theorem or an unrelated guessed recurrence.

The JSON schema supports integer `d >= degree_min` and
`lower <= h <= d-upper_offset`, rational expressions in `d,h`, and binomial
weights. Expressions use a restricted arithmetic parser. The denominator
sign criterion is sufficient, not complete: unsupported certificates are
rejected even when a stronger proof could establish them.

## Validate the harness

```sh
.venv/bin/python -m pytest tests/tools/risk_research -q -o addopts=''
```

The tests include independent oracle agreement, cache invalidation and
retry, source edits during a computation, false recurrences, missing initial
values, certificate/boundary mutations, and download corruption repair.
Sage smoke checks are separate from the normal development test suite.
The triangular-face symbolic verifier has native-Sage regression tests.
Run them from the project Python while selecting the separate Sage interpreter:

```sh
BSF_SAGE_PYTHON="$PWD/.research/skeleton-risk/sage-env/bin/python" \
DOT_SAGE="$PWD/.research/skeleton-risk/sage-state" \
  .venv/bin/python -m pytest tests/tools/risk_research/test_triangle_recurrence.py \
  -q -o addopts=''
```

These tests verify the valid operator, reject altered coefficients and initial
values, and reproduce all operator coefficients by symbolic state elimination.
Without `BSF_SAGE_PYTHON`, only these optional symbolic tests are skipped;
ordinary exact, parser and finite-residual tests still run without Sage.

## Proved risk extensions

The [nonvertex boundary-transfer theorem](../../docs/advanced/skeleton_risk_codimension_three.rst)
bounds every nonvertex face by the same source face on smaller simplices.
In particular `a_1(5,D)<30000/D^2` for every `D>=2`, with zero at degree1.
`codimension_three.five_vertex_edge_upper_bound(D)` evaluates a stronger
rational upper bound using only the standard library. This is an inequality,
not an exact codimension-three evaluation. The CLI emits source-bound bound data. The optional symbolic checker verifies
the finite rational and harmonic identities used in the proof.

The derivations and their limits are in
[Further evaluations of the skeleton risk](../../docs/advanced/skeleton_risk_extensions.rst).
They include a codimension-two mode reduction, the all-degree tetrahedron
edge representation, its certified scalar recurrences, a general vertex
boundary-pair reduction, and fixed-dimension P-recursiveness. The
five-vertex triangular-face coefficient now also has an eight-state
representation and an order-eight, degree-76 scalar recurrence.
[Tetrahedron vertex risk](../../docs/advanced/skeleton_risk_vertex.rst)
gives a positive harmonic-binomial single sum and the leading constant
`48*pi^2/D^5`; its explicit order-ten, degree-69 recurrence is proved in
[Tetrahedron vertex degree recurrence](../../docs/advanced/skeleton_risk_vertex_recurrence.rst).

Generate tetrahedron edge coefficients from the proved auxiliary recurrences:

```sh
.venv/bin/python -m tools.risk_research.edge_closed --max-degree 600 \
  --output .research/skeleton-risk/results/tetrahedron-edge-proved.json
```

Verify the complete polynomial operator over the exact rational function
field, rather than testing finitely many sequence values:

```sh
DOT_SAGE="$PWD/.research/skeleton-risk/sage-state" \
  conda run --no-capture-output -p .research/skeleton-risk/sage-env \
  python -m tools.risk_research.edge_recurrence \
  --verify tools/risk_research/examples/tetrahedron_edge_recurrence_order7.json \
  --report .research/skeleton-risk/results/edge-order7-certificate.json
```

The order-seven operator has polynomial degree 67; the alternative
`tetrahedron_edge_recurrence_order8.json` has order eight and degree 40.
Both are proved for every positive integer degree, with explicit initial
values and no positive-integer leading-coefficient exceptions. The verifier
reconstructs the eight-dimensional rational state, checks all eight
symbolic residuals, and records the operator and source hashes.

The five-vertex triangular-face operator is verified in the same environment:

```sh
DOT_SAGE="$PWD/.research/skeleton-risk/sage-state" \
  conda run --no-capture-output -p .research/skeleton-risk/sage-env \
  python -m tools.risk_research.triangle_recurrence \
  --verify tools/risk_research/examples/five_vertex_triangle_recurrence_order8.json \
  --report .research/skeleton-risk/results/triangle-order8-certificate.json
```

`triangle_faces.triangle_face_coefficients(max_degree)` evaluates the
fixed-state sequence, and `vertex_single.tetrahedron_vertex_coefficient(D)`
evaluates the tetrahedron vertex single sum, both using standard Python
exact fractions.

Generate complete fixed-degree comparison certificates, including the exact
dimension polynomials, finite radical enclosures and an infinite-tail bound:

```sh
.venv/bin/python -m tools.risk_research.comparison --degrees 9:20 \
  --output .research/skeleton-risk/results/comparisons-9-20.json
```

Every vertex count is classified for those degrees; there are no ties.
See [the proof and complete table](../../docs/advanced/skeleton_risk_comparisons.rst).
The same proved construction now gives the complete classifications for degrees
21 through 30. Use `--degrees 21:30` to reproduce that extension, including its
finite radical certificates and infinite tails. Separate Newton interpolation
of the original scalar reference reproduces all 255 new polynomial rows and
all ten classifications; no uniform crossover law in the degree is inferred.
The independent fixed-tetrahedron comparison over every positive degree is
reproduced with:

```sh
.venv/bin/python -m tools.risk_research.tetrahedron_comparison \
  --output .research/skeleton-risk/results/tetrahedron-comparison.json
```

The finite part checks degrees 1 through 624; an analytic inequality proves
the rest. Inductive fitting wins at degrees 1, 2 and every degree at least 8;
all-at-once fitting wins at degrees 3 through 7.

The [five-vertex all-degree comparison](../../docs/advanced/skeleton_risk_five_vertex_comparison.rst)
is also complete. Inductive fitting wins at degrees1,2 and every degree>=12;
all-at-once wins exactly at degrees3..11, with no ties. All63 finite
comparisons below64 use exact coefficients and rational radical bounds;
a separate analytic inequality proves the entire infinite tail.

```sh
python -m tools.risk_research.five_vertex_comparison \
  --output .research/skeleton-risk/five-vertex-comparison.json
```

The default uses the standard library; `--backend flint` optionally speeds
up kernel multiplication. Unresolved signs or changing sources preserve
any existing certificate and cause an error.

For larger exact exploratory datasets, `research_exact.py` clears kernel
denominators and multiplies integer polynomials using the optional FLINT
backend. `research_scan.py` screens specified recurrence spaces modulo a
prime before exact reconstruction. Full modular rank excludes only the
recorded order/degree space; a successful finite fit remains a conjecture
until a separate certificate proves it. `vertex_boundary.py` is a bounded
exact reference for the proved vertex reduction, not a completed closed form.
The experiment and scan reports record `peak_rss_bytes` where OS resource
accounting is available, and JSON `null` otherwise (including Windows).

## Leading constants at large vertex counts

`vertex_large_dimension` proves an explicit large-M approximation to
`C_M`, with relative error at most `25/(M-2)` for `M>=1026`.
It provides standard-library rational intervals and an optional independent
Arb integral. This concerns the leading constant, not a finite-degree
coefficient interval. See [the proof](../../docs/advanced/skeleton_risk_vertex_large_dimension.rst).

## Further vertex evaluations and explicit errors

`vertex_general.vertex_coefficient(M,D)` constructs every vertex coefficient
by exact univariate convolutions. For each fixed `M>=4`, the leading term is
`C_M/D^(M+1)`, with a strictly positive integral formula for `C_M`.
`vertex_general.leading_constant_enclosure(M,terms)` bounds that constant by
two exact fractions using a proved series-tail majorant. Both functions use
only the standard library; the limit is taken at fixed M. See
[the general proof](../../docs/advanced/skeleton_risk_vertex_general.rst).

For every `M>=4,D>=256`, the finite-degree error is additionally bounded
by `R_M*(1+log(D))^(M-2)/D^(M+2)` with an explicit computable `R_M`.
This bound shrinks relatively at each fixed M; the constants are conservative.
`vertex_general_asymptotic` combines the constant interval and analytic
remainder, uses the exact evaluator below 256, and optionally adds Arb
rounding with restored precision. See
[the error proof](../../docs/advanced/skeleton_risk_vertex_general_asymptotics.rst).

```sh
python -m tools.risk_research.vertex_general_asymptotic --vertices 6 \
  --degree 100000 --relative-target 1/100 --output .research/skeleton-risk/vertex6-error.json
```

```sh
.venv/bin/python -m tools.risk_research.vertex_general --vertices 6 \
  --constant-terms 128 --output .research/skeleton-risk/results/vertex6-constant.json
```

`vertex_five.five_vertex_coefficient(D)` evaluates the five-vertex vertex
coefficient using four harmonic states and one sum of length D. The exact
formula, a positivity proof for every transformed kernel power, and its
`7200*(zeta(2)-zeta(3))/D^6` leading term are in
[Five-vertex vertex risk](../../docs/advanced/skeleton_risk_five_vertex.rst).
For every integer `D>=256`, its leading-term error is at most
`3000000*(1+log(D))/D^7`, as proved in
[the quantitative chapter](../../docs/advanced/skeleton_risk_five_vertex_asymptotics.rst).
`vertex_five_asymptotic` supplies an exact rational error majorant and an
optional Arb enclosure. Its bound-only CLI works without python-flint:

```sh
.venv/bin/python -m tools.risk_research.vertex_five_asymptotic --degree 100000 \
  --bound-only --output .research/skeleton-risk/results/vertex5-error.json
```

The tetrahedron vertex now has a proved next asymptotic term and an explicit
error bound for every D>=256. Its
[quantitative proof](../../docs/advanced/skeleton_risk_vertex_asymptotics.rst)
supplies both the constant and the start of the valid domain. Generate a
rational error majorant without any optional numerical dependency:

```sh
.venv/bin/python -m tools.risk_research.vertex_asymptotic --degree 10000 \
  --bound-only --output .research/skeleton-risk/results/vertex-error.json
```

Without `--bound-only`, run this module in the prepared Sage environment,
which includes python-flint. It evaluates pi, Euler's constant and the
logarithm as real balls and returns an enclosure including the analytic error.
Native regression tests can be run from the project Python with:

```sh
BSF_FLINT_PYTHON="$PWD/.research/skeleton-risk/sage-env/bin/python" \
  .venv/bin/python -m pytest tests/tools/risk_research/test_vertex_asymptotic.py \
  -q -o addopts=''
```

The native tests compare the returned ball to the existing exact single sum
through D4096 and check that precision settings are restored. If python-flint
is installed in the project Python it is used directly; otherwise those
native tests are skipped unless an explicit native interpreter is selected.

### Refined single-log vertex errors

`vertex_refined_asymptotic` proves an explicit single-log error for every
`M>=5,D>=256`, combines it with the existing guarantee, and keeps exact
evaluation below 256. At M6 its 1% relative threshold is `2^38`, improving
on the earlier `2^56`. The thresholds remain conservative.
See [the proof](../../docs/advanced/skeleton_risk_vertex_refined_asymptotics.rst).

## Further certified degree recurrences

The nonvertex codimension-two family also has one common fifteen-state
construction for every source-face size `s>=2`, with `M=s+2`. It removes
the degree-length angular sum for every such face, includes the inactive
degree prefix, and handles both half-range parities. This is a uniform
state bound; the smaller eight-state cases remain useful, and no minimal
order or exception-free eliminated scalar operator is inferred. See
[the general proof](../../docs/advanced/codimension_two_family.rst).

```sh
.venv/bin/python -m tools.risk_research.codimension_two_family \
  --face-size 5 --max-degree 50 --output .research/skeleton-risk/results/family-m7.json
```

The evaluator needs only exact standard-library fractions. The optional
`codimension_two_family_certificate` module runs in the prepared Sage
environment to verify a selected symbolic output construction and its
domain, with both evaluator and verifier source hashes recorded.

The tetrahedron vertex operator has order ten and polynomial degree 69.
Its proof reduces to four rational telescoping identities with finite
endpoints and no poles on the integer domain. The next codimension-two
coefficient, `a_3(6,D)`, has eight rational states and an order-eight,
degree-83 operator. Both cover every positive integer degree, with explicit
initial values and no exceptional integer indices.

Use the prepared Sage interpreter to verify either supplied artifact:

```sh
python -m tools.risk_research.vertex_degree_recurrence \
  --verify tools/risk_research/examples/tetrahedron_vertex_recurrence_order10.json
python -m tools.risk_research.codimension_two_states \
  --verify tools/risk_research/examples/six_vertex_tetrahedral_face_recurrence_order8.json
```

The first module also supplies `recurrence_coefficients(maximum)` for
standard-library exact evaluation. The second supplies
`tetrahedral_face_coefficients(maximum)` using its fixed state, avoiding
the original degree-dependent angular sum. Their native Sage regressions
are selected by `BSF_SAGE_PYTHON` as for the triangular-face verifier above.
The proofs and domain checks are in the new recurrence chapters.

The five-vertex vertex also has an explicit order-seventeen, degree-188
operator, proved by eight rational telescoping identities. All 189
coefficients of its leading polynomial are positive, and all 280 certificate
parameter denominators have positive domain certificates. It is valid for
every positive integer degree with seventeen initial values and no exceptions.
See [the proof](../../docs/advanced/skeleton_risk_five_vertex_recurrence.rst).
Using the prepared Sage interpreter:

```sh
python -m tools.risk_research.vertex_five_recurrence \
  --verify tools/risk_research/examples/five_vertex_vertex_recurrence_order17.json
```

`vertex_five_recurrence.recurrence_coefficients(maximum)` evaluates the
sequence with exact standard-library arithmetic. Optional native Sage tests
rederive every coefficient and exercise altered data and both CLI actions.

## A sufficient linear vertex threshold at every degree

For every integer `D>=1` and `M>=9720*D`, optimal inductive asymptotic
risk is strictly below four fifths of all-at-once risk. This conservative
region follows from positive-kernel and factorial bounds, without generating
a separate dimension polynomial at each degree. It is not an exact crossover.
See [the proof](../../docs/advanced/skeleton_risk_uniform_dimension_tail.rst).

```sh
python -m tools.risk_research.uniform_dimension_tail --degree 31 \
  --output .research/skeleton-risk/uniform-tail31.json
python -m tools.risk_research.uniform_dimension_tail \
  --verify .research/skeleton-risk/uniform-tail31.json
```

The standard-library tool records exact specialized arithmetic, the domain
and source fingerprints. Its `source_hash_policy` is `sha256-crlf-to-lf-v1`:
each CRLF pair becomes LF before hashing, with every other byte retained.
The raw-byte before/after guard still rejects any source edit during
certification, including a line-ending change. A finite set of its
certificates checks the implementation; the documented inequalities prove
the all-degree theorem.
