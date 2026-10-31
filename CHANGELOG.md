# Changelog

## [0.25.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.24.0...v0.25.0) (2026-10-31)


### ⚠ BREAKING CHANGES

* **deps:** Require Python 3.11 or later and NumPy 1.23.2 or later. Recreate Python 3.10 environments before upgrading.

### Features

* **sampling:** generalize skeleton fitting to simploids ([#591](https://github.com/opthub-org/pytorch-bsf/issues/591)) ([0e79909](https://github.com/opthub-org/pytorch-bsf/commit/0e79909394ffb01024513a35a4efedf56f835fc8))
* **security:** automate alert repairs and reviewed merges ([#592](https://github.com/opthub-org/pytorch-bsf/issues/592)) ([7404818](https://github.com/opthub-org/pytorch-bsf/commit/74048185d33b9322d86b41c5b84273b7ef5d3c9a))


### Bug Fixes

* **deps:** remove vulnerable legacy release fixture dependencies ([#590](https://github.com/opthub-org/pytorch-bsf/issues/590)) ([4aa9825](https://github.com/opthub-org/pytorch-bsf/commit/4aa9825bee2e21408bb1ce2baf4eaf22da194cf1))
* **deps:** secure dependencies and support Python 3.11-3.15 ([#571](https://github.com/opthub-org/pytorch-bsf/issues/571)) ([a9aa35b](https://github.com/opthub-org/pytorch-bsf/commit/a9aa35b8537d39ec332b70f68ab1afca22a60872))
* **sampling:** check simploid conditioning before allocating ([#596](https://github.com/opthub-org/pytorch-bsf/issues/596)) ([7e481c4](https://github.com/opthub-org/pytorch-bsf/commit/7e481c45a60599dbbd450b7557af31270b345342))


### Performance Improvements

* **core:** vectorize control adjacency construction ([#604](https://github.com/opthub-org/pytorch-bsf/issues/604)) ([0849e60](https://github.com/opthub-org/pytorch-bsf/commit/0849e60004ed5f73f964c9d40c577510d073191e))
* **sampling:** compute simploid skeleton risk factor by factor ([#597](https://github.com/opthub-org/pytorch-bsf/issues/597)) ([87ad6d7](https://github.com/opthub-org/pytorch-bsf/commit/87ad6d785713c494b33d02dae1806749badd5f9c))
* **sampling:** reduce simplex risk propagation and contractions ([#603](https://github.com/opthub-org/pytorch-bsf/issues/603)) ([a2066cd](https://github.com/opthub-org/pytorch-bsf/commit/a2066cd3e3db395d546576e834628065dc230cca))


### Documentation

* **interpolation:** clarify Hermite continuity and PCHIP slopes ([#556](https://github.com/opthub-org/pytorch-bsf/issues/556)) ([9e39ad0](https://github.com/opthub-org/pytorch-bsf/commit/9e39ad0dfd510bbcf09d272ccdb23571b9a2bf5c))
* **sampling:** extend asymptotic risk to simploidal sets ([#599](https://github.com/opthub-org/pytorch-bsf/issues/599)) ([60781a5](https://github.com/opthub-org/pytorch-bsf/commit/60781a5ef1a126d4b3fffa602470401ef255c47a))
* **sampling:** preserve product risk proofs and closed forms ([#605](https://github.com/opthub-org/pytorch-bsf/issues/605)) ([50faa4d](https://github.com/opthub-org/pytorch-bsf/commit/50faa4d0634735e4827046bcd22a4053ec3e7cd7))
* **sampling:** prove simploid skeleton risk formulas ([#598](https://github.com/opthub-org/pytorch-bsf/issues/598)) ([39e8e2e](https://github.com/opthub-org/pytorch-bsf/commit/39e8e2eb502b4b6fba43cda233ba5a8ea8bf58a7))

## [0.24.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.23.0...v0.24.0) (2026-10-29)


### Features

* **interpolation:** add C1 Clough-Tocher macro-element interpolant ([#529](https://github.com/opthub-org/pytorch-bsf/issues/529)) ([50bb3a9](https://github.com/opthub-org/pytorch-bsf/commit/50bb3a940e9f1f54f577ee1c9425d2a822911eec))
* **interpolation:** add newest-vertex bisection to refine ([#484](https://github.com/opthub-org/pytorch-bsf/issues/484)) ([1eb05c4](https://github.com/opthub-org/pytorch-bsf/commit/1eb05c49292867e38af5e9dc6b506271913e3c95))
* **sampling:** add optimal skeleton sampling and inductive regression ([#516](https://github.com/opthub-org/pytorch-bsf/issues/516)) ([f3f606d](https://github.com/opthub-org/pytorch-bsf/commit/f3f606d4bc9e887949cd905b5b5abb09aec87e0a))


### Bug Fixes

* **active_learning:** preserve nearby farthest-point distances ([#480](https://github.com/opthub-org/pytorch-bsf/issues/480)) ([20ba304](https://github.com/opthub-org/pytorch-bsf/commit/20ba3049b0bf549983fe4239d372f99e2c61df69))
* **active_learning:** stabilize committee variance rankings ([#481](https://github.com/opthub-org/pytorch-bsf/issues/481)) ([567f067](https://github.com/opthub-org/pytorch-bsf/commit/567f067145240d56b281f6f69ba2536d88c53e6a))
* **core:** preserve MPS inputs during CPU float64 conversion ([#542](https://github.com/opthub-org/pytorch-bsf/issues/542)) ([00af52c](https://github.com/opthub-org/pytorch-bsf/commit/00af52c567c98c7500a3a7a113768dc5cd235cf0))
* **interpolation:** preserve finite refinement scores ([#544](https://github.com/opthub-org/pytorch-bsf/issues/544)) ([31cb4fd](https://github.com/opthub-org/pytorch-bsf/commit/31cb4fda164adacd7553d658411226006616b7b2))
* **kernels:** preserve polynomial derivatives at simplex boundaries ([#545](https://github.com/opthub-org/pytorch-bsf/issues/545)) ([eaa7942](https://github.com/opthub-org/pytorch-bsf/commit/eaa794280dfd7d8736a622562db0b8de1e08b207))
* move MPS tensors to CPU before float64 conversion ([#520](https://github.com/opthub-org/pytorch-bsf/issues/520)) ([2d2791b](https://github.com/opthub-org/pytorch-bsf/commit/2d2791b557eddd5ed21c5e0d0d46b3d73afe31b8))
* **preprocessing:** validate fitted feature counts in transforms ([#479](https://github.com/opthub-org/pytorch-bsf/issues/479)) ([1d0a636](https://github.com/opthub-org/pytorch-bsf/commit/1d0a636f15d1e675a6b79bff2917b029fb211371))
* **sampling:** generate the first Sobol point in float64 ([#519](https://github.com/opthub-org/pytorch-bsf/issues/519)) ([cef0fd6](https://github.com/opthub-org/pytorch-bsf/commit/cef0fd60c55084a6d7989b740cabcf4b7acfdd63))
* **validation:** require strings for enumerated choices ([#511](https://github.com/opthub-org/pytorch-bsf/issues/511)) ([94c5aef](https://github.com/opthub-org/pytorch-bsf/commit/94c5aef17863f2803ecc5d43d399e34a75a0b284))


### Performance Improvements

* **interpolation:** batch face-separation determinants ([#512](https://github.com/opthub-org/pytorch-bsf/issues/512)) ([e20296f](https://github.com/opthub-org/pytorch-bsf/commit/e20296f63920976a9db01dfd815d919f17736e7f))


### Documentation

* certify skeleton crossover dimensions through degree eight ([#528](https://github.com/opthub-org/pytorch-bsf/issues/528)) ([b9834dd](https://github.com/opthub-org/pytorch-bsf/commit/b9834dd1ea831bb26d57ecfb18fe25981d7f03ad))
* **cli:** document commands, CSV layouts and result contracts ([#499](https://github.com/opthub-org/pytorch-bsf/issues/499)) ([64edb5c](https://github.com/opthub-org/pytorch-bsf/commit/64edb5cd2e97db594fba7adb8804bf147e2175d3))
* **contributing:** sync Repository Structure with the tree ([#507](https://github.com/opthub-org/pytorch-bsf/issues/507)) ([5ed4bc9](https://github.com/opthub-org/pytorch-bsf/commit/5ed4bc99287be4bafa6b032bdf090eb1a3ff8963))
* **core:** explain point and sample data contracts ([#500](https://github.com/opthub-org/pytorch-bsf/issues/500)) ([268cb20](https://github.com/opthub-org/pytorch-bsf/commit/268cb20980ffb8bce81c929056eefc9e7815235a))
* derive a generating function for vertex skeleton risk ([#527](https://github.com/opthub-org/pytorch-bsf/issues/527)) ([323288f](https://github.com/opthub-org/pytorch-bsf/commit/323288f16248365b6213988ce99f5c2d96c489dc))
* derive triangular Jacobi extension for skeleton errors ([#526](https://github.com/opthub-org/pytorch-bsf/issues/526)) ([2ec3a28](https://github.com/opthub-org/pytorch-bsf/commit/2ec3a2872d4c016a384153ee66613c3af45be499))
* link versioned API references to matching source ([#504](https://github.com/opthub-org/pytorch-bsf/issues/504)) ([b09dea4](https://github.com/opthub-org/pytorch-bsf/commit/b09dea463498fbd9f16967a2d9298d1d03654d72))
* organize guides and align page names with their content ([#517](https://github.com/opthub-org/pytorch-bsf/issues/517)) ([99c6713](https://github.com/opthub-org/pytorch-bsf/commit/99c67135f5678cfbcb34b2a7560fa477814f3574))
* organize the root README for library users ([#547](https://github.com/opthub-org/pytorch-bsf/issues/547)) ([6b986b3](https://github.com/opthub-org/pytorch-bsf/commit/6b986b3477d381f460aaa0df9a3cc07215979645))
* **point_cloud:** verify result and boundary examples ([#502](https://github.com/opthub-org/pytorch-bsf/issues/502)) ([2ab44ee](https://github.com/opthub-org/pytorch-bsf/commit/2ab44eeb5bdf4c33c7452e581977ab53910fc1ed))
* **readme:** guide readers from inputs to runnable examples ([#501](https://github.com/opthub-org/pytorch-bsf/issues/501)) ([55aafe2](https://github.com/opthub-org/pytorch-bsf/commit/55aafe2a7d29c6edd5c0caba2d899624dc4ca946))
* **sampling:** close triangle edge risk in harmonic numbers ([843c98f](https://github.com/opthub-org/pytorch-bsf/commit/843c98ffac42c4623ff03ddbb48fb2831961d93b))
* **sampling:** consolidate skeleton risk theory and open questions ([#536](https://github.com/opthub-org/pytorch-bsf/issues/536)) ([12d1672](https://github.com/opthub-org/pytorch-bsf/commit/12d1672b214cc33bbce26f44b88f176ed792df5f))
* **sampling:** derive general scalar risk and third-face quartic ([#533](https://github.com/opthub-org/pytorch-bsf/issues/533)) ([d85fb4f](https://github.com/opthub-org/pytorch-bsf/commit/d85fb4f1d213ec00954fb1dba48326d6cede469c))
* **sampling:** prove the skeleton risk and its closed forms ([#522](https://github.com/opthub-org/pytorch-bsf/issues/522)) ([e9c28e7](https://github.com/opthub-org/pytorch-bsf/commit/e9c28e71fa0af9d41d7c654f78c496c4761f7f36))
* **sampling:** prove triangle vertex risk in harmonic numbers ([#535](https://github.com/opthub-org/pytorch-bsf/issues/535)) ([27bf1c5](https://github.com/opthub-org/pytorch-bsf/commit/27bf1c5aec8393e2bc25309f0daa1b7ad8225320))
* **sgd:** fix linear scalarization API link ([#510](https://github.com/opthub-org/pytorch-bsf/issues/510)) ([39b0ada](https://github.com/opthub-org/pytorch-bsf/commit/39b0ada0d79358efb714464b3c9e6c6d5550a452))

## [0.23.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.22.0...v0.23.0) (2026-10-28)


### ⚠ BREAKING CHANGES

* **interpolation:** SimplicialMesh raises ValueError for non-conforming meshes, which were silently accepted before. Pass strict=False to get a warning instead.

### Features

* **interpolation:** check mesh conformity, add lattice_mesh and refine ([#430](https://github.com/opthub-org/pytorch-bsf/issues/430)) ([4db3b80](https://github.com/opthub-org/pytorch-bsf/commit/4db3b80d380d7cce664de17b26070482841934a3))
* **splitting:** add conforming subdivision of a whole space ([#438](https://github.com/opthub-org/pytorch-bsf/issues/438)) ([5d01307](https://github.com/opthub-org/pytorch-bsf/commit/5d0130794fd4c8e0c62de0093760bd60b8d3dd10))


### Bug Fixes

* **ci:** test draft PRs and publish Docker only on releases ([#443](https://github.com/opthub-org/pytorch-bsf/issues/443)) ([740ae0c](https://github.com/opthub-org/pytorch-bsf/commit/740ae0cb6ef0adf1d596680115376f234de2f182))
* **core:** preserve non-tensor fixed control precision ([#437](https://github.com/opthub-org/pytorch-bsf/issues/437)) ([dc4730a](https://github.com/opthub-org/pytorch-bsf/commit/dc4730af14ac056737fefe5c205d564863d0cdcf))
* **core:** validate structural integers without coercion ([#471](https://github.com/opthub-org/pytorch-bsf/issues/471)) ([cde82a6](https://github.com/opthub-org/pytorch-bsf/commit/cde82a6f34acd23f5d2616f0ad4fb40ce6779704))
* handle extreme coordinates and add property-based tests ([#440](https://github.com/opthub-org/pytorch-bsf/issues/440)) ([341b825](https://github.com/opthub-org/pytorch-bsf/commit/341b82546f1f48f52efe5dc9889a1eb1f87323c7))
* **interpolation:** explain interpolant overflow from vertex data ([#448](https://github.com/opthub-org/pytorch-bsf/issues/448)) ([6299001](https://github.com/opthub-org/pytorch-bsf/commit/6299001a4490af97648da9682b3b1e1feda2c180))
* **interpolation:** handle vertex differences that overflow float64 ([#445](https://github.com/opthub-org/pytorch-bsf/issues/445)) ([d13306d](https://github.com/opthub-org/pytorch-bsf/commit/d13306d1aa26a10a5d237b476e4a04d564c5eeed))
* **interpolation:** read non-tensor inputs as float64 ([#434](https://github.com/opthub-org/pytorch-bsf/issues/434)) ([5d814ef](https://github.com/opthub-org/pytorch-bsf/commit/5d814ef43585a24b9976b34af0ad41e4e3661b06)), closes [#431](https://github.com/opthub-org/pytorch-bsf/issues/431)
* **interpolation:** snapshot mesh vertex storage ([#462](https://github.com/opthub-org/pytorch-bsf/issues/462)) ([581838a](https://github.com/opthub-org/pytorch-bsf/commit/581838ab83d1b3be037d964fe93fd412ed822bf3))
* **interpolation:** stabilize barycentrics and their gradients ([#442](https://github.com/opthub-org/pytorch-bsf/issues/442)) ([1d4ce4c](https://github.com/opthub-org/pytorch-bsf/commit/1d4ce4c3babe722261d6b482ffcd78efe837da9b))
* **kernels:** stabilize high-degree Bernstein evaluation ([#464](https://github.com/opthub-org/pytorch-bsf/issues/464)) ([42bc5d1](https://github.com/opthub-org/pytorch-bsf/commit/42bc5d15247973a454bfdfb9377810835388fc0a))
* **model_selection:** reject non-finite validation metrics ([#470](https://github.com/opthub-org/pytorch-bsf/issues/470)) ([1f13e94](https://github.com/opthub-org/pytorch-bsf/commit/1f13e9498e4425815ca523c36e2ba8be21a4c11f))
* **numeric:** preserve finite ensemble and constant-cloud means ([#468](https://github.com/opthub-org/pytorch-bsf/issues/468)) ([ba31efc](https://github.com/opthub-org/pytorch-bsf/commit/ba31efc32bcb3b4a1ee7139ef9c61c2c1f7ead81))
* **optimization:** reject non-finite objectives and updates ([#466](https://github.com/opthub-org/pytorch-bsf/issues/466)) ([1b7b125](https://github.com/opthub-org/pytorch-bsf/commit/1b7b125d3052e34d387afd89af8f4deca180f1fa))
* **preprocessing:** check and stabilize extreme-value scaling ([#469](https://github.com/opthub-org/pytorch-bsf/issues/469)) ([26f68e0](https://github.com/opthub-org/pytorch-bsf/commit/26f68e00ae3f7755c6c8a82ab53e18b93fb3efe3))
* **preprocessing:** restore integer MinMaxScaler compatibility ([#475](https://github.com/opthub-org/pytorch-bsf/issues/475)) ([adb036d](https://github.com/opthub-org/pytorch-bsf/commit/adb036d45927562600da54e9d3832593cce3649c))
* **splitting:** compare extreme chord lengths safely ([#465](https://github.com/opthub-org/pytorch-bsf/issues/465)) ([abc2274](https://github.com/opthub-org/pytorch-bsf/commit/abc22748de0627e204d4e92299b36ec76d3788f9))


### Performance Improvements

* **active-learning:** bound farthest-point distance blocks ([#467](https://github.com/opthub-org/pytorch-bsf/issues/467)) ([d5ed060](https://github.com/opthub-org/pytorch-bsf/commit/d5ed060e8f59ee7cfeb17a65a77273f33e4c6ca0))
* **core:** group basis rows and write outputs locally ([#472](https://github.com/opthub-org/pytorch-bsf/issues/472)) ([f680428](https://github.com/opthub-org/pytorch-bsf/commit/f68042868a2ba93fcb9afce64e93f3016a0e6d59))
* **optimization:** reuse metrics and bypass plain SGD solves ([#473](https://github.com/opthub-org/pytorch-bsf/issues/473)) ([6d17521](https://github.com/opthub-org/pytorch-bsf/commit/6d175218ac4720e58151f7af32e04118a4384b8d))


### Documentation

* **contributing:** refresh runnable API and CLI examples ([#463](https://github.com/opthub-org/pytorch-bsf/issues/463)) ([ea3980f](https://github.com/opthub-org/pytorch-bsf/commit/ea3980facc290ddd05d6958ccb44b46d1a670ac5))
* fix title underline length in point_cloud.rst ([#433](https://github.com/opthub-org/pytorch-bsf/issues/433)) ([bdd76c1](https://github.com/opthub-org/pytorch-bsf/commit/bdd76c15d898c304f93d0af95204a8e4a79895fb)), closes [#432](https://github.com/opthub-org/pytorch-bsf/issues/432)

## [0.22.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.21.0...v0.22.0) (2026-10-27)


### Features

* **interpolation:** add shape-preserving simplicial cubic interpolation ([#425](https://github.com/opthub-org/pytorch-bsf/issues/425)) ([93ec870](https://github.com/opthub-org/pytorch-bsf/commit/93ec87057f3571828b93f14fa4458f7ff3400db7))

## [0.21.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.20.0...v0.21.0) (2026-10-26)


### ⚠ BREAKING CHANGES

* **fitting:** numerical failures return unsuccessful FitResult values with ConvergenceWarning. Call require_success() for an exception retaining the result.
* **api:** remove legacy public models, fitting entry points, CLI flags, sampling and persistence aliases. Regression returns FitResult and solves the normalized original-unit objective in CPU float64. See docs/migration.rst.

### Features

* add value-only Bezier simplex fitting ([#422](https://github.com/opthub-org/pytorch-bsf/issues/422)) ([90c0f45](https://github.com/opthub-org/pytorch-bsf/commit/90c0f45d2d5250c969c390aa3edee8bad9f33446))
* **fitting:** complete dense and sparse numerical contracts ([e12a67c](https://github.com/opthub-org/pytorch-bsf/commit/e12a67c469c3cc285b208b492f6e5afc6104a021))


### Bug Fixes

* **cli:** isolate optional MLflow import failures ([#414](https://github.com/opthub-org/pytorch-bsf/issues/414)) ([23dbe03](https://github.com/opthub-org/pytorch-bsf/commit/23dbe03ac3202a04afd1b85c8c84c3dc3fd2d494))
* **sklearn:** import torch_bsf.sklearn without scikit-learn ([#412](https://github.com/opthub-org/pytorch-bsf/issues/412)) ([71fc7eb](https://github.com/opthub-org/pytorch-bsf/commit/71fc7eb7ac93794e6a0b1af2e3b88dfc35300d16))


### Documentation

* expand guides and explain simploid geometry ([#421](https://github.com/opthub-org/pytorch-bsf/issues/421)) ([eea187e](https://github.com/opthub-org/pytorch-bsf/commit/eea187e3e6b54ef9b5d8250d1a84c69357576912))
* restore detailed guides for the unified API ([#420](https://github.com/opthub-org/pytorch-bsf/issues/420)) ([3eef67d](https://github.com/opthub-org/pytorch-bsf/commit/3eef67d6fa3b271aabdf39ce4b1b098e28105f92))


### Code Refactoring

* **api:** unify spaces, models and numerical results ([#418](https://github.com/opthub-org/pytorch-bsf/issues/418)) ([cc6ec20](https://github.com/opthub-org/pytorch-bsf/commit/cc6ec20fd161a4587c36955f00ce720cd80217f6))

## [0.20.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.19.0...v0.20.0) (2026-10-22)


### ⚠ BREAKING CHANGES

* **fitting:** `pip install pytorch-bsf` no longer installs Lightning. To keep using solver="trainer" (or --solver=trainer), install the extra: pip install "pytorch-bsf[lightning]".
* **cli:** `pip install pytorch-bsf` no longer installs MLflow, so the CLI does not log to MLflow unless it is installed. To keep logging, install the extra: `pip install "pytorch-bsf[mlflow]"`.

### Dependencies

* **cli:** make MLflow an optional extra ([#402](https://github.com/opthub-org/pytorch-bsf/issues/402)) ([4c540ea](https://github.com/opthub-org/pytorch-bsf/commit/4c540eab90d769acf6e35961c41e63d88a97a96a))
* **fitting:** make Lightning an optional extra ([#401](https://github.com/opthub-org/pytorch-bsf/issues/401)) ([953b944](https://github.com/opthub-org/pytorch-bsf/commit/953b9448e304330799cca8f91e1fc890cb6a201d))


### Documentation

* **sgd:** fit the skew-MMD Pareto set via a monotone transform ([#396](https://github.com/opthub-org/pytorch-bsf/issues/396)) ([b2ba147](https://github.com/opthub-org/pytorch-bsf/commit/b2ba147e3180412e636799ef29eecccfe18015c4))

## [0.19.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.18.0...v0.19.0) (2026-10-21)


### ⚠ BREAKING CHANGES

* torch_bsf.bezier_simplex and torch_bsf.control_points are removed; use BezierSimploid, torch_bsf.fit and torch_bsf.simploidal.save/load instead. Pickled .pt models saved by earlier versions can no longer be loaded. ControlPointsLike is renamed to ControlPointData.
* torch_bsf.fit and fit_kfold return BezierSimploid models and default to solver="lstsq" (trainer options require solver="trainer"). BezierSimplex, BezierSimplexDataModule, validate_control_points and fit_simploid are no longer exported from torch_bsf. BezierSimplexRegressor is renamed to BezierSimploidRegressor and defaults to solver="lstsq". reparametrize's subsimplex argument is renamed to part, and split criteria return (factor, i, j, s). The CLI --init reads files written by torch_bsf.simploidal.save, k-fold drops --stratified, and output file names carry the simploid type (t_<type>,d_<degree>).
* **simploidal:** BezierSimploid and SimploidalBezierSpace are no longer LightningModules: they have no hparams, training_step or configure_optimizers, and cannot be passed to Trainer.fit directly. Use fit_simploid / fit_simploidal_set with solver="trainer" instead.

### Features

* add simploidal sets and simploidal Bézier space fitting ([#334](https://github.com/opthub-org/pytorch-bsf/issues/334)) ([2ae1b37](https://github.com/opthub-org/pytorch-bsf/commit/2ae1b372a7bb9cb2b109f7454a7d900635b371f3))
* **cli:** add init, freeze, smoothing and loss logging to sgd ([#394](https://github.com/opthub-org/pytorch-bsf/issues/394)) ([f633854](https://github.com/opthub-org/pytorch-bsf/commit/f633854bc17242ddabafc3645e657d77c8a16526))
* **cli:** fit Bezier simplices to Pareto sets from the CLI and MLflow ([#390](https://github.com/opthub-org/pytorch-bsf/issues/390)) ([0a6a0ec](https://github.com/opthub-org/pytorch-bsf/commit/0a6a0ec494b2c998dfb21dfd9b59b3649fb28b2e))
* **cli:** fit Bezier simploids from the CLI and MLflow ([#345](https://github.com/opthub-org/pytorch-bsf/issues/345)) ([73c2db9](https://github.com/opthub-org/pytorch-bsf/commit/73c2db948922e63eea7fa2677898224912abc6af))
* **plotting:** plot Bézier simploids and simploidal Bézier spaces ([#344](https://github.com/opthub-org/pytorch-bsf/issues/344)) ([e9a1cbd](https://github.com/opthub-org/pytorch-bsf/commit/e9a1cbdd234ec0d4745d5ad7a9926b75c3a71e3f))
* **simploidal:** add a callback to fit_sgd for monitoring and stopping ([#388](https://github.com/opthub-org/pytorch-bsf/issues/388)) ([c124e3f](https://github.com/opthub-org/pytorch-bsf/commit/c124e3f9328d35008f4e0153efdc055519ff3443))
* **simploidal:** add smoothness penalty and frozen control points ([#359](https://github.com/opthub-org/pytorch-bsf/issues/359)) ([449ed94](https://github.com/opthub-org/pytorch-bsf/commit/449ed94a60c29ed46a4a3c8929fcc618f3476826))
* **simploidal:** compute homology of simploidal sets ([#341](https://github.com/opthub-org/pytorch-bsf/issues/341)) ([3bd296b](https://github.com/opthub-org/pytorch-bsf/commit/3bd296bc268ba91f8300945043e2e5b519e1a1c3)), closes [#335](https://github.com/opthub-org/pytorch-bsf/issues/335)
* **simploidal:** fit models to Pareto sets by preconditioned SGD ([#383](https://github.com/opthub-org/pytorch-bsf/issues/383)) ([2a0a390](https://github.com/opthub-org/pytorch-bsf/commit/2a0a39045a385c915302c588311444715c5e35f7))
* **simploidal:** freeze control points and smooth in fit_sgd ([#389](https://github.com/opthub-org/pytorch-bsf/issues/389)) ([0703ffa](https://github.com/opthub-org/pytorch-bsf/commit/0703ffaee5a45cddc8d58548784277ae72555561)), closes [#385](https://github.com/opthub-org/pytorch-bsf/issues/385)
* **simploidal:** support per-factor degrees in SimploidalBezierSpace ([#342](https://github.com/opthub-org/pytorch-bsf/issues/342)) ([a21ff29](https://github.com/opthub-org/pytorch-bsf/commit/a21ff29cbd38b29844336c026ba68c6e134ad493)), closes [#338](https://github.com/opthub-org/pytorch-bsf/issues/338)


### Bug Fixes

* **deps:** test and declare the oldest working dependency versions ([#381](https://github.com/opthub-org/pytorch-bsf/issues/381)) ([a0910ce](https://github.com/opthub-org/pytorch-bsf/commit/a0910ce881d7d8952fd234163827f5ea4a62ed6c))
* **deps:** use a versioning strategy Dependabot accepts for pip ([#382](https://github.com/opthub-org/pytorch-bsf/issues/382)) ([19adee9](https://github.com/opthub-org/pytorch-bsf/commit/19adee99e59a25400e550c852031112eb8a31d00))
* **simploidal:** make random docstring comparisons deterministic ([#349](https://github.com/opthub-org/pytorch-bsf/issues/349)) ([547034e](https://github.com/opthub-org/pytorch-bsf/commit/547034eefa49ccd1e8742a5fdbecb71acbb7ff61)), closes [#346](https://github.com/opthub-org/pytorch-bsf/issues/346)


### Documentation

* add application examples with accurate Pareto front fits ([#328](https://github.com/opthub-org/pytorch-bsf/issues/328)) ([fb3b5d4](https://github.com/opthub-org/pytorch-bsf/commit/fb3b5d4b711ec9bb729654e6a14e19760017dedb))
* Adopt sphinx-design grids for Elastic Net figure layouts ([#331](https://github.com/opthub-org/pytorch-bsf/issues/331)) ([d5b75e5](https://github.com/opthub-org/pytorch-bsf/commit/d5b75e570e720060c59b8ad9cf03627081107be8))
* **bezier_simplex:** disable model summary in the training example ([#360](https://github.com/opthub-org/pytorch-bsf/issues/360)) ([5f234bf](https://github.com/opthub-org/pytorch-bsf/commit/5f234bfba4aa41e6e6cb2731c69c67b33fc88280))
* describe Bézier simploids where simplex-only wording was too narrow ([#352](https://github.com/opthub-org/pytorch-bsf/issues/352)) ([3033408](https://github.com/opthub-org/pytorch-bsf/commit/3033408a1028418fb1bb53c3e903900e9ef07438))
* **examples:** quiet Lightning output in Pareto front examples ([#351](https://github.com/opthub-org/pytorch-bsf/issues/351)) ([f3abeff](https://github.com/opthub-org/pytorch-bsf/commit/f3abeff919988e9c8ab8d8e5e26ea1623b53facc)), closes [#348](https://github.com/opthub-org/pytorch-bsf/issues/348)
* redefine BSF as Bézier Simploid Fitting ([#339](https://github.com/opthub-org/pytorch-bsf/issues/339)) ([af7fa87](https://github.com/opthub-org/pytorch-bsf/commit/af7fa87eed1601b78e69b77769845d65d2f219e6))
* refresh PyTorch-BSF visual identity ([#343](https://github.com/opthub-org/pytorch-bsf/issues/343)) ([48dbb3d](https://github.com/opthub-org/pytorch-bsf/commit/48dbb3d85ede4effb44bbf38b90368d146919f2e))
* **sgd:** compare SGD and PSGD on the benchmarks of the paper ([#391](https://github.com/opthub-org/pytorch-bsf/issues/391)) ([4223f15](https://github.com/opthub-org/pytorch-bsf/commit/4223f15d3a0aff57e0098ad43513272848161bbc))
* **sgd:** plot the MSE to the Pareto set in the SGD benchmarks ([#395](https://github.com/opthub-org/pytorch-bsf/issues/395)) ([b0f296e](https://github.com/opthub-org/pytorch-bsf/commit/b0f296e2e3f45fd788e81e2374bc1ea8fbd1e80d))
* **simploidal:** define the simploid category and prove sSpd is a topos ([#340](https://github.com/opthub-org/pytorch-bsf/issues/340)) ([f14223e](https://github.com/opthub-org/pytorch-bsf/commit/f14223ee4e3f8d1663a146a58ecbcf0dd2be0dc8))


### Code Refactoring

* **simploidal:** decouple Bézier simploid models from Lightning ([#358](https://github.com/opthub-org/pytorch-bsf/issues/358)) ([d8ac277](https://github.com/opthub-org/pytorch-bsf/commit/d8ac277532c32e3f8e771cfa67b1257007991c5c))
* unify peripheral features on BezierSimploid ([#361](https://github.com/opthub-org/pytorch-bsf/issues/361)) ([540c997](https://github.com/opthub-org/pytorch-bsf/commit/540c9977edc0cebf0e7f542b8fa68521c1438d12))
* unify save/load and remove the legacy Bézier simplex modules ([#362](https://github.com/opthub-org/pytorch-bsf/issues/362)) ([3b8e014](https://github.com/opthub-org/pytorch-bsf/commit/3b8e0147421ac9fdc14e93bf567cd6ebb6fbcc21))

## [0.18.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.17.0...v0.18.0) (2026-04-04)


### Features

* Bézier simplex splitting via de Casteljau algorithm ([#324](https://github.com/opthub-org/pytorch-bsf/issues/324)) ([cdc5750](https://github.com/opthub-org/pytorch-bsf/commit/cdc5750675586cbab67cab0a8fa982deab9016a7))


### Documentation

* add Bézier simplex splitting guide ([#327](https://github.com/opthub-org/pytorch-bsf/issues/327)) ([e8177cf](https://github.com/opthub-org/pytorch-bsf/commit/e8177cff74b3f8c2329e1ecf8031e65d69cb85a8))

## [0.17.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.16.1...v0.17.0) (2026-04-02)


### ⚠ BREAKING CHANGES

* unify `fit_kfold()` interface with `fit()` and return `nn.ModuleList` ([#314](https://github.com/opthub-org/pytorch-bsf/issues/314))

### Features

* 15 code-quality improvements — reproducibility, type safety, architecture, docs ([#318](https://github.com/opthub-org/pytorch-bsf/issues/318)) ([588334f](https://github.com/opthub-org/pytorch-bsf/commit/588334f40a9c7c3c8a89d3052ce5a630360a2fb1))
* support high-dimensional Bézier simplex plotting (n_params ≥ 4) via pairwise scatter plot ([#307](https://github.com/opthub-org/pytorch-bsf/issues/307)) ([5d43d24](https://github.com/opthub-org/pytorch-bsf/commit/5d43d2447d91cb0fb67a8ef8c920c9fc2721fab3))
* unify `fit_kfold()` interface with `fit()` and return `nn.ModuleList` ([#314](https://github.com/opthub-org/pytorch-bsf/issues/314)) ([8ac7afb](https://github.com/opthub-org/pytorch-bsf/commit/8ac7afbc92151ee25011e4d72e168c7fc3646fd2))


### Bug Fixes

* typo in error message — "sprit_ratio" → "split_ratio" ([#290](https://github.com/opthub-org/pytorch-bsf/issues/290)) ([cd165e3](https://github.com/opthub-org/pytorch-bsf/commit/cd165e34fadd1131cdb74e4e30916c4262722fd5))
* unify Bézier/Bezier spelling across source code, metadata, and CONTRIBUTING.md ([#321](https://github.com/opthub-org/pytorch-bsf/issues/321)) ([c57bf16](https://github.com/opthub-org/pytorch-bsf/commit/c57bf1643aa43c2ad87d7d38d78abee0201d6338))


### Documentation

* make README more readable — consolidate features, rebalance chapters ([#316](https://github.com/opthub-org/pytorch-bsf/issues/316)) ([5eb763a](https://github.com/opthub-org/pytorch-bsf/commit/5eb763acaf20427a089e7f82d44c046b532b98ea))
* rename 'fix' option to 'freeze' in README ([6b9d03c](https://github.com/opthub-org/pytorch-bsf/commit/6b9d03cdacd3946db33f247714343821af5d5c5d))
* start elastic net explanation from original λ/α formulation ([#310](https://github.com/opthub-org/pytorch-bsf/issues/310)) ([6fb5ed4](https://github.com/opthub-org/pytorch-bsf/commit/6fb5ed43c4579dae2f24060b580bc4b1a3dee997))

## [0.16.1](https://github.com/opthub-org/pytorch-bsf/compare/v0.16.0...v0.16.1) (2026-04-01)


### Bug Fixes

* elastic_net_grid returns empty (0, 3) when n_alphas == 0 ([#287](https://github.com/opthub-org/pytorch-bsf/issues/287)) ([99215db](https://github.com/opthub-org/pytorch-bsf/commit/99215db6f45c3f5ba354288c7598d9166eea1bc3))


### Documentation

* add documents of sampling ([#285](https://github.com/opthub-org/pytorch-bsf/issues/285)) ([8637677](https://github.com/opthub-org/pytorch-bsf/commit/8637677e31952dbfa57ec7c59fb40f303c9d37db))
* fix chapter ordering, unify Title Case headings, fix typos, and expand thin sections ([#289](https://github.com/opthub-org/pytorch-bsf/issues/289)) ([166f6c4](https://github.com/opthub-org/pytorch-bsf/commit/166f6c4fc4e093b966d08630468c37d8f3161a73))

## [0.16.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.15.1...v0.16.0) (2026-03-30)


### Features

* add pyproject.toml for uv support ([#213](https://github.com/opthub-org/pytorch-bsf/issues/213)) ([#274](https://github.com/opthub-org/pytorch-bsf/issues/274)) ([97bd22d](https://github.com/opthub-org/pytorch-bsf/commit/97bd22dc262c222c49a880bde26db1f6ea3c94e4))
* add scalers ([#166](https://github.com/opthub-org/pytorch-bsf/issues/166)) ([af5d0a3](https://github.com/opthub-org/pytorch-bsf/commit/af5d0a333b4c1fe681ad02bd34b860f017d6b16e))
* auto detect file types ([#134](https://github.com/opthub-org/pytorch-bsf/issues/134)) ([964c671](https://github.com/opthub-org/pytorch-bsf/commit/964c6717835087b9b93477cd6b9ddb553fcfaeec))
* disable checkpointing by default ([#192](https://github.com/opthub-org/pytorch-bsf/issues/192)) ([9d80470](https://github.com/opthub-org/pytorch-bsf/commit/9d80470887494d876f1f6d633663f33d4c04f46b))
* fixable control points ([#130](https://github.com/opthub-org/pytorch-bsf/issues/130)) ([e4d6c0e](https://github.com/opthub-org/pytorch-bsf/commit/e4d6c0e5d5aabadeb5f42f8c41ca8057f7a5fcf4))
* grid sampler for elastic net ([#150](https://github.com/opthub-org/pytorch-bsf/issues/150)) ([2b5da95](https://github.com/opthub-org/pytorch-bsf/commit/2b5da9501a1788671e30055dc71d8fdcd118144d))
* k-fold cross validation ([#153](https://github.com/opthub-org/pytorch-bsf/issues/153)) ([a345dfd](https://github.com/opthub-org/pytorch-bsf/commit/a345dfde5dc0ba8eb3fd8cf6ca65f314d9d2db2f))
* kfold entrypoint ([#158](https://github.com/opthub-org/pytorch-bsf/issues/158)) ([69c3d0b](https://github.com/opthub-org/pytorch-bsf/commit/69c3d0b4326e3f47aca1d761cf7be58dac3eb5ec))
* meshgrid parameter in entry points is now optional ([#226](https://github.com/opthub-org/pytorch-bsf/issues/226)) ([e82be30](https://github.com/opthub-org/pytorch-bsf/commit/e82be30c0d4a9e9392221ceac9cbf4450d0b6373))
* MLproject entrypoint for elastic net grid ([#161](https://github.com/opthub-org/pytorch-bsf/issues/161)) ([445cc52](https://github.com/opthub-org/pytorch-bsf/commit/445cc5268da83529e23c246ec8b971d39a0c8b32))
* precision flag ([#122](https://github.com/opthub-org/pytorch-bsf/issues/122)) ([3d74982](https://github.com/opthub-org/pytorch-bsf/commit/3d74982eb3c81a90150b43b36018525e692706bd))
* skeleton validator ([#124](https://github.com/opthub-org/pytorch-bsf/issues/124)) ([95767e5](https://github.com/opthub-org/pytorch-bsf/commit/95767e5766a41b4be9cce9b6c1bff35afc317b90))
* stratified splitting ([#156](https://github.com/opthub-org/pytorch-bsf/issues/156)) ([60adc3a](https://github.com/opthub-org/pytorch-bsf/commit/60adc3a01ed701206d13a2a2bda7acf0a30cb876))
* support 1D datasets ([#188](https://github.com/opthub-org/pytorch-bsf/issues/188)) ([8b3d29f](https://github.com/opthub-org/pytorch-bsf/commit/8b3d29f5bbd286a7951d21fdeb5d11c61c3b07c7))
* use pytorch-lightning v2 ([#119](https://github.com/opthub-org/pytorch-bsf/issues/119)) ([8afbf48](https://github.com/opthub-org/pytorch-bsf/commit/8afbf482539540028d14a3cd9eafac219bebd71d))
* visualization ([#247](https://github.com/opthub-org/pytorch-bsf/issues/247)) ([4a4e4a0](https://github.com/opthub-org/pytorch-bsf/commit/4a4e4a07a4211ba01de5ff123c23cbf41ba97531))
* write a file for each fold ([#170](https://github.com/opthub-org/pytorch-bsf/issues/170)) ([b00d4c3](https://github.com/opthub-org/pytorch-bsf/commit/b00d4c3807ab67e6d1963d778779903967816a7c))


### Bug Fixes

* close file handles using context managers in save() and load() ([#264](https://github.com/opthub-org/pytorch-bsf/issues/264)) ([fad6dfb](https://github.com/opthub-org/pytorch-bsf/commit/fad6dfb739afdb8cbe6e214cffd7a59372245007))
* fit() uses init and skeleton ([#127](https://github.com/opthub-org/pytorch-bsf/issues/127)) ([c92ffcf](https://github.com/opthub-org/pytorch-bsf/commit/c92ffcf2fb21a067da6868e96c671036a391dbd5))
* Fix sample codes. ([#212](https://github.com/opthub-org/pytorch-bsf/issues/212)) ([3369166](https://github.com/opthub-org/pytorch-bsf/commit/3369166ca5387e37fdaef4c5d6db3a73c723834b))
* kfold meshgrid ([979ca65](https://github.com/opthub-org/pytorch-bsf/commit/979ca65cb37c948459c0a28f22e9862106de4106))
* kfold meshgrid ([#177](https://github.com/opthub-org/pytorch-bsf/issues/177)) ([979ca65](https://github.com/opthub-org/pytorch-bsf/commit/979ca65cb37c948459c0a28f22e9862106de4106))
* load dataset ([2a8cd5b](https://github.com/opthub-org/pytorch-bsf/commit/2a8cd5bfaf2f060da217355b193d5f68755703fe))
* MLproject ([#174](https://github.com/opthub-org/pytorch-bsf/issues/174)) ([ef1d2d4](https://github.com/opthub-org/pytorch-bsf/commit/ef1d2d42d636a5535f521b7948405fcc9d3d622c))
* MLproject ([#176](https://github.com/opthub-org/pytorch-bsf/issues/176)) ([5b7afd9](https://github.com/opthub-org/pytorch-bsf/commit/5b7afd9e02b0cbf5bd29fa8d361a3edacc4eb5a3))
* MLproject can get additional params ([#163](https://github.com/opthub-org/pytorch-bsf/issues/163)) ([9ea524e](https://github.com/opthub-org/pytorch-bsf/commit/9ea524e2b82f48ddea9f493937db84b00995df06))
* pytest bug ([d8dfea7](https://github.com/opthub-org/pytorch-bsf/commit/d8dfea7b9e565ad6922eb616a61c6d4ee87237a9))
* pytest bug ([290b16d](https://github.com/opthub-org/pytorch-bsf/commit/290b16db7cb497ad40d5d9a460b4a7f656ea353a))
* remove cudatoolkit, add multi-OS conda CI ([#252](https://github.com/opthub-org/pytorch-bsf/issues/252)) ([118a841](https://github.com/opthub-org/pytorch-bsf/commit/118a841c2feb9fa20550199a4f1801c8c75a1923))
* replace deprecated `validation_end()` with Lightning v2 epoch logging ([#265](https://github.com/opthub-org/pytorch-bsf/issues/265)) ([aa2cfc9](https://github.com/opthub-org/pytorch-bsf/commit/aa2cfc9c339fb04f7591d0bca408369a143f0fe8))
* resolve UnpicklingError in PyTorch 2.6+ and YAML loading failures ([#232](https://github.com/opthub-org/pytorch-bsf/issues/232)) ([439e5ee](https://github.com/opthub-org/pytorch-bsf/commit/439e5eea1383ef2544ba7d94c4bef4082c881292))
* update meshgrid argument validation to check for directory ([1a9e1ee](https://github.com/opthub-org/pytorch-bsf/commit/1a9e1ee94715f19c85362dc89c5abee57a747787))


### Dependencies

* hotfix for mlflow dependency issue ([#245](https://github.com/opthub-org/pytorch-bsf/issues/245)) ([98964a0](https://github.com/opthub-org/pytorch-bsf/commit/98964a08e476b324879f02d8efc423491724487c))
* pin major version constraints for core dependencies ([#268](https://github.com/opthub-org/pytorch-bsf/issues/268)) ([431cc24](https://github.com/opthub-org/pytorch-bsf/commit/431cc24f3fb30522abca48455dabcf42e6fb1f7c))
* Python 3.14 ([#229](https://github.com/opthub-org/pytorch-bsf/issues/229)) ([b15dc5e](https://github.com/opthub-org/pytorch-bsf/commit/b15dc5e335b4e0ebb6b5652ba81b1c4ed97d06bf))
* Support Python 3.13 ([#184](https://github.com/opthub-org/pytorch-bsf/issues/184)) ([6b976ff](https://github.com/opthub-org/pytorch-bsf/commit/6b976fff1020edbf56e6cb4b0bd7e68a2d81ed06))
* types-setuptools and pyyaml ([#136](https://github.com/opthub-org/pytorch-bsf/issues/136)) ([f006b7f](https://github.com/opthub-org/pytorch-bsf/commit/f006b7f02eeb0633cedc8748ce18bea1e7a4863d))
* use git lfs ([#142](https://github.com/opthub-org/pytorch-bsf/issues/142)) ([4157690](https://github.com/opthub-org/pytorch-bsf/commit/41576902f7c4ee8fe235193f0068d8b628024cc3))


### Documentation

* add comprehensive documentation explaining Bézier simplex fitting and its relation to multi-objective optimization ([#237](https://github.com/opthub-org/pytorch-bsf/issues/237)) ([53d6e8a](https://github.com/opthub-org/pytorch-bsf/commit/53d6e8a5cc8a1de889e36840f4068e7c2275a581))
* add CONTRIBUTING.md with developer onboarding guide ([#251](https://github.com/opthub-org/pytorch-bsf/issues/251)) ([97b9864](https://github.com/opthub-org/pytorch-bsf/commit/97b9864a8959400a1bf6ab184831ba762da0a85b))
* add examples ([a363fc5](https://github.com/opthub-org/pytorch-bsf/commit/a363fc52f1a7b523b91c60b0a94245509134d842))
* add numpy-style docstrings to scaler classes in preprocessing.py ([#269](https://github.com/opthub-org/pytorch-bsf/issues/269)) ([8cd6d35](https://github.com/opthub-org/pytorch-bsf/commit/8cd6d35fa3428dd2946f9c56f0c1a800387b24f7))
* fix errors ([f1802b1](https://github.com/opthub-org/pytorch-bsf/commit/f1802b146d7f0a38edc2d23c8460862dd523de90))
* fix minor errors in document ([42d3362](https://github.com/opthub-org/pytorch-bsf/commit/42d336202363b96ab30991a4575648b8bd6ed429))
* fix minor errors in document ([fec508e](https://github.com/opthub-org/pytorch-bsf/commit/fec508ed9c5f15bc5406de4833282d9b9fd99b57))
* fix readme ([4517668](https://github.com/opthub-org/pytorch-bsf/commit/4517668ba33b5f10e1240a958c6a1aa85dd7dfaf))
* Fix sphinx multiversion compatibility issue ([#225](https://github.com/opthub-org/pytorch-bsf/issues/225)) ([dc513af](https://github.com/opthub-org/pytorch-bsf/commit/dc513af4b74ff0643bfb06def5fe4ee144c69b5c))
* fix typos in document ([5964b30](https://github.com/opthub-org/pytorch-bsf/commit/5964b30ebb168ea20589f2c7faa8eb848cfbf3ab))
* multiple document versions ([#144](https://github.com/opthub-org/pytorch-bsf/issues/144)) ([80f7d99](https://github.com/opthub-org/pytorch-bsf/commit/80f7d996a12279d1c7d172208c8286fdd0faed54))
* rewrite FAQ ([#239](https://github.com/opthub-org/pytorch-bsf/issues/239)) ([9018e2e](https://github.com/opthub-org/pytorch-bsf/commit/9018e2e052b2ae7afde2ea8029087d56c02a39bd))
* sphinx multiversion ([#139](https://github.com/opthub-org/pytorch-bsf/issues/139)) ([ae94f10](https://github.com/opthub-org/pytorch-bsf/commit/ae94f107902c2345bcd60db1956304854b59334f))
* Transfer owner ([#219](https://github.com/opthub-org/pytorch-bsf/issues/219)) ([2ac8016](https://github.com/opthub-org/pytorch-bsf/commit/2ac8016b3e15ac9c646b4aaed747f52260e8f73d))
* update examples ([5a07b7b](https://github.com/opthub-org/pytorch-bsf/commit/5a07b7b14008f287a03e38a2f5c5b4f12e2dad45))
* update terminology ([#209](https://github.com/opthub-org/pytorch-bsf/issues/209)) ([dc52c50](https://github.com/opthub-org/pytorch-bsf/commit/dc52c506bb1c556fc5cd5f636e98f5cf82734933))
* use license_files instead of license_file ([#146](https://github.com/opthub-org/pytorch-bsf/issues/146)) ([82b5fd8](https://github.com/opthub-org/pytorch-bsf/commit/82b5fd86f6b56454e69218f513de1a7c4431f76c))

## [0.15.1](https://github.com/opthub-org/pytorch-bsf/compare/v0.15.0...v0.15.1) (2026-03-27)


### Bug Fixes

* close file handles using context managers in save() and load() ([#264](https://github.com/opthub-org/pytorch-bsf/issues/264)) ([fad6dfb](https://github.com/opthub-org/pytorch-bsf/commit/fad6dfb739afdb8cbe6e214cffd7a59372245007))
* replace deprecated `validation_end()` with Lightning v2 epoch logging ([#265](https://github.com/opthub-org/pytorch-bsf/issues/265)) ([aa2cfc9](https://github.com/opthub-org/pytorch-bsf/commit/aa2cfc9c339fb04f7591d0bca408369a143f0fe8))


### Dependencies

* pin major version constraints for core dependencies ([#268](https://github.com/opthub-org/pytorch-bsf/issues/268)) ([431cc24](https://github.com/opthub-org/pytorch-bsf/commit/431cc24f3fb30522abca48455dabcf42e6fb1f7c))


### Documentation

* add numpy-style docstrings to scaler classes in preprocessing.py ([#269](https://github.com/opthub-org/pytorch-bsf/issues/269)) ([8cd6d35](https://github.com/opthub-org/pytorch-bsf/commit/8cd6d35fa3428dd2946f9c56f0c1a800387b24f7))

## [0.15.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.14.3...v0.15.0) (2026-03-26)


### Features

* visualization ([#247](https://github.com/opthub-org/pytorch-bsf/issues/247)) ([4a4e4a0](https://github.com/opthub-org/pytorch-bsf/commit/4a4e4a07a4211ba01de5ff123c23cbf41ba97531))


### Bug Fixes

* remove cudatoolkit, add multi-OS conda CI ([#252](https://github.com/opthub-org/pytorch-bsf/issues/252)) ([118a841](https://github.com/opthub-org/pytorch-bsf/commit/118a841c2feb9fa20550199a4f1801c8c75a1923))


### Documentation

* add CONTRIBUTING.md with developer onboarding guide ([#251](https://github.com/opthub-org/pytorch-bsf/issues/251)) ([97b9864](https://github.com/opthub-org/pytorch-bsf/commit/97b9864a8959400a1bf6ab184831ba762da0a85b))

## [0.14.3](https://github.com/opthub-org/pytorch-bsf/compare/v0.14.2...v0.14.3) (2026-03-24)


### Dependencies

* hotfix for mlflow dependency issue ([#245](https://github.com/opthub-org/pytorch-bsf/issues/245)) ([98964a0](https://github.com/opthub-org/pytorch-bsf/commit/98964a08e476b324879f02d8efc423491724487c))


### Documentation

* rewrite FAQ ([#239](https://github.com/opthub-org/pytorch-bsf/issues/239)) ([9018e2e](https://github.com/opthub-org/pytorch-bsf/commit/9018e2e052b2ae7afde2ea8029087d56c02a39bd))

## [0.14.2](https://github.com/opthub-org/pytorch-bsf/compare/v0.14.1...v0.14.2) (2026-03-21)


### Bug Fixes

* resolve UnpicklingError in PyTorch 2.6+ and YAML loading failures ([#232](https://github.com/opthub-org/pytorch-bsf/issues/232)) ([439e5ee](https://github.com/opthub-org/pytorch-bsf/commit/439e5eea1383ef2544ba7d94c4bef4082c881292))


### Documentation

* add comprehensive documentation explaining Bézier simplex fitting and its relation to multi-objective optimization ([#237](https://github.com/opthub-org/pytorch-bsf/issues/237)) ([53d6e8a](https://github.com/opthub-org/pytorch-bsf/commit/53d6e8a5cc8a1de889e36840f4068e7c2275a581))

## [0.14.1](https://github.com/opthub-org/pytorch-bsf/compare/v0.14.0...v0.14.1) (2026-03-16)


### Dependencies

* Python 3.14 ([#229](https://github.com/opthub-org/pytorch-bsf/issues/229)) ([b15dc5e](https://github.com/opthub-org/pytorch-bsf/commit/b15dc5e335b4e0ebb6b5652ba81b1c4ed97d06bf))

## [0.14.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.13.4...v0.14.0) (2026-03-16)


### Features

* meshgrid parameter in entry points is now optional ([#226](https://github.com/opthub-org/pytorch-bsf/issues/226)) ([e82be30](https://github.com/opthub-org/pytorch-bsf/commit/e82be30c0d4a9e9392221ceac9cbf4450d0b6373))


### Bug Fixes

* update meshgrid argument validation to check for directory ([1a9e1ee](https://github.com/opthub-org/pytorch-bsf/commit/1a9e1ee94715f19c85362dc89c5abee57a747787))

## [0.13.4](https://github.com/opthub-org/pytorch-bsf/compare/v0.13.3...v0.13.4) (2026-03-15)


### Documentation

* Fix sphinx multiversion compatibility issue ([#225](https://github.com/opthub-org/pytorch-bsf/issues/225)) ([dc513af](https://github.com/opthub-org/pytorch-bsf/commit/dc513af4b74ff0643bfb06def5fe4ee144c69b5c))
* Transfer owner ([#219](https://github.com/opthub-org/pytorch-bsf/issues/219)) ([2ac8016](https://github.com/opthub-org/pytorch-bsf/commit/2ac8016b3e15ac9c646b4aaed747f52260e8f73d))

## [0.13.3](https://github.com/opthub-org/pytorch-bsf/compare/v0.13.2...v0.13.3) (2026-02-21)


### Bug Fixes

* Fix sample codes. ([#212](https://github.com/opthub-org/pytorch-bsf/issues/212)) ([3369166](https://github.com/opthub-org/pytorch-bsf/commit/3369166ca5387e37fdaef4c5d6db3a73c723834b))

## [0.13.2](https://github.com/opthub-org/pytorch-bsf/compare/v0.13.1...v0.13.2) (2025-09-26)


### Documentation

* update terminology ([#209](https://github.com/opthub-org/pytorch-bsf/issues/209)) ([dc52c50](https://github.com/opthub-org/pytorch-bsf/commit/dc52c506bb1c556fc5cd5f636e98f5cf82734933))

## [0.13.1](https://github.com/opthub-org/pytorch-bsf/compare/v0.13.0...v0.13.1) (2025-03-10)


### Bug Fixes

* pytest bug ([d8dfea7](https://github.com/opthub-org/pytorch-bsf/commit/d8dfea7b9e565ad6922eb616a61c6d4ee87237a9))
* pytest bug ([290b16d](https://github.com/opthub-org/pytorch-bsf/commit/290b16db7cb497ad40d5d9a460b4a7f656ea353a))

## [0.13.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.12.1...v0.13.0) (2025-01-25)


### Features

* disable checkpointing by default ([#192](https://github.com/opthub-org/pytorch-bsf/issues/192)) ([9d80470](https://github.com/opthub-org/pytorch-bsf/commit/9d80470887494d876f1f6d633663f33d4c04f46b))

## [0.12.1](https://github.com/opthub-org/pytorch-bsf/compare/v0.12.0...v0.12.1) (2025-01-08)


### Bug Fixes

* load dataset ([2a8cd5b](https://github.com/opthub-org/pytorch-bsf/commit/2a8cd5bfaf2f060da217355b193d5f68755703fe))

## [0.12.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.11.3...v0.12.0) (2025-01-08)


### Features

* support 1D datasets ([#188](https://github.com/opthub-org/pytorch-bsf/issues/188)) ([8b3d29f](https://github.com/opthub-org/pytorch-bsf/commit/8b3d29f5bbd286a7951d21fdeb5d11c61c3b07c7))

## [0.11.3](https://github.com/opthub-org/pytorch-bsf/compare/v0.11.2...v0.11.3) (2025-01-07)


### Dependencies

* Support Python 3.13 ([#184](https://github.com/opthub-org/pytorch-bsf/issues/184)) ([6b976ff](https://github.com/opthub-org/pytorch-bsf/commit/6b976fff1020edbf56e6cb4b0bd7e68a2d81ed06))

## [0.11.2](https://github.com/opthub-org/pytorch-bsf/compare/v0.11.1...v0.11.2) (2025-01-02)


### Bug Fixes

* kfold meshgrid ([979ca65](https://github.com/opthub-org/pytorch-bsf/commit/979ca65cb37c948459c0a28f22e9862106de4106))
* kfold meshgrid ([#177](https://github.com/opthub-org/pytorch-bsf/issues/177)) ([979ca65](https://github.com/opthub-org/pytorch-bsf/commit/979ca65cb37c948459c0a28f22e9862106de4106))

## [0.11.1](https://github.com/opthub-org/pytorch-bsf/compare/v0.11.0...v0.11.1) (2025-01-01)


### Bug Fixes

* MLproject ([#174](https://github.com/opthub-org/pytorch-bsf/issues/174)) ([ef1d2d4](https://github.com/opthub-org/pytorch-bsf/commit/ef1d2d42d636a5535f521b7948405fcc9d3d622c))
* MLproject ([#176](https://github.com/opthub-org/pytorch-bsf/issues/176)) ([5b7afd9](https://github.com/opthub-org/pytorch-bsf/commit/5b7afd9e02b0cbf5bd29fa8d361a3edacc4eb5a3))

## [0.11.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.10.0...v0.11.0) (2024-12-25)


### Features

* write a file for each fold ([#170](https://github.com/opthub-org/pytorch-bsf/issues/170)) ([b00d4c3](https://github.com/opthub-org/pytorch-bsf/commit/b00d4c3807ab67e6d1963d778779903967816a7c))

## [0.10.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.9.0...v0.10.0) (2024-12-17)


### Features

* add scalers ([#166](https://github.com/opthub-org/pytorch-bsf/issues/166)) ([af5d0a3](https://github.com/opthub-org/pytorch-bsf/commit/af5d0a333b4c1fe681ad02bd34b860f017d6b16e))


### Bug Fixes

* MLproject can get additional params ([#163](https://github.com/opthub-org/pytorch-bsf/issues/163)) ([9ea524e](https://github.com/opthub-org/pytorch-bsf/commit/9ea524e2b82f48ddea9f493937db84b00995df06))

## [0.9.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.8.0...v0.9.0) (2024-10-12)


### Features

* MLproject entrypoint for elastic net grid ([#161](https://github.com/opthub-org/pytorch-bsf/issues/161)) ([445cc52](https://github.com/opthub-org/pytorch-bsf/commit/445cc5268da83529e23c246ec8b971d39a0c8b32))

## [0.8.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.7.0...v0.8.0) (2024-09-25)


### Features

* kfold entrypoint ([#158](https://github.com/opthub-org/pytorch-bsf/issues/158)) ([69c3d0b](https://github.com/opthub-org/pytorch-bsf/commit/69c3d0b4326e3f47aca1d761cf7be58dac3eb5ec))

## [0.7.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.6.0...v0.7.0) (2024-09-25)


### Features

* k-fold cross validation ([#153](https://github.com/opthub-org/pytorch-bsf/issues/153)) ([a345dfd](https://github.com/opthub-org/pytorch-bsf/commit/a345dfde5dc0ba8eb3fd8cf6ca65f314d9d2db2f))
* stratified splitting ([#156](https://github.com/opthub-org/pytorch-bsf/issues/156)) ([60adc3a](https://github.com/opthub-org/pytorch-bsf/commit/60adc3a01ed701206d13a2a2bda7acf0a30cb876))

## [0.6.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.5.1...v0.6.0) (2024-09-21)


### Features

* grid sampler for elastic net ([#150](https://github.com/opthub-org/pytorch-bsf/issues/150)) ([2b5da95](https://github.com/opthub-org/pytorch-bsf/commit/2b5da9501a1788671e30055dc71d8fdcd118144d))


### Dependencies

* use git lfs ([#142](https://github.com/opthub-org/pytorch-bsf/issues/142)) ([4157690](https://github.com/opthub-org/pytorch-bsf/commit/41576902f7c4ee8fe235193f0068d8b628024cc3))


### Documentation

* multiple document versions ([#144](https://github.com/opthub-org/pytorch-bsf/issues/144)) ([80f7d99](https://github.com/opthub-org/pytorch-bsf/commit/80f7d996a12279d1c7d172208c8286fdd0faed54))
* sphinx multiversion ([#139](https://github.com/opthub-org/pytorch-bsf/issues/139)) ([ae94f10](https://github.com/opthub-org/pytorch-bsf/commit/ae94f107902c2345bcd60db1956304854b59334f))
* use license_files instead of license_file ([#146](https://github.com/opthub-org/pytorch-bsf/issues/146)) ([82b5fd8](https://github.com/opthub-org/pytorch-bsf/commit/82b5fd86f6b56454e69218f513de1a7c4431f76c))

## [0.5.1](https://github.com/opthub-org/pytorch-bsf/compare/v0.5.0...v0.5.1) (2023-10-19)


### Dependencies

* types-setuptools and pyyaml ([#136](https://github.com/opthub-org/pytorch-bsf/issues/136)) ([f006b7f](https://github.com/opthub-org/pytorch-bsf/commit/f006b7f02eeb0633cedc8748ce18bea1e7a4863d))

## [0.5.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.4.0...v0.5.0) (2023-10-19)


### Features

* auto detect file types ([#134](https://github.com/opthub-org/pytorch-bsf/issues/134)) ([964c671](https://github.com/opthub-org/pytorch-bsf/commit/964c6717835087b9b93477cd6b9ddb553fcfaeec))


### Documentation

* add examples ([a363fc5](https://github.com/opthub-org/pytorch-bsf/commit/a363fc52f1a7b523b91c60b0a94245509134d842))
* fix errors ([f1802b1](https://github.com/opthub-org/pytorch-bsf/commit/f1802b146d7f0a38edc2d23c8460862dd523de90))
* fix minor errors in document ([42d3362](https://github.com/opthub-org/pytorch-bsf/commit/42d336202363b96ab30991a4575648b8bd6ed429))
* update examples ([5a07b7b](https://github.com/opthub-org/pytorch-bsf/commit/5a07b7b14008f287a03e38a2f5c5b4f12e2dad45))

## [0.4.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.3.1...v0.4.0) (2023-10-19)


### Features

* fixable control points ([#130](https://github.com/opthub-org/pytorch-bsf/issues/130)) ([e4d6c0e](https://github.com/opthub-org/pytorch-bsf/commit/e4d6c0e5d5aabadeb5f42f8c41ca8057f7a5fcf4))


### Documentation

* fix minor errors in document ([fec508e](https://github.com/opthub-org/pytorch-bsf/commit/fec508ed9c5f15bc5406de4833282d9b9fd99b57))
* fix readme ([4517668](https://github.com/opthub-org/pytorch-bsf/commit/4517668ba33b5f10e1240a958c6a1aa85dd7dfaf))
* fix typos in document ([5964b30](https://github.com/opthub-org/pytorch-bsf/commit/5964b30ebb168ea20589f2c7faa8eb848cfbf3ab))

## [0.3.1](https://github.com/opthub-org/pytorch-bsf/compare/v0.3.0...v0.3.1) (2023-10-17)


### Bug Fixes

* fit() uses init and skeleton ([#127](https://github.com/opthub-org/pytorch-bsf/issues/127)) ([c92ffcf](https://github.com/opthub-org/pytorch-bsf/commit/c92ffcf2fb21a067da6868e96c671036a391dbd5))

## [0.3.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.2.0...v0.3.0) (2023-10-17)


### Features

* skeleton validator ([#124](https://github.com/opthub-org/pytorch-bsf/issues/124)) ([95767e5](https://github.com/opthub-org/pytorch-bsf/commit/95767e5766a41b4be9cce9b6c1bff35afc317b90))

## [0.2.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.1.0...v0.2.0) (2023-10-12)


### Features

* precision flag ([#122](https://github.com/opthub-org/pytorch-bsf/issues/122)) ([3d74982](https://github.com/opthub-org/pytorch-bsf/commit/3d74982eb3c81a90150b43b36018525e692706bd))

## [0.1.0](https://github.com/opthub-org/pytorch-bsf/compare/v0.0.2...v0.1.0) (2023-10-12)


### Features

* use pytorch-lightning v2 ([#119](https://github.com/opthub-org/pytorch-bsf/issues/119)) ([8afbf48](https://github.com/opthub-org/pytorch-bsf/commit/8afbf482539540028d14a3cd9eafac219bebd71d))
