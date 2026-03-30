# Changelog

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
