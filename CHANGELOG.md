# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Repository hygiene & reorganization
- **Licensing/packaging**: added a real `LICENSE` file (MIT) backing the `pyproject` declaration and ensured it
  ships in built distributions via `[tool.setuptools] license-files`; single-sourced the version through
  `[tool.setuptools.dynamic]` (no more dual `pyproject`/`__init__` literal).
- **Repo structure**: the hand-written design docs under `docs/` are now tracked (the directory was previously
  git-ignored wholesale, leaving the README's `docs/design/` links dead in a clone); added `CITATION.cff` and a
  GitHub Actions CI workflow (`ruff` + `pytest`, CPU-only torch). The source-paper PDFs are kept out of version
  control by decision (root `*.pdf` ignored).
- **Dependencies / reproducibility**: added the genuinely-used `pandas` (and `pytest-html`/`ruff`) to
  `environment.yml` and regenerated the conda lock; removed the mandatory `--html` from `pytest.ini addopts`
  (the HTML report is now opt-in) so a non-`[dev]` install can run `pytest`.
- **Tests**: retired the legacy `tests/test_utils.py`; its checks moved to `tests/test_core_graph.py`,
  `tests/test_core_ego_features.py`, `tests/test_core_objective.py`, and (the generator-integration Picard
  check) `tests/test_generators.py`, all using the real module names.

### Changed — general-framework refactor

Refactored the package into a **general framework** for adversarial structural
estimation of network-equilibrium models, with a `scikit-learn` / `DoubleML`-style
public API and a separated fast computational core. See `docs/design/` for the full
design.

- **Package rename** `src` → `adversarial_networks` (the distribution/repo name); added
  `[tool.setuptools.packages.find]`, `pandas`, and a `py.typed` marker.
- **Fast computational core** `adversarial_networks/core/`: `equilibrium` (`picard`,
  `newton` with analytic/AD-diagonal Jacobian, `solve_equilibrium` with
  `unroll`/`implicit` differentiation), `graph`, `neighborhoods`, `objective` (losses +
  convergence + instance-noise), `ego_features` (the single PyG seam), `types`. Kernels
  are `torch_geometric`-free and unit-tested in isolation.
- **Framework layer**: `NetworkGameGenerator` base (subclass with `best_response` xor
  `foc_residual` + optional `peer_aggregate`/`sample_shocks`/`initial_state`); declarative
  `transforms` (`Real`/`Positive`/`Interval`); `check_model`/`ModelReport` admissibility
  surface (operator-∞-norm contraction, locality, shock monotonicity, uniqueness,
  residual, gradient flow). `SCMGenerator` renamed `LinearInMeansGenerator`; the effort
  game refactored onto the base — both **numeric-equivalence guarded** (forward
  bit-identical, gradients `allclose`).
- **Data & estimator**: `NetworkData` (mandatory outcome, validate-before-assign,
  finite/float32 contract); `datasets.make_linear_in_means` / `make_effort_game`;
  `reporting.recovery_table`; a single sklearn-shaped `AdversarialEstimator` with
  `fit(data) -> self`, trailing-underscore learned attributes, `estimates_`,
  `NotFittedError`, clone-safety, receptive-field guard, and the `MinimaxStepContext`
  gradient-transform seam (future Fisher / standard-error milestone). The verified loop
  is the free function `_run_minimax`, shared by `fit` and `MonteCarloRunner`.
- **Curated public surface**: ~24 framework-first names; advanced machinery
  (`EgoSubstrate`, `RootSampler`, `losses`, `core.*`) reachable from its submodule.
- **`io_utils`** generality fix: deleted the dead `save_realization_history`; the loader
  now coerces CSV columns by suffix (model-agnostic, not hard-coded `beta/gamma/sigma_sq`).
- Rewrote the two built-in notebooks and added a from-scratch custom-game notebook;
  rewrote/extended the test suite (core equivalence, framework, `check_model` soundness,
  AD-vs-analytic and unroll-vs-implicit gradients, data/datasets/reporting, clone-safety).

### Changed — earlier (pre-framework)
- **Module refactoring**: Split GAN component classes into dedicated modules for better organization:
  - Created `src/generator.py` containing `SCMGenerator` with comprehensive documentation on the structural causal model, Picard iteration, and parameter reparameterization
  - Created `src/discriminator.py` containing `RootedMPNNDiscriminator` with detailed documentation on GIN architecture, root-aware message passing, and design rationale
  - Updated `src/utils.py` to contain only network utilities (`build_row_stochastic_W`, `extract_ego_batch`)
  - Updated `src/__init__.py` to import from new module structure
  - Updated notebook imports in `experiments/linear_in_means_model.ipynb` to reference new modules
  - All functionality preserved; backward compatibility maintained through package-level imports

### Added
- **Configuration management**: Added `src/config.py` with type-safe dataclasses for all experiment parameters (`ExperimentConfig`, `GraphConfig`, `ModelConfig`, `TrainingConfig`, `TrueParams`, `InitParams`).
- **Named constants**: Added `src/constants.py` to centralize all magic numbers and configuration values.
- **Visualization utilities**: Added `src/visualization.py` with reusable plotting functions (`plot_parameter_convergence`, `plot_loss_convergence`, `plot_tail_stability`).
- **I/O utilities**: Added `src/io_utils.py` with functions for saving CSV tables, computing file hashes, and JSON manifests.
- **Test fixtures**: Added `tests/conftest.py` with shared pytest fixtures for graphs, normalization stats, and ego-caches.
- **Test reporting**: Optional pytest-html HTML report at `tests/reports/report.html` (opt-in via `pytest --html=… --self-contained-html` with the `[dev]` extra; no longer auto-run from `addopts`).
- **Package installation**: Added `pyproject.toml` for standard Python packaging with editable install support (`pip install -e .`).

### Changed
- **Enhanced tests**: All 4 tests now include detailed assertion messages showing expected vs actual values for easier debugging.
- **Improved imports**: Updated `src/__init__.py` to export configuration classes and utility modules alongside core models.
- **pytest configuration**: Consolidated pytest settings in `pytest.ini` with HTML reporting, verbose output, and proper Python path setup.
- **`.gitignore`**: Added entries for `*.egg-info/`, `build/`, `dist/`, and `tests/reports/`.

### Improved
- **Code readability**: Extracted magic numbers to named constants; added comprehensive docstrings with examples.
- **Reproducibility**: Configuration validation with clear error messages; structured test output for auditing.
- **Maintainability**: Modular organization with focused utility modules while keeping core `utils.py` intact.

## [0.1.0] - 2026-02-08

### Added
- Initial implementation of adversarial structural estimation MVP on a single synthetic network.
- `SCMGenerator` and `RootedMPNNDiscriminator` in `src/utils.py`.
- Core utilities `build_row_stochastic_W` and `extract_ego_batch` in `src/utils.py`.
- Package re-exports and version in `src/__init__.py`.
- End-to-end experiment notebook `experiments/linear_in_means_model.ipynb` (C1-C14 workflow).
- Test suite `tests/test_utils.py` with 4 deterministic CPU tests.
- Reproducibility files: `environment.yml` and locked `conda-lock.yml`.
- Baseline tracked artifacts in `artifacts/baseline/` including figures, tables, and `run_manifest.json`.
- Project documentation and repo hygiene files: `README.md`, `.gitignore`, and `pytest.ini`.

