# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
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
- **Test reporting**: Added pytest-html integration for visual HTML test reports at `tests/reports/report.html`.
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

