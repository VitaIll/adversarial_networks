# Adversarial Structural Estimation on a Single Network
Minimal MVP for adversarial estimation of a linear-in-means structural model from one synthetic observed equilibrium on a single graph.

## Install (Exact)

### Option 1: Conda Environment (Locked Dependencies - Recommended)
Use the committed lock file for exact dependency resolution:

```bash
conda-lock install -n adversarial_networks conda-lock.yml
conda activate adversarial_networks
pip install -e .
```

If you need to regenerate the lock:

```bash
conda-lock -f environment.yml -p win-64
```

### Option 2: pip (Development Mode)
For rapid development with editable install:

```bash
pip install -e .
pip install -e .[dev]  # Include development tools (pytest, pytest-html, ruff)
```

**Tested baseline platform:**
- OS: Windows 10
- Python: 3.11.9
- Device: CPU

## Verify Tests
Run tests from repository root:

```bash
pytest
```

This will:
- Run all 4 deterministic CPU tests with verbose output
- Generate an HTML test report at `tests/reports/report.html`
- Show detailed expected vs actual values for any failures

For strict warning checking (repo code only):

```bash
python -W error -m pytest
```

The `pytest.ini` includes narrow filters for unavoidable third-party deprecation warnings from PyG/Torch integration.

## Run Experiment
From repository root:

```bash
jupyter notebook
```

Open `experiments/linear_in_means_model.ipynb` and run all cells.  
The notebook writes artifacts to `artifacts/runs/<RUN_ID>/` and includes `run_manifest.json` with:
- package versions
- platform info
- git metadata (`null` if unavailable)
- SHA256 hashes for `environment.yml`, `conda-lock.yml`, and produced artifacts

Tracked baseline release artifacts are in `artifacts/baseline/`.

## Repository Structure
```text
adversarial_networks/
├── docs/
│   ├── design_doc.md                    # Complete design specification
│   ├── paper.md                         # Theory and results
│   └── paper_structure.md               # Paper outline
├── src/
│   ├── __init__.py                      # Package exports and version
│   ├── generator.py                     # SCMGenerator (structural causal model)
│   ├── discriminator.py                 # RootedMPNNDiscriminator (GIN-based classifier)
│   ├── utils.py                         # Network utilities (W matrix, ego-batching)
│   ├── config.py                        # Configuration dataclasses
│   ├── constants.py                     # Named constants
│   ├── visualization.py                 # Plotting utilities
│   └── io_utils.py                      # I/O functions (CSV, JSON, hashing)
├── experiments/
│   └── linear_in_means_model.ipynb      # End-to-end demo (C1-C14)
├── artifacts/
│   ├── baseline/                        # Tracked release artifacts
│   │   ├── fig01_observed_data.png
│   │   ├── fig02_theta_convergence.png
│   │   ├── fig03_loss_convergence.png
│   │   ├── fig04_discriminator_scores.png
│   │   ├── fig05_Y_distributions.png
│   │   ├── fig06_tail_stability.png
│   │   ├── run_manifest.json
│   │   ├── tab01_data_summary.csv
│   │   ├── tab02_estimation_results.csv
│   │   └── tab03_convergence_tail.csv
│   └── runs/                            # Local runs (gitignored)
├── tests/
│   ├── conftest.py                      # Shared pytest fixtures
│   ├── test_utils.py                    # 4 core tests with enhanced assertions
│   └── reports/                         # HTML test reports (gitignored)
├── .gitignore
├── CHANGELOG.md
├── conda-lock.yml                       # Locked dependencies
├── environment.yml                      # Editable dependency spec
├── pyproject.toml                       # Python packaging and tool config
├── pytest.ini                           # pytest configuration
└── README.md
```

## References
- Kaji, T., Manresa, E., and Pouliot, G. (2023). *An Adversarial Approach to Structural Estimation*. Econometrica 91(6): 2041-2063. DOI: 10.3982/ECTA18707.
- Bramoulle, Y., Djebbari, H., and Fortin, B. (2009). *Identification of peer effects through social networks*. Journal of Econometrics 150(1): 41-55.
