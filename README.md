# Adversarial Structural Estimation on a Single Network
Minimal MVP for adversarial estimation of a linear-in-means structural model from one synthetic observed equilibrium on a single graph.

## Installation

### Prerequisites
- Python `>=3.11`
- One of:
  - Conda + `conda-lock` (exact locked environment; lock file is `win-64`)
  - pip/venv (editable install)

### Option 1: Conda + lock file (exact, recommended on Windows)
```bash
conda-lock install -n adversarial_networks conda-lock.yml
conda activate adversarial_networks
pip install -e .
```

Re-generate the lock file (if `environment.yml` changes):
```bash
conda-lock -f environment.yml -p win-64
```

### Option 2: Conda from `environment.yml` (portable)
```bash
conda env create -n adversarial_networks -f environment.yml
conda activate adversarial_networks
pip install -e .
```

### Option 3: pip editable install (development)
```bash
pip install -e .[dev]
pip install notebook jupyterlab
```

### Verify installation
Run from repository root:
```bash
python -c "import src; from src import SCMGenerator, RootedMPNNDiscriminator, RootSampler; print('src version:', src.__version__)"
```

## Usage

### 1) Run tests
```bash
pytest
```

What this does:
- Runs the current test suite in `tests/` (17 tests at the time of writing).
- Writes an HTML report to `tests/reports/report.html`.

Strict warnings mode:
```bash
python -W error -m pytest
```

### 2) Run the end-to-end notebook
From repository root:
```bash
jupyter notebook experiments/linear_in_means_model.ipynb
```

The notebook:
- Loads code from `src/`.
- Runs the C1-C14 workflow.
- Writes run outputs to `artifacts/runs/<RUN_ID>/` and saves `run_manifest.json` with versions, platform, git metadata, and file hashes.

Notes:
- Default notebook config is large-scale (`n_nodes=250000`, `n_steps=800`) and can take significant time on CPU.
- For a fast local smoke run, edit cell C2 and override config (example):

```python
from dataclasses import replace
from config import ExperimentConfig

cfg = ExperimentConfig.default()
cfg = replace(
    cfg,
    graph=replace(cfg.graph, n_nodes=5000, graph_type="ba"),
    training=replace(
        cfg.training,
        n_steps=100,
        n_disc=1,
        batch_size=32,
        root_sampler_mode="uniform",
    ),
)
```

### 3) Optional: run lint checks
```bash
ruff check src tests
```

## Repository Structure
Tracked files users see in GitHub:

```text
adversarial_networks/
|-- experiments/
|   `-- linear_in_means_model.ipynb
|-- src/
|   |-- __init__.py
|   |-- config.py
|   |-- constants.py
|   |-- discriminator.py
|   |-- generator.py
|   |-- io_utils.py
|   |-- plot_style.py
|   |-- root_sampling.py
|   |-- utils.py
|   `-- visualization.py
|-- tests/
|   |-- conftest.py
|   |-- test_root_sampling.py
|   |-- test_utils.py
|   `-- test_visualization.py
|-- .gitignore
|-- CHANGELOG.md
|-- conda-lock.yml
|-- environment.yml
|-- pyproject.toml
|-- pytest.ini
`-- README.md
```

Gitignored local paths (not part of the GitHub tree) include `artifacts/`, `docs/`, `tests/reports/`, `.pytest_cache/`, `*.egg-info/`, and `__pycache__/`.

## References
- Kaji, T., Manresa, E., and Pouliot, G. (2023). *An Adversarial Approach to Structural Estimation*. Econometrica 91(6): 2041-2063. DOI: 10.3982/ECTA18707.
- Bramoulle, Y., Djebbari, H., and Fortin, B. (2009). *Identification of peer effects through social networks*. Journal of Econometrics 150(1): 41-55.
