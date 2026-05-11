# Discovering Causal Relationships using Proxy Variables under Unmeasured Confounding

This repository contains simulation code and statistical testing procedures for $X\perp Y|U$ by proximal causal discovery using Proxy Variables.

The project includes:

* Proximal proxy-based causal testing methods
* RKHS residual learning with cross-validation
* Kernel conditional independence (KCI) baselines
* Discrete proxy testing baselines
* Synthetic data generation pipelines
* Simulation scripts for reproducing experimental figures

---

# Repository Structure

```text
proximal_causal_discovery_cv/
│
├── CausalTestBench.py        # Main testing interface
├── Simu_Fig2.py              # Simulation scripts for experiments/figures
├── Simu_Fig3.py
├── Simu_Fig4.py
├── Simu_Fig5.py
├── Simu_Fig6.py
├── Simu_Fig8.py
├── Simu_Fig9.py
├── Simu_Fig10.py
│
├── model/
│   ├── MMR_test.py           # RKHS residual learning
│   ├── kcm.py                # Kernel conditional moment test
│   └── kcm_complex.py        # Complex-valued kernel moment test
│
├── data/
│   ├── generate_data.py      # Synthetic data generators
│   ├── structual_equations.py
│   ├── linear_causal_generator.py
│   ├── generate_obersved.py
│   └── generate_miao.py
│
├── discrete/
│   ├── estimate_H.py
│   ├── sigma.py
│   ├── testing.py
│   └── liu/
│
├── KCI/
│   ├── KCI.py                # Kernel conditional independence testing
│   ├── GaussianKernel.py
│   ├── PolynomialKernel.py
│   └── LinearKernel.py
│
└── utils/
    ├── kernel_fun.py
    └── utils.py
```

---

# Main Components

## 1. `CausalTester`

The main interface is implemented in:

```python
from CausalTestBench import CausalTester
```

The tester supports several causal testing procedures:

| Method                | Description                                                          |
| --------------------- | -------------------------------------------------------------------- |
| `kernel_proxy_test()` | RKHS-based proximal proxy test using complex-valued moment functions |
| `moment_proxy_test()` | Moment-based proximal proxy test                                     |
| `proxy_test()`        | Discretized proxy-based testing baseline                             |
| `kernel_test()`       | Standard kernel conditional independence test                        |

The input data format is:

```python
(X, W, Y)
```

or

```python
(X, W, Y, V)
```

where:

* `X`: treatment variable
* `W`: proxy variable
* `Y`: outcome variable
* `V`: optional observed covariates

---

# Installation

## Requirements

The codebase relies on the following Python packages:

```text
numpy
scipy
pandas
joblib
tqdm
scikit-learn
```

Install dependencies with:

```bash
pip install numpy scipy pandas joblib tqdm scikit-learn
```

---

# Quick Start

## Example: Running a Single Test

```python
import numpy as np
from CausalTestBench import CausalTester
from data.generate_data import simulate_samples

# Generate synthetic data
samples = simulate_samples(num_samples=500, type='causal')

X = samples['X']
W = samples['W']
Y = samples['Y']

# Initialize tester
ct = CausalTester((X, W, Y))

# Run RKHS proximal proxy test
reject, pvalue = ct.kernel_proxy_test()

print('Reject:', reject)
print('P-value:', pvalue)
```

---

# Running Simulation Experiments

Simulation scripts are provided for reproducing experimental figures.

For example:

```bash
python Simu_Fig2.py
```

The simulation pipeline:

1. Generates synthetic datasets
2. Runs multiple causal testing procedures
3. Aggregates rejection statistics
4. Saves outputs as JSON files

Typical output:

```text
Sample size: 200, causal: causal,
kernel_proxy_test_sum: 85,
proxy_test_sum: 73,
kernel_test_sum: 51
```

Results are stored in:

```text
decs_results.json
```

---

# RKHS Proxy Testing Procedure

The proposed testing framework follows the general pipeline:

1. Learn residual representations in RKHS
2. Construct moment restrictions using proxy variables
3. Apply kernel conditional moment testing
4. Estimate p-values via bootstrap sampling

The implementation uses:

* Cross-validated regularization
* Complex-valued feature mappings (`cos` and `sin` moments)
* Kernel conditional moment statistics
* Bootstrap calibration

---

# Parallel Execution

Most simulation routines support parallel computation through `joblib`.

Example:

```python
results = run_tests(
    datas,
    method='kernel_proxy_test',
    use_parallel=True,
    n_jobs=50
)
```

---

# Synthetic Data Generation

Synthetic datasets are generated from:

```python
from data.generate_data import simulate_samples
```

Example:

```python
samples = simulate_samples(
    num_samples=1000,
    type='causal'
)
```
---

# Reproducibility

To reproduce experiments:

1. Run the corresponding `Simu_Fig*.py` script
2. Collect generated JSON outputs
3. Aggregate and visualize the rejection rates

Example:

```bash
python Simu_Fig5.py
```

---

# Notes

* The repository focuses on methodological experimentation and simulation studies.
* Some scripts may require substantial computational resources.
* Large-scale experiments are recommended on multi-core machines.

---

# Citation

If you use this repository in your research, please cite the associated paper or project.

```bibtex
@inproceedings{wu2025bivariate,
  title={Bivariate Causal Discovery with Proxy Variables: Integral Solving and Beyond},
  author={Wu, Yong and Fu, Yanwei and Wang, Shouyan and Sun, Xinwei},
  booktitle={International Conference on Machine Learning},
  pages={67202--67259},
  year={2025},
  organization={PMLR}
}

@article{wu2025discovering,
  title={Discovering Causal Relationships using Proxy Variables under Unmeasured Confounding},
  author={Wu, Yong and Fu, Yanwei and Wang, Shouyan and Wang, Yizhou and Sun, Xinwei},
  journal={arXiv preprint arXiv:2510.17167},
  year={2025}
}
```