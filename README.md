# OPTical Quantum Sdk

<div align="center">
  <img width="543" height="181" alt="optiqs" src="https://github.com/user-attachments/assets/e58501a3-93cb-4825-8272-b121cd70783e" />
  <h3>Efficient, low-memory simulation of photonic quantum systems with sparse matrices</h3>

  <img src="https://img.shields.io/github/license/oe1307/optiqs?logo=open-source-initiative&logoColor=green">
  <img src="https://img.shields.io/github/last-commit/oe1307/optiqs?logo=git&logoColor=white">
  <img src="https://img.shields.io/github/issues/oe1307/optiqs?logo=github&logoColor=white">
  <img src="https://img.shields.io/github/issues-pr/oe1307/optiqs?logo=github&logoColor=white">
  <img src="https://img.shields.io/github/languages/code-size/oe1307/optiqs?logo=github&logoColor=white">
</div>


## Setup

```sh
pip install git+https://github.com/oe1307/optiqs.git
```

## Usage

```python
import optiqs as opt

dim = 20

# Create a single-mode coherent state with amplitude alpha = 1.
state = opt.coherent(dim, alpha=1.0)

# Find the probability of detecting 0, 1, 2, ... photons.
probabilities = opt.pnrd(state)
for n, probability in enumerate(probabilities[:5]):
    print(f"P({n}) = {probability:.3f}")
```
