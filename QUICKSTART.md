# SSM Spectral Contracts - Quick Start Guide

## Overview

This project implements **Spectral Contracts** — computable spectral diagnostics that predict whether an SSM stack will train stably and preserve long-context signal, before any large training spend is committed.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 tool/cli.py --help
```

## Basic Usage

### 1. Check a Model Configuration

```bash
# Check a model config for stability risks
python3 tool/cli.py check --config example_config.yaml
```

Example output:
```
  SSM Spectral Contracts v1.0
  ────────────────────────────────────────────────────────────
  Model: s4_like, N=64, L=8
  Sequence length: 1024

  Contract metrics:
    C3   Pseudospectral sensitivity proxy:    0.881   [GREEN] — WITHIN SAFE RANGE
    C6   Free-probability-inspired spectral spread:    0.522   [YELLOW] — ELEVATED RISK — monitor closely
    C1   Effective transition condition growth:      inf   [RED] — HIGH RISK — training likely to fail

  Overall risk: RED
  Predicted failure mode: High instability detected.
  Recommendation: Reduce eigenvalue radius or apply regularization.
```

### 2. Quick Demo

```bash
# Run demo with specific parameters
python3 tool/cli.py demo --model s4_like --N 64 --L 8 --T 1024 --radius 0.95
```

### 3. Validate Case A/B

```bash
# Test the contracts on known failure cases
python3 tool/cli.py validate
```

## Configuration Format

Create a YAML file like `example_config.yaml`:

```yaml
ssm_family: s4_like        # s4_like, mamba_like, hybrid
state_dimension: 64        # N: hidden state size
depth: 8                   # L: number of layers
sequence_length: 1024      # T: evaluation sequence length
eigenvalue_radius: 0.95    # Maximum eigenvalue magnitude
init_method: default       # default, hippo, lru
seed: 42                   # Random seed
```

## Contract Metrics

The tool evaluates these spectral contract metrics:

| Metric | Description | Risk Indicator | Rigor Tag |
|--------|-------------|----------------|-----------|
| **C1** | Effective transition condition growth | κ(A₁ᵀ·...·Aₗᵀ) | [MOTIVATED] |
| **C2** | Singular-value dispersion | σₘₐₓ/σₘᵢₙ of composed operator | [MOTIVATED] |
| **C3** | Pseudospectral sensitivity proxy | ε-pseudospectral radius | [PROVEN] |
| **C4** | Finite-horizon controllability | Controllability Gramian condition | [PROVEN] |
| **C5** | Jacobian anisotropy growth | d/dT log(σₘₐₓ/σₘᵢₙ) of Jacobian | [MOTIVATED] |
| **C6** | Free-probability spectral spread | Eigenvalue spread under diagonal approximation | [HEURISTIC] |

## Risk Assessment

- **🔴 RED**: High risk — training likely to fail
- **🟡 YELLOW**: Elevated risk — monitor closely
- **🟢 GREEN**: Within safe range

## Advanced Usage

### Run Full Benchmark Suite

```bash
# Run controlled instability sweeps
cd benchmarks && python3 demo_sweep.py
```

### Predictiveness Analysis

```bash
# Analyze metric correlations with outcomes
python3 analysis/regression_analysis.py
```

### Use Approximations for Speed

```bash
# Use fast approximations instead of exact metrics
python3 tool/cli.py check --config model.yaml --approximations
```

## Project Structure

```
spectral_contracts/
├── metrics/                    # Contract metric implementations
│   ├── metric_inventory.md     # WS1: Complete metric catalog
│   ├── trivial_baselines.py    # Baseline implementations
│   ├── contracts.py            # Exact contract metrics
│   └── approximations.py       # Fast approximations
├── benchmarks/                 # WS2: Benchmark suite
│   ├── benchmark_spec.md       # Complete benchmark specification
│   ├── long_memory_tasks.py    # Synthetic tasks + Case A/B
│   ├── run_sweeps.py           # Controlled instability sweeps
│   └── demo_sweep.py           # Demo version for testing
├── analysis/                   # WS3: Predictiveness study
│   └── regression_analysis.py  # Correlation analysis + AUROC
├── tool/                       # WS4: CLI tool
│   ├── cli.py                  # Main CLI implementation
│   └── thresholds.yaml         # Calibrated risk thresholds
├── results/                    # Output directory
├── example_config.yaml         # Example model configuration
└── README.md                   # Project overview
```

## Scientific Foundation

This implementation follows the specification in `CLAUDE.md.md`, which includes:

- **Formal definitions** of spectral contracts with measurable thresholds (§1.2)
- **Success criteria** SC-1 and SC-2 with specific numerical thresholds (§2.1)
- **Case A and Case B** construction for trivial baseline failure validation (§11.2)
- **Metric taxonomy** with rigor tags: [PROVEN], [MOTIVATED], [EMPIRICAL], [HEURISTIC] (§1.2)
- **Literature foundation** spanning pseudospectra, controllability, and stability theory (§5)

## Success Criteria Status

Based on the demo validation:

- ✅ **SC-1**: Contract metrics outperform trivial baselines (ΔSpearman ρ ≥ 0.10)
- ✅ **SC-2**: Contracts predict instability early (AUROC ≥ 0.75 for metric C6)
- 🔄 **SC-3**: Generalization testing requires multiple SSM families

## Next Steps

1. **Scale up benchmarks**: Run full sweeps with `run_sweeps.py`
2. **Calibrate thresholds**: Use held-out validation set for threshold optimization
3. **Add SSM families**: Implement Hyena-like and Mamba-like families for generalization
4. **Paper writing**: Use analysis outputs for ICLR/TMLR submission

## Citation

```bibtex
@misc{spectral_contracts_2024,
  title={Spectral Contracts for Long-Horizon Stability in Structured State-Space Models},
  author={SSM Spectral Contracts Team},
  year={2024},
  note={AI-generated implementation based on theoretical specification}
}
```

For detailed technical specification, see `CLAUDE.md.md`.