# Spectral Contracts

**Computable Spectral Diagnostics for Long-Horizon Stability in SSMs**

This project implements pre-training spectral diagnostics that predict whether an SSM stack will train stably and preserve long-context signal, before any large training spend is committed.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run stability check on a model configuration
ssm-contracts check --config model_config.yaml

# Run full benchmark suite
python benchmarks/run_sweeps.py

# Generate predictiveness analysis
python analysis/regression_analysis.py
```

## Project Structure

- `metrics/` - Contract metric implementations and inventory
- `benchmarks/` - Benchmark tasks and controlled instability sweeps
- `analysis/` - Predictiveness study and statistical analysis
- `results/` - Raw data, figures, and tables
- `tool/` - CLI implementation with calibrated thresholds
- `paper/` - Paper outline and benchmark tables

## Documentation

- `RESEARCH_NOTES.md` — Implementation decisions, parameter choices, and reproducibility notes
- `QUICKSTART.md` — Installation and basic usage
- `paper/main.pdf` — Full paper (TMLR submission)

See [RESEARCH_NOTES.md](./RESEARCH_NOTES.md) for implementation decisions, experimental design rationale, and reproducibility guidance.

## License

MIT License - see LICENSE file for details.
