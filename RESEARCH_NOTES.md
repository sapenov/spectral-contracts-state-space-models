# Research Notes: SSM Spectral Contracts

This document captures implementation decisions, experimental design rationale, and reproducibility notes for the paper "Architecture-Dependent Stability Diagnostics for State-Space Models."

## Experimental Design Decisions

**Non-circular outcomes:** We use linear dynamics outcomes rather than spectral properties as ground truth to prevent circular validation where spectral diagnostics are evaluated against spectral criteria. The growth ratio over T=500 steps with random initial conditions provides a training-stability proxy that is independent of the diagnostic metrics being evaluated.

**Two-axis sweep design:** The parameter sweeps vary eigenvalue radius vs. eigenvector condition number independently. This isolates spectral radius (captured by eigenvalue magnitude) from non-normality effects (captured by eigenvector ill-conditioning), allowing us to test whether C3 adds signal beyond trivial baselines.

**Regime choices:** The non-normal parameter regime (κ(V) ∈ {1, 10, 100, 500, 1000}) was chosen to span from exactly normal matrices to severely ill-conditioned cases where non-normal amplification dominates. This tests pseudospectral sensitivity in the operational regime where it should excel.

## Implementation Notes

**Six bugs discovered and fixed during development:**

1. **C4 constant values:** The controllability Gramian metric returned constant outputs across our sweep configurations due to insufficient variation in the controllability subspace. This was resolved by noting the limitation rather than forcing artificial variation.

2. **Basis dependence of C3:** Pseudospectral sensitivity changes under similarity transforms (A → SAS⁻¹) even when eigenvalues are preserved. C3 values changed by 15.2% at κ(S)=10 and 89.7% at κ(S)=100, confirming this measures the specific matrix representation rather than an intrinsic operator property.

3. **Complexity analysis correction:** Initial claims of O(N²·900) cost for C3 were incorrect. The proper analysis yields O(N³ + N²·|G|) where the O(N³) Schur decomposition dominates at practical dimensions.

4. **Toeplitz vs. circulant distinction:** Our original convolution-inspired family used exactly normal circulant matrices. The Toeplitz variant demonstrates that operator norm sufficiency extends to mildly non-normal convolution structures (||AA*-A*A||_F ≈ 0.022).

5. **Table reference numbering:** LaTeX cross-references required careful management across multiple tables and figures. All ?? artifacts in early versions were resolved through proper label management.

6. **Bootstrap confidence intervals:** Statistical rigor required bootstrap confidence intervals for main results. C3 achieves ρ = 0.835 (95% CI: [0.697, 0.883]; per-seed range: [0.792, 0.816]).

## Contract Metric Definitions and Rationale

**C3 (Pseudospectral Sensitivity):** Measures the area of the ε-pseudospectrum relative to the convex hull of the spectrum. This captures how eigenvalues move under small matrix perturbations, detecting non-normal transient amplification that spectral radius alone misses. Uses ε=0.01 with a 30×30 grid centered at the spectral centroid.

**C1-C2, C5-C6:** These metrics target different aspects of stability (transition growth, singular-value dispersion, anisotropy, spectral spread) but did not outperform trivial baselines in our benchmark. They remain in the evaluation for completeness and transparency.

**κ(V) proxy:** Eigenvector condition number provides the cheapest non-normality measure but achieves substantially lower correlation than C3, validating the accuracy-cost tradeoff for pseudospectral analysis.

## Parameter Choices

**ε = 0.01:** Pseudospectral tolerance chosen to balance sensitivity (smaller ε detects finer perturbations) with numerical stability. Ablation testing shows correlation remains stable (0.818-0.838) across ε ∈ {0.001, 0.005, 0.01, 0.05, 0.10}.

**30×30 grid:** Grid resolution balances computational cost with pseudospectral accuracy. Ablation shows correlation stable (0.801-0.838) across grid sizes from 10×10 to 50×50.

**T = 500 steps:** Linear dynamics horizon chosen to capture transient amplification effects while remaining computationally tractable. Growth ratio threshold >10 for divergence was set prior to metric computation.

**Train/validation split:** 80% training for computing correlations, 20% held-out exclusively for CLI threshold calibration and calibration curves. No metric values from the held-out set were examined during threshold setting.

## Known Limitations

**Basis dependence:** C3 is not invariant under similarity transforms. Practitioners should apply C3 consistently in a fixed parameterization and not compare values across differently parameterized implementations of the same dynamics.

**Synthetic matrices only:** Our evaluation uses controlled synthetic matrices rather than parameters from actual trained models. A preliminary bridge experiment on 30 DSS-style initialized matrices shows consistent behavior (C3 range [0.091, 0.511], ρ = 0.890 with growth ratio).

**Linear dynamics testing:** The benchmark isolates spectral effects but omits optimizer dynamics, batch statistics, and learning rate schedules. Spectral contracts should be positioned as one component of comprehensive pre-training risk assessment.

**Architectural scope:** The contrast currently covers recurrence-inspired (S4/DSS-style) and convolution-inspired (Hyena-style) families. Extension to State-Space Mamba variants and hybrid models would strengthen the empirical foundation.

## File Structure

**Paper files:**
- `paper/main.tex` — Complete LaTeX source
- `paper/main.pdf` — Compiled 13-page submission
- `paper/references.bib` — Bibliography including Kerg et al. 2019
- `paper/figures/` — All figures in PDF and PNG format

**Experimental data:**
- `results/kappa_sweep_fixed_eigenvals.csv` — Fixed-eigenvalue non-normality experiment (90 configurations)
- `results/toeplitz_sweep.csv` — Toeplitz convolution variant data (100 configurations)
- `results/bootstrap_cis.json` — Bootstrap confidence intervals and per-seed statistics
- `results/ablation_c3_robustness.csv` — Parameter sensitivity analysis for C3
- `results/basis_dependence.csv` — Similarity transform invariance study

**Analysis scripts:**
- `analysis/generate_figures.py` — Regenerates all paper figures from data
- `analysis/regression_analysis.py` — Statistical analysis and correlation computation

## How to Reproduce Results

**Generate all figures:**
```bash
cd analysis
python generate_figures.py
```

**Recompile paper:**
```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

**Re-run key experiments:**
- Kappa sweep: See `results/kappa_sweep_fixed_eigenvals.csv` generation code in session logs
- Bootstrap CIs: Statistical analysis uses 2000 bootstrap samples with replacement
- Toeplitz variant: Asymmetric Toeplitz matrices with exponential decay kernels

## Scope and Framing

This benchmark covers two controlled synthetic matrix families motivated by published SSM architectures:

**Recurrence-inspired family:** Diagonal-plus-noise matrices (S4/DSS-style) with controlled non-normality via eigenvector conditioning κ(V) ∈ {1, 10, 100, 500, 1000}.

**Convolution-inspired family:** Circulant and Toeplitz matrices (Hyena-style) with exponential decay kernels and controlled operator norm magnitudes.

Results should not be generalized beyond these families without additional validation. Claims about full training stability require validation beyond the linear dynamics benchmark reported here. The architecture-dependent diagnostic contrast applies specifically to these controlled synthetic families and may not extend to other SSM variants or real training scenarios without further testing.