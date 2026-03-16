# Claims and Evidence Summary

## Core Scientific Claims

### Claim 1: Spectral Radius Insufficient for Non-Normal Recurrence Regime
**Statement**: "In recurrence-based SSMs with non-normal transition matrices near the stability boundary, spectral radius alone provides inadequate stability prediction."

**Evidence**:
- **Non-normal Case A validation**: κ(V)=714 at r=1.0 causes divergence while κ(V)=1 at same r remains stable
- **Systematic sweep**: ΔSpearman = +0.158 for C3 over best trivial baseline (n=75, p<0.001)
- **Parameter regime**: Eigenvalue radius ∈ [0.99, 1.005], eigenvector conditioning ∈ [100, 1000]

**Scope**: S4-like recurrence architectures with ill-conditioned eigenvector matrices

---

### Claim 2: Pseudospectral Sensitivity Detects Hidden Instabilities
**Statement**: "Pseudospectral sensitivity (C3) captures non-normal transient amplification effects that traditional eigenvalue analysis misses."

**Evidence**:
- **Correlation strength**: Spearman ρ = 0.835 with linear growth outcomes
- **Binary prediction**: AUROC = 0.948 for divergence detection
- **Mechanism validation**: Grid-centered ε-pseudospectrum computation resolves non-normal amplification
- **Outperforms alternatives**: ΔSpearman = +0.158 over operator norm baseline

**Theoretical Foundation**: Trefethen & Embree pseudospectral theory [PROVEN rigor tag]

---

### Claim 3: Architectural Failure Mode Taxonomy
**Statement**: "Recurrence-based and convolution-based SSM families exhibit fundamentally different failure modes requiring different diagnostic approaches."

**Evidence**:

**Recurrence SSMs (S4-like)**:
- Failure mode: Non-normal transient amplification
- Trivial baseline performance: operator norm ρ = 0.677 (insufficient)
- Required diagnostic: Pseudospectral sensitivity (ρ = 0.835)

**Convolution SSMs (Hyena-like)**:
- Failure mode: Amplitude saturation
- Trivial baseline performance: operator norm ρ = 0.819 (excellent)
- Contract metric performance: All ρ < 0.45 (no added value)

**Statistical Power**: Both analyses adequately powered (n=75, n=125) for reliable conclusions

---

## Methodological Contributions

### Non-Circular Outcome Generation
**Innovation**: Linear dynamics testing vs. eigenvalue-derived synthetic labels
- **Problem identified**: Previous circular correlation (outcomes derived from predictors)
- **Solution implemented**: Actual iterative dynamics with growth ratio measurement
- **Validation**: Demonstrates legitimate stability differences in controlled experiments

### Regime-Appropriate Testing
**Innovation**: Parameter space selection based on mathematical theory
- **Problem identified**: Testing contracts on diagonal matrices where they cannot succeed
- **Solution implemented**: Non-normal regime (controlled eigenvector conditioning)
- **Result**: Reveals contracts work in intended domain, fail outside it

---

## Practical Contributions

### CLI Tool with Architectural Boundaries
- **Red/yellow/green** risk assessment with calibrated thresholds
- **Family-specific** diagnostic recommendations
- **Cost estimates** and approximation modes for large models
- **Conservative calibration** to avoid false security assessments

### Benchmark Suite
- **Case A/B validation** for trivial baseline failure modes
- **Non-circular outcomes** for legitimate stability testing
- **Multi-family architecture** support with family-specific matrix generation
- **Reusable framework** for testing additional SSM families

---

## Limitations and Scope Boundaries

## Claim: C1 Architecturally Undefined for Convolution SSMs

**Statement**: "Effective transition condition growth (C1) is not a meaningful diagnostic for Hyena-like architectures because their broad-spectrum convolution filter design structurally requires small eigenvalues (min ≈ 0.006), causing C1 to saturate at the numerical cap for all configurations regardless of stability outcome."

**Evidence**:
- **Structural verification**: min|λ| = 0.006091 consistent across all seeds (architectural property)
- **Eigenvalue spread**: ~156x range inherent to circulant convolution design
- **C1 saturation**: All 125 Hyena configurations hit 1e8 cap due to 0.006^T underflow
- **Architecture-specific**: S4-like matrices have min|λ| ≈ 0.67, avoiding underflow

**Rigor Tag**: [EMPIRICAL] confirmed by sweep; theoretical explanation [MOTIVATED]

**Sufficient Evidence**: YES - demonstrates structural incompatibility, not implementation failure

**Reviewer Response**: "Why not use shorter T?" → T=5 gives computable values but they don't predict stability at operationally relevant T=500; the metric is inapplicable at diagnostic horizons.

---

## Enhanced Failure Mode Taxonomy

| Architecture | Primary Failure Mode | Trivial Baseline Performance | C3 Performance | C1 Applicability |
|-------------|---------------------|------------------------------|-----------------|-------------------|
| **S4-like** | Non-normal transient amplification | Operator norm ρ=0.677 (insufficient) | ρ=0.835 (excellent) | Finite values, competitive |
| **Hyena-like** | Amplitude saturation | Operator norm ρ=0.819 (excellent) | ρ=-0.107 (no signal) | **Structurally undefined** |

**Architectural Insight**: The failure mechanisms are **structurally determined** by eigenvalue distributions, and appropriate metrics match the underlying mathematical structure.

---

### Explicitly Acknowledged Limitations
1. **Recurrence-focused**: C3 optimized for recurrence-based non-normal effects
2. **Linear regime**: Dynamics testing in SSM linear regime, not full training simulation
3. **Small scale validation**: N≤64 for computational feasibility
4. **Architecture-specific**: Different contracts needed for different SSM families

### Why These Are Strengths
- **Honest scope** makes findings more credible than over-broad claims
- **Clear boundaries** enable practitioners to know when contracts apply
- **Methodological rigor** prevents inappropriate generalization
- **Foundation for extension** to additional SSM families with appropriate contract design

---

## Success Criteria Achievement

| Criterion | Threshold | Result | Status |
|-----------|-----------|---------|--------|
| **SC-1** | ΔSpearman ≥ 0.10 | C3: +0.158 | ✅ **ACHIEVED** |
| **SC-2** | AUROC ≥ 0.75 | C3: 0.948 | ✅ **ACHIEVED** |
| **SC-3** | Generalization ≥2 families | C3 Hyena: ρ=-0.107 | ❌ Failed → Track B3 |

**Outcome**: Track B3 - Architectural failure mode taxonomy (stronger contribution than universal generalization)

---

## Paper Positioning

### Venue Strategy
**Primary target**: TMLR (honest empirical study with clear scope)
**Secondary target**: AISTATS (methodological contribution)
**Avoid**: ICLR/NeurIPS (requires broader generalization claims)

### Contribution Framing
**Lead with taxonomy discovery**, not limitation acknowledgment
- "First systematic analysis of architectural failure mode differences in SSMs"
- "When contracts beat baselines vs. when baselines are sufficient"
- "Principled framework for testing stability diagnostics in appropriate regimes"

### Competitive Positioning
- **vs. LRU/HiPPO**: Complementary (they fix initialization, we diagnose problems)
- **vs. FreeInit**: Different approach (empirical validation first, theory second)
- **vs. Universal stability**: More honest (acknowledges architectural boundaries)

This positions the work as methodologically rigorous and scientifically honest rather than over-claiming scope.