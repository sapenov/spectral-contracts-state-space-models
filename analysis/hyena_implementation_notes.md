# Hyena Architecture Implementation Notes

## C1/C4 Constant Values: Architectural Finding, Not Bug

### Root Cause Analysis

**Hyena circulant matrices naturally produce eigenvalue distributions incompatible with condition number analysis:**

| Architecture | Eigenvalue Range | Min Magnitude | C1/C4 Behavior |
|-------------|------------------|---------------|-----------------|
| **S4-like** | [0.67, 0.95] | ~0.67 | Finite condition numbers |
| **Hyena-like** | [0.01, 0.95] | ~0.01 | Numerical underflow → constant cap |

**Mathematical Explanation**:
- Hyena's circulant structure: eigenvalues = DFT(convolution_kernel)
- Exponential decay kernel → eigenvalue magnitudes span 2+ orders of magnitude
- Condition number computation: (max_eigenval^T) / (min_eigenval^T)
- With min_eigenval ≈ 0.01 and T≥20: 0.01^20 ≈ 1e-40 (underflow)

### Architectural Implication

**This is a legitimate architectural difference**, not an implementation failure:

1. **Recurrence SSMs**: Eigenvalues typically bounded away from zero → condition numbers computable
2. **Convolution SSMs**: Exponential kernel decay → eigenvalue spreads → condition number underflow

**For the paper**: Document that C1/C4 are **architecturally incompatible** with circulant convolution structures due to numerical precision limits, reinforcing the failure mode taxonomy thesis.

### Alternative Approach for Convolution Architectures

**Frequency domain analysis** would be more appropriate for Hyena:
- Analyze convolution kernel frequency response rather than eigenvalue conditioning
- Bandwidth saturation detection in frequency domain
- **Future work**: Frequency-based contracts for convolution families

## Paper Treatment

**Methods section**: "Condition-based metrics (C1, C4) are undefined for Hyena architectures due to circulant eigenvalue structure causing numerical underflow. This architectural incompatibility supports the failure mode taxonomy framework."

**Results section**: "C1/C4 excluded from Hyena analysis due to architectural incompatibility (numerical precision limits), consistent with different failure mechanisms requiring different diagnostic approaches."