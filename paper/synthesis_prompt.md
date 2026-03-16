# Paper Synthesis Prompt

Generate paper/outline.md and paper/claims_and_evidence.md using Prompt P6 from CLAUDE.md.

## Confirmed Results:

### Primary Finding (SC-1 + SC-2 achieved):
- **C3 (pseudospectral sensitivity)**: ΔSpearman = +0.158, AUROC = 0.948 on S4-like (n=75, p<0.001)
- **Best trivial baseline on S4-like**: operator norm ρ = 0.677
- **Non-normality regime validated**: κ(V)=714 causes divergence at r=1.0, κ(V)=1 stable at same r

### Architectural Generalization (SC-3 failed → Track B3):
- **Hyena-like**: operator norm ρ = 0.819, C3 ρ = -0.107 (n=125, powered test)
- **No contract outperforms trivial baseline on Hyena**: Different failure mechanism confirmed
- **C1/C4 numerical issues on Hyena**: Circulant structure causes composition underflow

### Success Criteria Status:
- **SC-1**: ✅ PASSED (C3 on S4-like)
- **SC-2**: ✅ PASSED (C3 on S4-like)
- **SC-3**: ❌ FAILED → **Track B3** (failure mode taxonomy)

## Target Publication:

**Venue**: TMLR first (no deadline pressure), then AISTATS if strong reception
**Paper Type**: Empirical study with architectural taxonomy contribution
**Title direction**: Names the finding (failure mode differences), not just the method

## Three Core Claims with Evidence:

1. **Spectral radius is insufficient for recurrence-based SSMs in the non-normal regime**
   - Evidence: C3 ΔSpearman=+0.158 over best trivial baseline, p<0.001
   - Scope: S4-like, non-normal matrices (κ(V)≥100), near-unit spectral radius

2. **Pseudospectral sensitivity detects non-normal transient amplification that trivial baselines miss**
   - Evidence: AUROC=0.948 vs 0.616 for max-eigenvalue baseline
   - Scope: Same as claim 1

3. **Convolution-based SSMs fail via amplitude-based mechanisms where trivial baselines are sufficient**
   - Evidence: Hyena operator norm ρ=0.819, no contract adds ΔSpearman>0
   - Scope: Hyena-like family with circulant structure

**Novel Contribution**: Claim 3 provides the first principled explanation of why recurrence and convolution SSMs need different stability diagnostics.

## Framing Strategy:

**Lead with architectural taxonomy**, not limitations. The finding that different SSM families have **fundamentally different failure modes** is more valuable than a universal metric that works everywhere.

**Honest scope**: "Spectral contracts are optimized for non-normal transient effects in recurrence-based architectures. Convolution-based architectures show amplitude-dominated failure modes where spectral norm alone provides excellent prediction."