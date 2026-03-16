# Architectural Failure Mode Taxonomy in State-Space Models: When Spectral Contracts Beat Trivial Baselines

## Abstract (200 words)

Pre-training stability failures in state-space models (SSMs) cost significant computational resources. While spectral radius provides a basic stability diagnostic, recent SSM architectures exhibit failure modes that spectral radius alone cannot predict. We introduce **spectral contracts** — computable pre-training diagnostics that capture stability risks beyond eigenvalue magnitude — and establish when they provide genuine predictive value over trivial baselines.

Through systematic evaluation across recurrence-based (S4-like) and convolution-based (Hyena-like) SSM families, we identify **architecturally distinct failure modes**. In recurrence-based SSMs with non-normal transition matrices, pseudospectral sensitivity achieves Spearman ρ=0.835 with growth ratio outcomes, outperforming trivial baselines by ΔSpearman=+0.158 (p<0.001). However, convolution-based SSMs exhibit **amplitude-dominated failure modes** where operator norm alone achieves ρ=0.819, and spectral contracts provide no additional predictive value.

We provide a **failure mode taxonomy**: recurrence SSMs require non-normality-aware diagnostics when eigenvector conditioning is high, while convolution SSMs fail via amplitude saturation detectable with spectral norms. We release a calibrated CLI tool and benchmark suite for pre-training stability assessment, with clear architectural scope boundaries.

**Artifact**: `ssm-contracts` CLI tool with red/yellow/green stability assessment.

---

## 1. Introduction (~2 pages)

### 1.1 Problem: Costly SSM Training Failures
- Training instabilities in long-sequence SSM training
- Cost of failed multi-day GPU runs (cite Mamba, S4 training costs)
- Current practice: spectral radius as only pre-training diagnostic

### 1.2 Insufficiency of Spectral Radius Alone
- **Case A demonstration**: matrices with all eigenvalues<1 that still cause training divergence
- **Case B demonstration**: matrices with eigenvalues≈1 that train stably
- Mathematical explanation: non-normal transient amplification vs. spectral stability

### 1.3 Spectral Contracts Framework
- Definition: cheap pre-training diagnostics with predictive power beyond spectral radius
- Cost constraint: O(N²·L) or better for practical use
- Success criteria: ΔSpearman ≥ 0.10 over trivial baselines, AUROC ≥ 0.75

### 1.4 Architectural Scope Discovery
- Initial hypothesis: universal contracts across SSM families
- **Actual finding**: architectural failure mode taxonomy
- Preview: recurrence needs non-normality detection, convolution needs amplitude control

---

## 2. Related Work (~1.5 pages)

### 2.1 SSM Stability and Initialization
- LRU eigenvalue constraints (Orvieto et al.)
- HiPPO initialization theory (Gu et al.)
- S4/DSS structured approaches
- Gap: no systematic pre-training diagnostics

### 2.2 Pseudospectral Analysis in Linear Systems
- Trefethen & Embree: pseudospectra for non-normal matrices
- Transient amplification vs asymptotic stability
- Application to recurrence stability (novel)

### 2.3 Deep Learning Stability Diagnostics
- Jacobian conditioning (Pennington et al.)
- Gradient explosion/vanishing detection
- Limited application to structured SSMs

---

## 3. Spectral Contract Metrics (~2 pages)

### 3.1 Trivial Baselines
- Max eigenvalue magnitude: max_l max_i |λ_i(A_l)|
- Operator norm: max_l ‖A_l‖₂
- Performance: surprisingly strong on Hyena (ρ=0.819), insufficient on non-normal S4 (ρ=0.677)

### 3.2 Contract Metrics Taxonomy
**C3 - Pseudospectral Sensitivity** [PROVEN]
- ε-pseudospectral radius computation
- Theoretical foundation: Trefethen & Embree non-normal amplification
- **Lead metric**: ΔSpearman=+0.158 on S4-like

**C1 - Condition Growth** [MOTIVATED]
- Condition number of composed operator A₁ᵀ·...·Aₗᵀ
- Cost: O(N³·L), approximations available
- Performance: competitive but not optimal

**C2, C4, C5, C6** - [Details and performance results]

### 3.3 Computational Cost Analysis
- Measured wall-clock times at N=64, L=8
- Approximation methods for practical deployment

---

## 4. Benchmark Suite and Evaluation Framework (~2 pages)

### 4.1 Non-Circular Outcome Generation
- **Critical methodological contribution**: linear dynamics testing
- Growth ratio measurement vs. eigenvalue-derived labels
- Memory retention outcomes for complementary validation

### 4.2 Non-Normal Regime Testing
- **Parameter space**: eigenvalue_radius × condition_V sweep
- Why normal matrices (diagonal) cannot test non-normal contracts
- Case A/Case B validation methodology

### 4.3 Statistical Power Analysis
- S4-like: n=75, adequately powered for correlation detection
- Hyena-like: n=125, adequately powered to detect/reject generalization

---

## 5. Results: Architectural Failure Mode Taxonomy (~3 pages)

### 5.1 Recurrence-Based SSMs (S4-like)
**Table: S4-like Contract Performance**

| Metric | Spearman ρ | ΔSpearman | AUROC | Status |
|--------|------------|-----------|-------|--------|
| trivial_max_operator_norm | 0.677 | 0 (baseline) | 0.934 | Strong baseline |
| **contract_C3** | **0.835** | **+0.158** | **0.948** | **FULL CONTRACT** |
| contract_C1 | 0.582 | -0.095 | 0.879 | THRESHOLD_ONLY |

**Key finding**: Non-normal transient amplification requires specialized detection

### 5.2 Convolution-Based SSMs (Hyena-like)
**Table: Hyena-like Contract Performance**

| Metric | Spearman ρ | ΔSpearman | Status |
|--------|------------|-----------|--------|
| **trivial_max_operator_norm** | **0.819** | **0 (baseline)** | **Already optimal** |
| contract_C3 | -0.107 | -0.926 | No signal |
| contract_C6 | 0.422 | -0.397 | Below threshold |

**Key finding**: Amplitude-dominated failure mode where spectral norm is sufficient

### 5.3 Failure Mode Taxonomy
**Architectural Classification Framework:**

| SSM Family | Failure Mode | Optimal Diagnostic | Mechanism |
|------------|--------------|-------------------|-----------|
| **Recurrence** (S4, DSS) | Non-normal amplification | Pseudospectral sensitivity | Eigenvector ill-conditioning |
| **Convolution** (Hyena) | Amplitude saturation | Operator norm | Filter magnitude blowup |

---

## 6. The ssm-contracts Tool (~1 page)

### 6.1 CLI Implementation
- Red/yellow/green risk assessment
- Configuration file support
- Architectural family detection

### 6.2 Threshold Calibration
- Held-out validation methodology
- Family-specific threshold documentation
- Conservative calibration for production use

### 6.3 Usage Examples and Limitations
- When to use contracts vs. trivial baselines
- Architectural scope boundaries clearly documented

---

## 7. Discussion (~1.5 pages)

### 7.1 When Are Contracts Necessary?
- Well-initialized SSMs (LRU, HiPPO) are largely normal → spectral radius sufficient
- Experimental architectures or drift during training → contracts valuable
- **Practical guidance**: Start with trivial baselines, escalate to contracts when needed

### 7.2 Architectural Implications
- Recurrence design: monitor non-normality during initialization
- Convolution design: operator norm bounds are sufficient
- Hybrid architectures: require mixed diagnostic approach

### 7.3 Future Work
- Extension to selective SSMs (Mamba) with input-dependent gating
- Real training validation beyond linear dynamics
- Integration with existing SSM initialization methods

---

## 8. Conclusion (~0.5 pages)

**Three contributions:**
1. **Empirical validation** that spectral contracts outperform trivial baselines in the non-normal recurrence regime
2. **Architectural taxonomy** distinguishing recurrence vs. convolution failure modes
3. **Methodological framework** for testing stability diagnostics in appropriate mathematical regimes

**Practical impact**: Pre-training stability assessment tool with clear scope boundaries
**Theoretical impact**: First systematic analysis of failure mode architectural dependence in SSMs

---

## Target Venue: TMLR
- **Rationale**: Careful empirical study with genuine architectural boundaries
- **Strength**: Honest scope limitations make findings more credible
- **Novelty**: Failure mode taxonomy is unexplored in SSM literature

**Estimated Length**: 8-9 pages (well within TMLR scope)
**Anticipated Review Concerns**: Scope limitations (addressed head-on), numerical stability (documented and fixed), generalization claims (appropriately bounded)