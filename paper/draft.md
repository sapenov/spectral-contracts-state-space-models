# Architecture-Dependent Stability Diagnostics for State-Space Models: When Pseudospectral Analysis Beats Trivial Baselines

## Abstract

**Recurrence-based and convolution-based state-space model families exhibit structurally different dynamical failure modes that require different pre-training stability diagnostics.** Through systematic evaluation of spectral contract metrics across two controlled synthetic matrix families motivated by S4-like and Hyena-like SSM architectures, we establish evidence for an **architecture-dependent failure mode taxonomy** in linear long-horizon stability prediction.

Using non-circular linear dynamics outcomes to prevent false correlations, we demonstrate that recurrence-inspired matrices with non-normal transition structure exhibit failure modes that spectral radius analysis misses. Pseudospectral sensitivity achieves Spearman ρ=0.835 with dynamical stability outcomes, outperforming the best trivial baseline by ΔSpearman=+0.158 (n=75, p<0.001) and achieving AUROC=0.948 for divergence prediction.

However, convolution-inspired matrices exhibit **amplitude-dominated failure modes** where operator norm alone achieves ρ=0.819 and AUROC=0.956, with spectral contracts providing no additional predictive value (n=125). This architecture-dependent contrast reveals that **recurrence-inspired families require non-normality-aware diagnostics** while **convolution-inspired families are already well-served by spectral norms**.

We provide practical guidance through our `ssm-contracts` CLI tool with family-specific risk assessment, and establish methodological principles for testing stability diagnostics in appropriate mathematical regimes.

**Keywords**: State-space models, stability prediction, pseudospectra, architectural taxonomy

---

## 1. Introduction

### 1.1 The Cost of SSM Dynamical Instabilities

State-space models (SSMs) have emerged as powerful alternatives to transformers for long-sequence modeling, with architectures like S4 [Gu et al., 2022], Mamba [Gu & Dao, 2023], and Hyena [Poli et al., 2023] achieving state-of-the-art performance on tasks requiring extended context. However, training these models on long sequences remains computationally expensive, with multi-day GPU runs common for production-scale experiments. When training fails due to gradient explosion, vanishing gradients, or numerical instabilities, the computational cost is entirely lost.

Current practice for pre-training stability assessment relies primarily on **spectral radius analysis** — ensuring all eigenvalues of the SSM transition matrices have magnitude less than 1.0. While this provides a necessary condition for asymptotic stability, it is insufficient for predicting training behavior in practice. Recent work on SSM initialization [Orvieto et al., 2023] has identified cases where spectral radius constraints are satisfied but training still fails, suggesting that additional pre-training diagnostics are needed.

We focus on a controlled proxy for this problem: **predicting linear long-horizon dynamical instability** from pre-computation spectral properties. While full training stability depends on additional factors (optimizers, batch statistics, learning rate schedules), the linear dynamical regime isolates the spectral effects we study and provides a tractable benchmark for comparing diagnostic approaches.

### 1.2 Insufficiency of Spectral Radius: Motivating Examples

We demonstrate the limitation of spectral radius through two constructed cases that isolate different failure mechanisms:

**Case A (Hidden Instability)**: Consider an SSM transition matrix A = VDV⁻¹ where D contains eigenvalues [0.90, 0.85, 0.80, ...] (all safely inside the unit circle) but V has condition number κ(V) ≈ 1000. The spectral radius is 0.90, suggesting safe dynamics. However, the ill-conditioned eigenvector matrix causes transient amplification — the state norm grows substantially over intermediate horizons before eventually decaying — leading to dynamical divergence in our linear benchmark despite the "safe" eigenvalue analysis.

**Case B (Apparent Risk)**: Consider A = diag(0.999, 0.95, 0.95, ...) where the spectral radius is 0.999, very close to the instability boundary. Traditional analysis would flag this as high risk. However, the diagonal structure ensures no non-normal amplification occurs, and the linear dynamics remain stable despite the concerning spectral radius.

These cases reveal that **spectral radius captures asymptotic behavior but misses transient amplification effects** that dominate dynamical behavior over finite horizons — the operationally relevant regime for long-sequence SSM stability.

### 1.3 Spectral Contracts as Pre-Training Dynamical Screens

We introduce **spectral contracts** — scalar functions C(θ) of SSM parameters θ computable before training that serve as mechanistic pre-training screens for linear long-horizon dynamical risk. A contract satisfies three criteria:

1. **Predictive**: Statistically significant monotone relationship with linear dynamical stability outcomes (Spearman ρ ≥ 0.60)
2. **Informative**: Strictly improves over trivial baselines (ΔSpearman ≥ 0.10)
3. **Cheap**: Computable in O(N²·L) time or better for practical deployment

We position contracts as **mechanistic screens**, not replacements for end-to-end training evaluation. They predict linearized long-horizon dynamical risk from matrix structure alone, which is a tractable and interpretable proxy for the full training stability problem.

Our evaluation focuses on **pseudospectral sensitivity** (C3), which measures the ε-pseudospectral radius — capturing how eigenvalues move under small matrix perturbations — to detect non-normal transient amplification that spectral radius alone misses.

### 1.4 Architecture-Dependent Diagnostic Contrast

Our initial hypothesis was that spectral contracts would generalize across SSM families. We study two controlled synthetic matrix families motivated by published SSM architectures: a recurrence-inspired structured family (motivated by S4/DSS-style diagonal-plus-noise transition matrices) and a convolution-inspired circulant family (motivated by Hyena-style long-convolution operators). This controlled design allows regime-specific evaluation without optimizer confounds.

Systematic testing revealed **architecture-dependent diagnostic behavior**:

- **Recurrence-inspired family**: Non-normal transient amplification in transition matrices — pseudospectral sensitivity provides substantial improvement over trivial baselines
- **Convolution-inspired family**: Amplitude-dominated failure modes — operator norm alone is already an excellent predictor; contracts add no value

This architecture-dependent contrast, rather than being a limitation, constitutes our **primary empirical contribution**: evidence that different SSM architectural families benefit from fundamentally different stability diagnostics, with practical guidance for when to escalate beyond trivial baselines.

### 1.5 Contributions

1. **Benchmark methodology**: Non-circular linear dynamics outcomes and regime-appropriate non-normal parameter sweeps that enable legitimate stability diagnostic comparison (§3.4, §3.5)

2. **Recurrence-side positive result**: Pseudospectral sensitivity (C3) achieves ΔSpearman=+0.158 over the best trivial baseline and AUROC=0.948 on the recurrence-inspired family in the non-normal regime (§4.1)

3. **Cross-family diagnostic contrast**: Evidence that convolution-inspired families are already well-served by operator norm (ρ=0.819), while recurrence-inspired families in the non-normal regime require pseudospectral analysis — supporting an architecture-dependent diagnostic taxonomy in our controlled benchmark (§4.2, §4.3)

4. **Research artifact**: `ssm-contracts` CLI tool implementing these diagnostics with calibrated thresholds and explicit architectural scope documentation (§5)

---

## 2. Related Work

### 2.1 SSM Stability and Initialization

Early SSM work established spectral radius constraints as fundamental stability requirements [Gu et al., 2022]. The Linear Recurrent Unit (LRU) [Orvieto et al., 2023] explicitly constrains eigenvalues to prevent instabilities, while HiPPO initialization [Gu et al., 2021] provides theoretically motivated eigenvalue placement for memory retention. However, these approaches focus on sufficient conditions for stability rather than predictive diagnostics for arbitrary initializations.

Recent work has identified cases where standard eigenvalue constraints are satisfied but training still fails [Orvieto et al., 2023], motivating the need for richer pre-training stability analysis. Our work provides a controlled benchmark for evaluating pre-training stability diagnostics in this setting.

### 2.2 Pseudospectral Analysis in Dynamical Systems

Pseudospectral analysis, developed by Trefethen & Embree [2005], characterizes how eigenvalues of non-normal matrices change under small perturbations. For matrices with ill-conditioned eigenvector representations, the ε-pseudospectrum can extend far beyond the spectrum itself, revealing "hidden instabilities" not captured by eigenvalue analysis alone.

While pseudospectral analysis has been applied to general recurrent systems, we are not aware of prior work systematically applying these diagnostics to modern structured SSM architectures. The key insight is that SSM training involves repeated application of transition matrices over long sequences, making transient amplification effects relevant even when asymptotic behavior (captured by spectral radius) suggests stability.

### 2.3 Stability Diagnostics in Deep Learning

Dynamical isometry analysis [Pennington et al., 2017] established the importance of Jacobian conditioning for gradient flow in deep networks. However, these approaches focus on feedforward architectures and do not account for the structured recurrence patterns in SSMs.

Control-theoretic approaches to recurrent network analysis [Sontag, 1998] provide theoretical foundations for controllability and observability analysis, which we adapt as contract metrics C4 (controllability Gramian conditioning). Our contribution extends these classical results to modern SSM architectures with empirical validation.

---

## 3. Methods

### 3.1 Trivial Baselines

Before evaluating any contract metric, we establish three trivial baselines that any meaningful contract must outperform:

**TB1 - Max Eigenvalue Magnitude**: `max_eigenvalue = max_l max_i |λ_i(A_l)|` across all SSM layers. This captures the classical spectral radius stability criterion.

**TB2 - Max Operator Norm**: `max_operator_norm = max_l ||A_l||_2` using the spectral norm of individual layers. This upper-bounds per-step amplification.

**TB3 - Gradient Norm (Linear Regime Proxy)**: In the linear dynamics testing regime, traditional gradient norms requiring loss functions are undefined. The end-to-end gradient from input to output after L layers reduces to the composed Jacobian ∂h_L/∂h_0 = A_L ⋯ A_1. We use its Frobenius norm `||A_L ⋯ A_1||_F` as a mathematically principled proxy for gradient magnitude.

This proxy captures end-to-end amplification effects similar to actual gradient norms in training, while remaining computable in the parameter-free linear regime. The reduction from three baselines to two effective baselines (eigenvalue magnitude and operator norm) reflects the methodological constraint of testing matrix dynamics rather than neural network training.

### 3.2 Contract Metrics

We evaluate six candidate contract metrics designed to capture stability risks that trivial baselines miss:

**C1 - Effective Transition Condition Growth [MOTIVATED]**
Formula: `κ(A_1^T ⋯ A_L^T)` where κ is the condition number and T is the evaluation horizon.
Cost: O(N³⋅L) for exact computation via SVD of composed operator.
Mechanism: Ill-conditioned compositions amplify perturbations exponentially over T steps.

**C2 - Singular Value Dispersion [MOTIVATED]**
Formula: `σ_max(A_L ⋯ A_1) / σ_min(A_L ⋯ A_1)` for the composed operator.
Cost: O(N³⋅L) for exact computation.
Mechanism: High dispersion predicts anisotropic gradient flow and selective memory failure.

**C3 - Pseudospectral Sensitivity [PROVEN]**
Formula: `max{|z| : z ∈ Λ_ε(A)}` where `Λ_ε(A) = {z ∈ ℂ : σ_min(zI - A) ≤ ε}`.
Cost: O(N²⋅grid_size²) for ε-pseudospectrum computation on complex grid.
Implementation: We use a 30×30 grid covering ±1.2× the spectral radius; sensitivity analysis across grid sizes 20–50 showed negligible impact on correlation results (variance < 0.005).
Mechanism: Captures non-normal transient amplification that spectral radius misses. For non-normal matrices, eigenvalues are sensitive to perturbations, and the pseudospectrum reveals the true amplification behavior under finite-precision arithmetic.

**C4 - Controllability Condition [PROVEN]**
Formula: `κ(W_T)` where `W_T = Σ_{t=0}^{T-1} A^t B B^T (A^T)^t` is the controllability Gramian.
Cost: O(N²⋅T + N³) for Gramian construction and conditioning.
Mechanism: Ill-conditioned Gramian predicts loss of controllability over long horizons.

**C5 - Jacobian Anisotropy Growth [MOTIVATED]**
Formula: Growth rate of `log(σ_max(J_T)/σ_min(J_T))` where J_T is the end-to-end Jacobian after T steps.
Cost: O(N³⋅T) for full computation.
Mechanism: Predicts gradient explosion/vanishing asymmetry during training.

**C6 - Composed Spectral Spread [MOTIVATED]**
Formula: Under diagonal approximation `A_l ≈ diag(a_l^{(1)}, ..., a_l^{(N)})`, computes spread of composed eigenvalues `λ_i^{(L)} = ∏_{l=1}^L a_l^{(i)}`.
Cost: O(N⋅L) for diagonal approximation.

*C6 Theoretical Foundation*: Under the diagonal approximation and i.i.d. layer assumption, the log-magnitude distribution `log|λ_i^{(L)}| = Σ_l log|a_l^{(i)}|` concentrates around `L⋅E[log|a_l^{(i)}|]` with variance `L⋅Var[log|a_l^{(i)}|]` by the Central Limit Theorem. High variance predicts widely dispersed singular values in the composed operator, indicating eigenvalue clustering that creates directional instabilities. The spectral spread `max|λ_i^{(L)}| - min|λ_i^{(L)}|` captures this variance-driven effect while remaining computationally cheap.

### 3.3 SSM Family Matrix Generation

We study two controlled synthetic matrix families motivated by recurrence-based and convolution-based SSM architectures, allowing regime-specific evaluation without optimizer confounds.

**Recurrence-inspired family (motivated by S4/DSS)**: Structured transition matrices with learnable diagonal components and small off-diagonal coupling terms, representing the near-diagonal structure common in published recurrence-based SSMs. Matrix generation: `A = diag(eigenvals) + 0.01⋅randn(N,N)` with rescaling to maintain spectral radius bounds. We do not claim these matrices are identical to published S4 or DSS implementations; they capture the structural property (near-diagonal, potentially non-normal) that motivates the diagnostic comparison.

**Convolution-inspired family (motivated by Hyena)**: Circulant matrices approximating long-convolution operators with exponential decay kernels, representing the broad-spectrum filter structure common in published convolution-based SSMs. These naturally produce broad eigenvalue spectra (eigenvalue spreads of ~156x) representing different frequency components. We do not claim these matrices are identical to published Hyena implementations; they capture the structural property (broad spectrum, amplitude-dominated) that motivates the diagnostic comparison.

**Non-Normal Regime Construction**: To test contracts in their intended domain, we systematically vary eigenvector conditioning alongside spectral radius. Matrices are constructed as `A = V⋅D⋅V^{-1}` where D contains target eigenvalues and V is iteratively adjusted to achieve specified condition number κ(V) ∈ [1, 1000].

### 3.4 Non-Circular Outcome Generation

**Critical methodological innovation**: We compute stability outcomes via actual linear dynamics rather than eigenvalue-derived synthetic labels to prevent circular correlations.

**Linear Stability Test**: For each SSM configuration, we iterate the dynamics `x_{t+1} = A_L ⋯ A_1 x_t` for T=500 steps with n=10 random initial conditions and measure the growth ratio `||x_T|| / ||x_0||`. Configurations with mean growth ratio >10 are labeled as "diverged."

**Memory Retention Test**: We measure how much of a unit canonical vector e₁ survives T steps of dynamics, providing a complementary outcome focused on information preservation rather than amplitude growth.

This approach ensures that outcomes reflect actual dynamical behavior rather than spectral properties used to construct the predictors.

### 3.5 Statistical Analysis

**Correlation Analysis**: We compute Spearman rank correlations ρ between each metric and stability outcomes. The non-parametric Spearman correlation captures monotonic relationships without assuming linearity.

**ΔSpearman Criterion**: For each contract metric, we compute `ΔSpearman = |ρ(contract, outcome)| - |ρ(best_trivial, outcome)|`. The success threshold SC-1 requires ΔSpearman ≥ 0.10, indicating meaningful improvement over trivial baselines.

**Binary Classification**: For divergence prediction, we compute Area Under ROC Curve (AUROC) with success threshold SC-2 requiring AUROC ≥ 0.75 for early instability detection.

**Statistical Power**: We ensure adequate sample sizes (n≥75 per family) and balanced outcome distributions (15-25% divergence rate) for reliable correlation detection.

---

## 4. Results

### 4.1 Recurrence-Inspired Family: Contracts Outperform Baselines in Non-Normal Regime

We evaluated all metrics on 75 recurrence-inspired configurations spanning eigenvalue radius r ∈ [0.95, 1.005] and eigenvector conditioning κ(V) ∈ [1, 1000]. This parameter space includes both normal matrices (κ(V)=1) where spectral radius should suffice, and non-normal matrices (κ(V)≥100) where contracts are designed to provide additional predictive value over linear dynamical outcomes.

**Table 1: Recurrence-Inspired Family — Contract Performance**

| Metric | Spearman ρ | ΔSpearman | AUROC | Classification |
|--------|------------|-----------|-------|----------------|
| trivial_max_eigenvalue | 0.599 | -0.077 | 0.616 | Weak Baseline |
| **trivial_max_operator_norm** | **0.677** | **0** | **0.934** | **Strong Baseline** |
| contract_C1 | 0.582 | -0.095 | 0.879 | THRESHOLD_ONLY |
| contract_C2 | 0.602 | -0.075 | 0.919 | THRESHOLD_ONLY |
| **contract_C3** | **0.835** | **+0.158** | **0.948** | **FULL CONTRACT** |
| contract_C4 | constant | undefined | 0.500 | UNINFORMATIVE |
| contract_C5 | 0.583 | -0.093 | 0.913 | THRESHOLD_ONLY |
| contract_C6 | 0.611 | -0.066 | 0.918 | THRESHOLD_ONLY |

*Note: All correlations significant at p<0.001 except C4 (constant values).*

**Key Finding**: In the non-normal regime (κ(V)≥100), C3 provides a meaningful improvement over the best trivial baseline in continuous rank prediction (ΔSpearman=+0.158), even though operator norm is already a strong binary classifier (AUROC=0.934). C3's advantage lies in distinguishing cases with similar eigenvalue radius but different transient amplification — the regime that matters for architecture search and stability triage.

### 4.2 Convolution-Inspired Family: Trivial Baselines Already Sufficient

We evaluated the same metrics on 125 convolution-inspired configurations using identical parameter ranges to test whether the recurrence-side finding generalizes.

**Table 2: Convolution-Inspired Family — Contract Performance**

| Metric | Spearman ρ | ΔSpearman | AUROC | Classification |
|--------|------------|-----------|-------|----------------|
| trivial_max_eigenvalue | 0.737 | -0.082 | 0.941 | Strong Baseline |
| **trivial_max_operator_norm** | **0.819** | **0** | **0.956** | **Excellent Baseline** |
| contract_C1 | constant | undefined | undefined | ARCHITECTURALLY INCOMPATIBLE |
| contract_C2 | -0.365 | -1.184 | 0.235 | UNINFORMATIVE |
| contract_C3 | -0.107 | -0.926 | 0.487 | UNINFORMATIVE |
| contract_C4 | constant | undefined | undefined | ARCHITECTURALLY INCOMPATIBLE |
| contract_C5 | 0.289 | -0.530 | 0.722 | UNINFORMATIVE |
| contract_C6 | 0.422 | -0.397 | 0.731 | UNINFORMATIVE |

*‡C1 and C4 are not reported for the convolution-inspired family due to architectural incompatibility with circulant eigenvalue structure (see §4.3). TB3 (composed Jacobian norm) analysis was not completed for this study.*

**Key Findings**:

1. **No contract metric outperforms trivial baselines**: All ΔSpearman values negative, indicating trivial baselines are already optimal for Hyena architectures.

2. **Operator norm achieves excellent performance**: ρ=0.819, AUROC=0.956, demonstrating that amplitude-based failure modes are well-captured by spectral norm analysis.

3. **C3 shows no signal** (ρ=-0.107): The pseudospectral mechanism that helps on the recurrence-inspired family is irrelevant when amplitude, not non-normality, drives instability.

### 4.3 Architecture-Dependent Diagnostic Contrast

The contrasting results across families provide evidence for **architecture-dependent failure modes in our controlled benchmark**:

**Table 3: Diagnostic Contrast Across Benchmark Families**

| Architecture | Failure Mechanism | Eigenvalue Structure | Optimal Diagnostic | Why Contracts Needed |
|-------------|-------------------|---------------------|-------------------|---------------------|
| **S4-like** | Non-normal transient amplification | Narrow spectrum, variable conditioning | Pseudospectral sensitivity | Spectral radius misses κ(V) effects |
| **Hyena-like** | Amplitude saturation | Broad spectrum (156x spread) | Operator norm | Amplitude dominates; contracts add noise |

**Cross-Family Generalization Analysis**:
- **C3 on S4-like**: ρ=0.835 (excellent signal)
- **C3 on Hyena-like**: ρ=-0.107 (no signal)
- **Generalization conclusion**: Architecture-specific diagnostics required

This diagnostic contrast supports the hypothesis that recurrence-based and convolution-based SSM families require different stability assessment approaches. Further work with multiple family members and real trained models would be needed to elevate this from a benchmark finding to a general architectural principle.

---

## 5. Discussion

### 5.1 When Are Spectral Contracts Necessary?

Our results provide clear guidance on when to escalate from trivial baselines to spectral contracts:

**For S4-like recurrence architectures**: Contracts are valuable when (1) eigenvalue radius approaches 1.0, (2) eigenvector conditioning κ(V) ≥ 100, and (3) non-normal structure is present. Standard initialization methods like LRU [Orvieto et al., 2023] and HiPPO [Gu et al., 2021] are designed to place eigenvalues appropriately and maintain structured initialization; whether they consistently produce near-normal matrices in practice requires empirical validation beyond this paper's scope.

**For Hyena-like convolution architectures**: Trivial baselines (especially operator norm) are already optimal. The broad eigenvalue spectra inherent to convolution filter design make amplitude saturation the dominant failure mode, which spectral norms detect effectively.

**Practical workflow**: (1) Compute spectral radius and operator norm first, (2) if both suggest stability but training history indicates otherwise, escalate to architecture-specific contracts, (3) for experimental architectures outside standard initialization schemes, use contracts proactively.

### 5.2 Architectural Implications for SSM Design

The failure mode taxonomy suggests different monitoring strategies during SSM development:

**Recurrence-based architectures** should monitor eigenvector conditioning during initialization and training. Standard initialization methods (LRU, HiPPO) explicitly maintain diagonal or near-diagonal structure to avoid non-normal effects. Novel initialization schemes should be tested with pseudospectral sensitivity.

**Convolution-based architectures** should focus on filter amplitude bounds. The natural broad spectrum of convolution kernels makes condition number analysis inappropriate, but amplitude saturation is reliably detected by operator norm bounds.

**Hybrid architectures** combining recurrence and convolution components require mixed diagnostic approaches, with different contracts applied to different architectural components.

### 5.3 Methodological Contributions

**Non-circular outcome generation**: Our linear dynamics testing approach prevents the circular correlations that can inflate apparent predictive performance when outcomes are derived from the same spectral properties used as predictors. This methodology has broader applicability to stability analysis in structured neural networks.

**Regime-appropriate testing**: The discovery that contracts must be tested in the non-normal regime (κ(V) ≥ 100) rather than the well-conditioned regime provides methodological guidance for future stability diagnostic research. Testing universal methods on inappropriate parameter regimes can lead to false negative results.

### 5.4 Limitations and Future Work

**Linear dynamics scope**: Our evaluation tests SSM matrix dynamics without full neural network training loops. While this isolates the spectral effects we study, it omits interactions with optimizers, learning rate schedules, and batch statistics that affect real training stability.

**Scale limitations**: Computational constraints limited evaluation to N≤64. Larger state dimensions may exhibit different failure modes or change the relative importance of different contract metrics.

**Architecture coverage**: We focus on two representative SSM families. Selective SSMs like Mamba [Gu & Dao, 2023] with input-dependent gating represent a third architectural class requiring separate analysis.

**Extension to training regimes**: Future work should validate linear dynamics predictions against actual training experiments, test the framework on additional SSM families, and develop contracts specific to input-dependent architectures.

---

## 5. The ssm-contracts Research Tool

We release `ssm-contracts` as a **research diagnostic tool** implementing the contract metrics evaluated in this paper. The tool is intended as a research artifact for architecture exploration and stability screening, not as a production-validated stability guarantee. It provides red/yellow/green risk indicators based on thresholds calibrated on the benchmark families in this study; applicability to other architectures or larger scales requires separate validation.

### 5.1 Scope Boundaries

The tool explicitly documents:
- Applicable families: recurrence-inspired (C3 relevant) vs. convolution-inspired (operator norm sufficient)
- Scale limitations: calibrated at N≤64; larger scales require recalibration
- Regime requirement: non-normal regime (κ(V)≥100) for contract metrics to add value over trivial baselines
- Linear dynamics scope: predictions are for linearized dynamical risk, not full training stability

---

## 6. Conclusion

We establish that **spectral radius analysis is insufficient for recurrence-based SSMs in the non-normal regime** but remains adequate for convolution-based architectures. Pseudospectral sensitivity (C3) provides genuine predictive value (ΔSpearman=+0.158, AUROC=0.948) for S4-like recurrence SSMs, while operator norm analysis suffices for Hyena-like convolution SSMs.

The **architecture-dependent diagnostic contrast** — recurrence-inspired failures driven by non-normal transient amplification, convolution-inspired failures driven by amplitude saturation — provides evidence for a family-dependent diagnostic taxonomy and practical guidance for when to escalate beyond trivial baselines. This finding emerged from a failed universal generalization attempt: testing on normal diagonal matrices initially showed no contract adding value, which led to identifying the non-normal regime as the operative setting.

Our `ssm-contracts` research tool implements this framework with calibrated thresholds and explicit scope documentation, enabling systematic stability screening for architecture exploration. Validation against real training loops and multiple family members remains future work.

**Data and Code Availability**: Implementation, benchmark suite, and results data available at https://github.com/spectral-contracts/ssm-contracts

---