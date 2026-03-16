# Spectral Contracts — AI Agent Starter Bundle
### Computable Spectral Diagnostics for Long-Horizon Stability in SSMs

> **Source:** Built from the uploaded Spectral Contracts bundle. Added formal definitions,
> agent architecture, phase gate criteria, metric schema with cost and failure-mode mapping,
> benchmark design as hard constraints, generalization operationalization, CLI tool spec,
> calibration vs. correlation distinction, full literature base, negative result plan,
> full prompt suite, deliverable schemas, rigor tagging adapted for diagnostics,
> and commercial product spec.

---

## Table of Contents

1. [Mission and Definitions](#1-mission-and-definitions)
2. [Success, Survival, and Kill Criteria](#2-success-survival-and-kill-criteria)
3. [Research Thesis and Design Philosophy](#3-research-thesis-and-design-philosophy)
4. [Agent Architecture](#4-agent-architecture)
5. [Literature Inputs](#5-literature-inputs)
6. [Workstreams and Phase Plan](#6-workstreams-and-phase-plan)
7. [Phase Gate Criteria](#7-phase-gate-criteria)
8. [Negative Result and Pivot Plan](#8-negative-result-and-pivot-plan)
9. [Agent Prompts](#9-agent-prompts)
10. [Deliverables and Schemas](#10-deliverables-and-schemas)
11. [Benchmark Suite Specification](#11-benchmark-suite-specification)
12. [Resources](#12-resources)
13. [Timeline](#13-timeline)
14. [Main Traps](#14-main-traps)
15. [Target Outcome, Venues, and Commercial Path](#15-target-outcome-venues-and-commercial-path)

---

## 1. Mission and Definitions

**Mission.** Create computable spectral diagnostics — "contracts" — that predict whether an SSM stack will train stably and preserve long-context signal, before any large training spend is committed.

This is the fastest wedge into the SSM stability problem because it does not require solving the deepest theorem first. A useful diagnostic framework can be built now; theory can backfill later. A tool with genuine predictive power is publishable and commercially useful even without a closed-form proof behind it.

### 1.1 Initial Scope

**Families to include:**

| Family | Representative architecture | Scope constraint |
|--------|---------------------------|-----------------|
| S4-like | S4, DSS, S5 | **Primary — Phase 1.** All WS1 metric implementations and WS2 benchmark runs must be complete and passing on S4-like models before any other family is added. |
| Hyena-like | Hyena Hierarchy | **Phase 2 — generalization test.** Add only after G3 passes on S4-like. Long-convolution operator tests a structurally different failure mode. |
| Mamba-like | Mamba (selective SSM approximation) | **Phase 3 — optional extension.** Add only after G3.5 (generalization) passes on S4-like + Hyena. The input-dependent gate means pre-training contracts apply to the *average gate behavior* only; this must be stated as a limitation, not papered over. If the gate makes contracts uninformative, this is a finding, not a scope failure. |
| Hybrid baseline | Simple recurrence + attention | One baseline only; added alongside S4-like in Phase 1. Validates contracts are not trivially attention-specific. |

**Mamba sequencing rationale.** Mamba's input-dependent gating means the state transition matrix A changes per token — the "pre-training contract" concept applies only to the initialization distribution of the gate, not to A at inference time. Claiming that a pre-training contract predicts Mamba training stability without this caveat is a scope error. The gate is not "a complication" to work around — it is a qualitatively different regime that requires its own analysis. Test the simpler cases first; add Mamba only when the simpler contracts are validated.

Start with small models only. "Small" means: state dimension N ≤ 256, sequence length T ≤ 4,096, parameter count ≤ 10M. Do not scale up until predictive correlations are established on small models.

### 1.2 Formal Definitions

**Spectral contract.** A spectral contract is a scalar or vector function C(θ) of model parameters θ, computable before or at the start of training (pre-training or early-training), that satisfies:
1. **Predictive**: C(θ) has a statistically significant monotone relationship with at least one downstream stability or long-context performance metric (Spearman ρ ≥ 0.60, p < 0.05).
2. **Cheap**: C(θ) is computable in time O(N² · L) or less (where N is state dimension, L is depth) — at most one eigendecomposition per layer.
3. **Informative beyond the trivial baseline**: C(θ) strictly improves on the trivial baseline (spectral radius alone or operator norm alone) per §2.1.

**Trivial baseline.** The trivial baseline consists of: (a) max eigenvalue magnitude of each A_l individually, (b) operator norm ‖A_l‖, and (c) gradient norm at step 0. Any contract metric that does not beat all three of these on at least one benchmark is not informative.

**Long-horizon stability.** An SSM stack is long-horizon stable in a given experiment if the training loss does not diverge (loss < 10× initial loss at step 500), gradient norms remain bounded (‖∇L‖ < 100× initial norm at step 100), and performance on the long-sequence benchmark task does not degrade below the random baseline.

**Generalization across families.** A contract metric generalizes across families if its rank correlation with stability outcomes is statistically significant (Spearman ρ ≥ 0.55, p < 0.05) for each of at least 2 distinct SSM families tested independently.

**Rigor tagging for contract metrics.** Every contract metric must carry one of:
- `[PROVEN]` — metric has a formal theoretical justification derivable from the literature
- `[MOTIVATED]` — metric is theoretically motivated (plausible argument) but not proven
- `[EMPIRICAL]` — metric works in validation experiments but lacks theoretical story
- `[HEURISTIC]` — metric is intuitive but neither proven nor systematically validated

A contract metric may be accepted into the tool with any tag, but `[HEURISTIC]` metrics that fail empirical validation must be dropped.

---

## 2. Success, Survival, and Kill Criteria

### 2.1 Strong Claim Threshold (AND — required for the target paper)

| Code | Criterion | Measurable threshold |
|------|-----------|----------------------|
| SC-1 | Contract metrics outperform trivial baselines | ΔSpearman ρ ≥ 0.10 over best trivial baseline on ≥ 2 of 3 benchmark tasks; paired Wilcoxon test p < 0.05 |
| SC-2 | Contracts predict instability or long-context failure early | At least one contract metric achieves AUROC ≥ 0.75 for predicting training divergence before step 100 |
| SC-3 | Generalization across at least 2–3 SSM families | Each of SC-1 and SC-2 holds independently within ≥ 2 distinct SSM families |

**On SC-1 operationalized.** "Outperform trivial baselines" is defined as: ΔSpearman ρ = ρ(contract metric, stability outcome) − ρ(best trivial baseline, stability outcome) ≥ 0.10. The 0.10 threshold means the contract adds at least 10 percentage points of rank correlation over what spectral radius alone gives. This is a meaningful improvement — not just statistical significance on a tiny gap.

**On the key experiment design rule.** The benchmark must include cases where the trivial baseline fails:
- Case A: spectral radius says "safe" (all eigenvalues < 1) but training diverges anyway
- Case B: spectral radius says "risky" (eigenvalue near 1) but training is stable

Without these cases, the paper cannot demonstrate that contracts add anything. These cases must be explicitly constructed in the benchmark design (§11); they are not optional.

### 2.2 Survival Threshold (OR — project continues if any one holds at the 8-week check)

| Code | Criterion |
|------|-----------|
| S-A | At least one contract metric achieves SC-1 on one SSM family (not necessarily generalizing) |
| S-B | The predictiveness study reveals which failure modes the trivial baseline misses — even if no single contract beats it, the characterization of when and why it fails is publishable |
| S-C | The benchmark suite itself is a contribution: a reusable, multi-family, multi-task SSM stability benchmark that did not previously exist |

S-C is a real outcome. A well-designed benchmark paper is publishable at NeurIPS benchmarking track or as a TMLR paper and is highly cited.

### 2.3 Kill Criteria

**Kill or pause if, at the 6–8 week checkpoint, all three hold:**

1. **Contract metrics do not beat trivial baselines.** Formal threshold: no contract metric achieves ΔSpearman ρ ≥ 0.05 over any trivial baseline on any benchmark task after full WS3 analysis. Not even a marginally informative signal.

2. **Results are too architecture-specific.** No contract metric achieves Spearman ρ ≥ 0.50 for more than one SSM family. Every metric that works for S4-like recurrences fails for Hyena-like, and vice versa. No structural explanation for why.

3. **Metrics are too expensive.** No informative contract metric can be computed in O(N² · L) time or less on practical model sizes. Every metric that works requires full singular-value decomposition of the composed operator, which is O(N³ · L) and impractical for N > 64.

If killed: write a 3-page memo documenting which contract metrics were tested, why they failed, and what structural properties of SSMs make pre-training stability prediction harder than anticipated. This closes the question for the community.

---

## 3. Research Thesis and Design Philosophy

**Thesis.** The spectral radius of individual SSM transition matrices is a necessary but insufficient predictor of long-horizon training stability. The missing predictors involve interactions across the stack — composed operator condition numbers, singular-value dispersion across depth, and pseudospectral sensitivity — that are cheap to compute before training and systematically informative about failure modes that the spectral radius misses.

**Why faster than the theorem-first approaches.** FreeInit and Idealized Subblock both require a spectral theorem before implementation. Spectral Contracts inverts this: build the diagnostic, validate its predictive power empirically, then backfill with theory. This means a useful tool and a paper are achievable in 90 days even if the theoretical justification remains `[MOTIVATED]` rather than `[PROVEN]`.

**The mathematical thesis requirement.** "Theory can backfill later" does not mean "no theory required." Every FULL CONTRACT metric entering the tool must have at least a `[MOTIVATED]` theoretical justification — a specific argument for *why* the metric predicts stability, grounded in linear systems theory, pseudospectral analysis, or free-probability results. A metric with only an `[EMPIRICAL]` tag and no theoretical story is a data mining result; it is not publishable at the target venues and will be rejected on reviewer scrutiny alone. The mathematical thesis of this paper is: *pre-training spectral properties beyond the spectral radius are informative because they capture non-normality, condition growth, or controllability degradation that eigenvalue magnitude misses*. Every accepted metric must connect to one of these mechanisms explicitly.

### 3.1 Design Philosophy (Agent Instruction Constraints)

| Priority | Constraint | Rationale |
|----------|-----------|-----------|
| 1 | **Every metric must have a computational cost estimate.** A metric without a cost estimate is not a contract — it is a research curiosity. Cost must be reported in O(·) notation before any metric enters the inventory. | Prevents building metrics that look good in experiments but are unusable in practice |
| 2 | **The trivial baseline must be run first, in full.** Do not compute a single contract metric before you have complete trivial-baseline results. This prevents cherry-picking contracts that happen to beat a weak baseline. | Ensures SC-1 comparison is honest |
| 3 | **Generalization test is mandatory, not a bonus.** Every metric validated on one family must be tested on a second family before it enters the final tool. A metric that works only for S4 is an S4-specific heuristic, not a contract. | Prevents over-claiming generality |
| 4 | **Calibration is as important as correlation.** A metric that ranks models correctly but assigns the wrong absolute risk score is not a useful tool. Both Spearman ρ (rank) and calibration curves (absolute) must be reported. | Prevents a tool that says "high risk" when the model is fine |
| 5 | **The tool must have a threshold, not just a score.** The CLI must output red/yellow/green, not just a number. Threshold-setting requires a held-out calibration set. | Ensures the tool is actionable, not just informative |

### 3.2 Five Anti-Patterns (Binding Prohibitions)

These are failure modes that invalidate the contribution regardless of empirical results. Each carries a detection test run at G2.5 and G3.5.

| Anti-pattern | Prohibition | Detection test |
|-------------|-------------|----------------|
| **Jumping to full Mamba theory** | Mamba-like models are Phase 3 (§1.1) — added only after G3.5 passes on S4-like + Hyena-like. The input-dependent gate means "pre-training contract" has a different meaning for Mamba; this must be explicitly scoped, not silently extended. Any WS1 or WS2 deliverable that includes Mamba before G3.5 passes is out of sequence. | G2.5 check: does `metric_inventory.md` or `benchmark_spec.md` include Mamba configurations before G3.5 has passed? If yes, defer to Phase 3. |
| **Claiming trained-network spectral predictions** | Contract metrics are pre-training diagnostics. They predict *training stability* (divergence, gradient health), not *post-training performance* (final accuracy, generalization). Any sentence of the form "our contract predicts the trained model will achieve..." requires a separate post-training validation experiment. The CLI output must say "stability risk" not "performance prediction." | G2.5 check: does any metric description, README, or tool output claim to predict post-training outcomes from pre-training metrics alone? If yes, rewrite to scope the claim to training stability only. |
| **Free probability as branding for ordinary spectral-radius heuristics** | C6 ("free-probability-inspired composed spectral spread") uses a diagonal approximation. If the approximation reduces to computing eigenvalues of A_l individually and taking a product, it is a spectral-radius heuristic, not a free-probability result. The `[HEURISTIC]` tag is correct for C6 as described; do not upgrade it to `[MOTIVATED]` or `[PROVEN]` without invoking a specific free-probability tool (R-transform, free convolution, S-transform) in the justification. The same applies to any other metric whose "free-probability inspiration" consists only of citing Mingo & Speicher without using the machinery. | G2.5 check: for each metric tagged `[MOTIVATED]` or higher with "free probability" in the justification, name the specific tool used. If the answer is "none — I compute eigenvalues," downgrade the tag to `[HEURISTIC]`. |
| **Building only toy theory with no empirical correspondence** | Every `[MOTIVATED]` or `[PROVEN]` metric justification must connect to a measurable failure mode. If the theoretical story explains why the metric *should* predict instability but WS3 shows it does not, the metric is demoted to `[EMPIRICAL]` or dropped. The mathematical thesis (§3 above) must be actively tested — not just stated in the introduction and ignored in the analysis. | G3 check: for each metric with a theoretical justification, does the WS3 regression show the predicted direction of effect? If a `[MOTIVATED]` metric predicts the wrong direction, document this explicitly — it is a finding, not something to quietly drop. |
| **Building only engineering benchmarks with no new mathematical thesis** | The S-C survival path (§2.2) allows a benchmark-only outcome, but it must not be reached by drifting — it must be a deliberate pivot after Kill criteria are checked. If WS3 shows no metric beats the trivial baseline, the correct action is §8 Track A or Track A3 (characterize *why* spectral radius is sufficient), not to simply publish the benchmark as if the project's mathematical thesis were always "let's build a benchmark." The paper must explain, not just measure. | G3 check: if no metric achieves SC-1, does the analysis contain a mechanistic explanation for why? "Spectral radius is sufficient because [structural property of current SSMs]" is a finding. "We ran the benchmark" is not. |

| Priority | Constraint | Rationale |
|----------|-----------|-----------|
| 1 | **Every metric must have a computational cost estimate.** A metric without a cost estimate is not a contract — it is a research curiosity. Cost must be reported in O(·) notation before any metric enters the inventory. | Prevents building metrics that look good in experiments but are unusable in practice |
| 2 | **The trivial baseline must be run first, in full.** Do not compute a single contract metric before you have complete trivial-baseline results. This prevents cherry-picking contracts that happen to beat a weak baseline. | Ensures SC-1 comparison is honest |
| 3 | **Generalization test is mandatory, not a bonus.** Every metric validated on one family must be tested on a second family before it enters the final tool. A metric that works only for S4 is an S4-specific heuristic, not a contract. | Prevents over-claiming generality |
| 4 | **Calibration is as important as correlation.** A metric that ranks models correctly but assigns the wrong absolute risk score is not a useful tool. Both Spearman ρ (rank) and calibration curves (absolute) must be reported. | Prevents a tool that says "high risk" when the model is fine |
| 5 | **The tool must have a threshold, not just a score.** The CLI must output red/yellow/green, not just a number. Threshold-setting requires a held-out calibration set. | Ensures the tool is actionable, not just informative |

---

## 4. Agent Architecture

### 4.1 LLM and Tool Stack

| Component | Specification |
|-----------|--------------|
| Backbone LLM | Claude 3.5 Sonnet or GPT-4o |
| Numerical computation | NumPy / SciPy for eigenvalue and SVD computation; JAX for GPU-accelerated operator composition |
| Statistical analysis | statsmodels or scipy.stats for Spearman ρ, Wilcoxon test, calibration; scikit-learn for AUROC |
| Visualization | matplotlib / seaborn for calibration curves, scatter plots, and risk maps |
| Tool packaging | Click or Typer for CLI; Rich for terminal output formatting |
| Experiment tracking | Weights & Biases or MLflow for sweep results |
| Literature search | Semantic Scholar API and arXiv |
| File I/O | Shared project directory; all outputs versioned in Git |

### 4.2 Orchestration Model

Four workstreams with a metric-audit gate between definition and validation. WS1 (metric inventory) and WS2 (benchmark suite) run in parallel during weeks 1–2. WS3 (predictiveness study) cannot begin until WS1 produces a stable metric inventory and WS2 produces a runnable benchmark. WS4 (tooling) begins only after WS3 identifies which metrics survive validation.

```
WS1 Metric inventory ─── [G1: inventory complete, costs estimated] ──────────────────┐
WS2 Benchmark suite ──── [G2: benchmark runs clean, trivial baseline complete] ────── ┘
                                                                                       │
                          [G2.5: metric audit checkpoint] ──────────────────────────── ┤
                                                                                       │
WS3 Predictiveness ────── [G3: SC-1 and SC-2 tested on ≥ 1 family] ─────────────────── ┤
                          [G3.5: generalization test on ≥ 2 families] ──────────────── ┤
                                                                                       │
WS4 Tooling ─────────────── [G4: CLI passes integration tests, thresholds calibrated] ─┤
                                                                                       │
                          [Kill check week 6–8] ─── stop or continue ─────────────────┘
                                                                                       │
                                                                                  paper + tool
```

### 4.3 Memory and Context Management

Each workstream invocation receives:
1. **Persistent context** — this document and `metrics/metric_inventory.md` (once created). Always included in full.
2. **Workstream-specific context** — deliverables from the directly preceding step only.
3. **No raw conversation accumulation** — decisions live in deliverables.

### 4.4 Tool Permissions by Workstream

| Workstream | Lit search | Code execution | File write | Statistical analysis |
|-----------|-----------|---------------|------------|---------------------|
| WS1 Metric inventory | Yes | Yes (cost estimation) | Yes | No |
| WS2 Benchmark | Yes | Yes | Yes | No |
| WS2.5 Metric audit | No | No | Yes (comments) | No |
| WS3 Predictiveness | No | Yes | Yes | Yes |
| WS4 Tooling | No | Yes | Yes | No |

---

## 5. Literature Inputs

This project draws from a different literature base than FreeInit or Idealized Subblock. The theory papers are background, not the primary source — the key references here are on pseudospectra, controllability, Jacobian analysis, and systems-theoretic stability.

### 5.1 Spectral Diagnostics and Pseudospectra

| Paper | Key contribution | Relevance to contracts |
|-------|-----------------|----------------------|
| Trefethen & Embree — *Spectra and Pseudospectra* (2005, book) | Pseudospectral sensitivity: how much can eigenvalues move under small perturbations | Foundation for the pseudospectral sensitivity proxy; explains why spectral radius alone is insufficient for non-normal matrices |
| Trefethen (1992) — *Pseudospectra of matrices* | ε-pseudospectrum definition and numerical methods | Direct source for pseudospectral contract metric |
| Greenbaum (1997) — *Iterative Methods for Solving Linear Systems* | Normal vs. non-normal operator behavior under iteration | Motivates why non-normality of A predicts different stability than spectral radius suggests |
| Higham & Tisseur (2000) — *A block algorithm for matrix 1-norm estimation* | Cheap norm estimation for large matrices | Methods for O(N²) pseudospectral approximation |

### 5.2 Controllability, Observability, and Systems Theory

| Paper | Key contribution | Relevance |
|-------|-----------------|-----------|
| Kalman (1960) — *A new approach to linear filtering and prediction* | Controllability/observability Gramians | Foundation for finite-horizon controllability proxy; Gramian condition number predicts memory degradation |
| Sontag — *Mathematical Control Theory* (textbook, 1998) | State-space stability; Lyapunov functions | Background for stability region characterization |
| Zhou, Doyle & Glover — *Robust and Optimal Control* (1996) | H∞ and H₂ norms; operator gain bounds | H∞ norm as a contract metric candidate |

### 5.3 Jacobian Anisotropy and Deep Learning Stability

| Paper | Key contribution | Relevance |
|-------|-----------------|-----------|
| Saxe, McClelland & Ganguli (2014) — *Exact solutions to the nonlinear dynamics of learning* | Singular value dynamics during learning | Foundation for Jacobian anisotropy growth metric |
| Pennington, Schoenholz & Ganguli (2017) — *Resurrecting the sigmoid* | Dynamical isometry as a stability criterion | Singular-value spread of the end-to-end Jacobian as a contract candidate |
| Glorot & Bengio (2010) — *Understanding the difficulty of training deep networks* | Variance scaling for stable gradient flow | Baseline comparison: Glorot init as a trivial contract |
| Raghu et al. (2017) — *On the expressive power of deep neural networks* | Trajectory length and expressivity via singular values | Background for anisotropy growth contract |

### 5.4 SSM-Specific Stability References

| Paper | Key contribution |
|-------|-----------------|
| Orvieto et al. (2023) — *LRU* | Empirical stability condition (eigenvalue magnitude); the paper most closely related to this project's goals |
| Gu et al. (2022) — *S4* | HiPPO initialization; structured eigenvalue placement |
| Gupta, Gu & Berant (2022) — *DSS* | Diagonal simplification; most tractable for contract analysis |
| Gu & Dao (2023) — *Mamba* | Input-dependent gating; complicates pre-training contracts |
| Poli et al. (2023) — *Hyena Hierarchy* | Long-convolution operator; different failure modes than recurrence |

### 5.5 Benchmarking and Evaluation Methodology

| Paper | Key contribution |
|-------|-----------------|
| Tay et al. (2021) — *Long Range Arena* | Standard long-sequence benchmark; use as a reference point |
| Zhai et al. (2022) — *Scaling Vision Transformers* | Methodology for controlled stability sweeps |
| Papyan (2020) — *Traces of class/cross-class structure in the full Hessian* | Hessian spectral analysis during training |

---

## 6. Workstreams and Phase Plan

### Workstream 1 — Metric Inventory (Weeks 1–2)

**Deliverable:** `metrics/metric_inventory.md`

**Six candidate contract metrics to evaluate and implement:**

| ID | Metric name | Informal description | Theoretical motivation | Rigor tag |
|----|-------------|---------------------|----------------------|-----------|
| C1 | Effective transition condition growth | Condition number of A_1 · ... · A_L as a function of T | Condition number bounds amplification of perturbations over T steps | `[MOTIVATED]` |
| C2 | Singular-value dispersion of stacked operator | Ratio σ_max / σ_min of the composed operator; also spectral spread (σ_max − σ_min) / σ_mean | Wide spread predicts anisotropic gradient flow and selective memory loss | `[MOTIVATED]` |
| C3 | Pseudospectral sensitivity proxy | ε-pseudospectral radius: max{|z| : z ∈ Λ_ε(A)} for small ε | Non-normal matrices can have misleadingly small spectral radii; pseudospectrum reveals true amplification | `[PROVEN]` for non-normal matrices via Trefethen & Embree |
| C4 | Finite-horizon controllability proxy | Condition number of the finite-horizon controllability Gramian W_T = Σ_{t=0}^{T} A^t B B^T (A^T)^t | Gramian condition number predicts whether all state components receive input signal; ill-conditioned Gramian predicts selective memory failure | `[PROVEN]` in linear systems theory |
| C5 | Jacobian anisotropy growth | Growth rate of σ_max / σ_min of the end-to-end Jacobian ∂h_T / ∂h_0 as a function of T | Anisotropy growth predicts gradient explosion/vanishing asymmetry | `[MOTIVATED]` from Pennington et al. |
| C6 | Free-probability-inspired composed spectral spread | Predicted support edge of the limiting singular-value distribution of A^L under the i.i.d. diagonal approximation | Connects to FreeInit theoretical results; uses the diagonal approximation as a fast proxy | `[HEURISTIC]` — requires diagonal approximation of non-diagonal A |

**For each metric, WS1 must complete the full schema (§10.3). This includes a cost estimate before any implementation begins.**

**Step 1.1 — Cost estimation before implementation.**

For each metric, estimate the computational cost in O(·) notation and wall-clock time for N = 64, L = 8 before writing a single line of implementation code. If the cost estimate exceeds O(N³), the metric is deprioritized unless an approximation can bring it to O(N²) or below.

| Metric | Exact cost | Approximation available? | Approx. cost |
|--------|-----------|--------------------------|-------------|
| C1 | O(N³ · L) for full SVD | Yes — power iteration for max/min singular values | O(N² · L · k) where k = iterations |
| C2 | O(N³ · L) for full SVD | Yes — randomized SVD | O(N² · L) |
| C3 | O(N² · grid) for ε-pseudospectrum on grid | Yes — Kreiss matrix constant as cheap proxy | O(N³) once per model |
| C4 | O(N² · T) for Gramian construction | Yes — low-rank approximation of Gramian | O(N · T · r) where r = rank |
| C5 | O(N³ · T) for full Jacobian SVD at each T | Yes — track only σ_max and σ_min via power iteration | O(N² · T · k) |
| C6 | O(N · L) for diagonal approximation | N/A — already cheap | O(N · L) |

**Step 1.2 — Implement trivial baselines first.**

Before implementing any contract metric, implement and run all three trivial baselines in full:
1. Max eigenvalue magnitude: max_l max_i |λ_i(A_l)|
2. Operator norm: max_l ‖A_l‖₂
3. Gradient norm at step 0: ‖∇L(θ)‖ before any update

Complete trivial baseline results are Gate G2's prerequisite. No contract metric results may be reported without them alongside.

**Step 1.3 — Implement contract metrics.**

Implement C1–C6 in `metrics/contracts.py` (exact) and `metrics/approximations.py` (cheap versions). Each function must match the schema in §10.3 and include a doctest verifying the cost estimate.

---

### Workstream 2 — Benchmark Suite (Weeks 1–3)

**Deliverable:** `benchmarks/benchmark_spec.md`

The benchmark suite must be honest: it must include cases where the trivial baseline fails. This is a hard constraint, not a quality-of-life feature.

**Step 2.1 — Task selection.**

| Task | Type | T | What it tests | SSM families |
|------|------|---|--------------|-------------|
| Copying task | Synthetic long memory | 512, 1024, 4096 | Pure memory retention | All |
| Selective recall | Synthetic | 1024 | Selective memory under distraction | All |
| Long-range parity | Synthetic | 2048, 4096 | Long-range dependency with distractors | All |
| Controlled instability sweep | Synthetic | variable | Stability boundary characterization | All |
| Short LM surrogate | Real (Wikitext-103 at 256 tokens) | 256 | Practical proxy for training stability | S4, Mamba |
| Sequence classification with distractors | Synthetic | 1024, 2048 | Long-context with irrelevant tokens | All |

**Step 2.2 — Construct the "trivial baseline failure" cases.**

These cases must be explicitly built into the benchmark. They are not hoped for — they are engineered.

*Case A construction (spectral radius says safe, model fails):*
- Use a near-orthogonal non-normal A matrix with all eigenvalues inside the unit circle but high pseudospectral radius
- Example: A = V D V⁻¹ where D has eigenvalues at 0.95 but V is ill-conditioned (condition number κ(V) ~ 1000)
- Expected: spectral radius ≈ 0.95 (safe), but Kreiss constant is large, amplification occurs, training destabilizes
- Verification: confirm empirically that training diverges on the copying task with this A

*Case B construction (spectral radius says risky, model trains):*
- Use a diagonal A with one eigenvalue at 0.999 (near unit circle) but otherwise well-conditioned and low pseudospectral radius
- Expected: spectral radius ≈ 0.999 (risky by trivial baseline), but training is stable because there is no non-normal amplification
- Verification: confirm empirically that training converges on the copying task with this A

Both cases must be in the final benchmark. They are the empirical foundation for SC-1.

**Step 2.3 — Controlled instability sweep design.**

A sweep over (eigenvalue radius r, depth L, sequence length T) with stability outcome recorded. This generates the regression dataset for WS3.

| Sweep axis | Values | Fixed while sweeping |
|-----------|--------|---------------------|
| Eigenvalue radius r | 0.7, 0.8, 0.9, 0.95, 0.99, 1.0, 1.01 | L=8, T=1024 |
| Depth L | 2, 4, 8, 12, 16, 24 | r=0.95, T=1024 |
| Sequence length T | 256, 512, 1024, 2048, 4096 | r=0.95, L=8 |

Each configuration: 5 seeds, S4-like architecture, N=64. Repeated for at least one other SSM family for the generalization test.

---

### Workstream 2.5 — Metric Audit Checkpoint (End of Week 3)

**Gate G2.5.** Before WS3 begins, the PI and at least one external colleague review `metric_inventory.md` against the following checklist:

- [ ] Every metric has a computational cost estimate in O(·) notation
- [ ] Every metric has a rigor tag and the tag is accurate
- [ ] Trivial baselines are fully implemented and their results are available
- [ ] Case A and Case B (trivial baseline failure cases) have been constructed and empirically verified
- [ ] The benchmark suite runs end-to-end without errors on at least one small model

This checkpoint catches cost-estimate errors (e.g., a metric that looked O(N²) but is actually O(N³) in practice) before the full predictiveness study is run.

---

### Workstream 3 — Predictiveness Study (Weeks 4–9)

**Deliverable:** `analysis/predictiveness_report.md`

**Step 3.1 — Compute all contracts pre-training for all benchmark configurations.**

For every (model family, architecture config, benchmark task) combination:
1. Compute all trivial baselines
2. Compute all C1–C6 contract metrics (exact and approximate versions)
3. Record wall-clock time for each
4. Log everything to `results/contract_values.csv` per the schema in §10.5

**Step 3.2 — Measure early-training health.**

For each configuration, record:
- Training loss at steps 1, 10, 50, 100, 500
- Gradient norm at steps 1, 10, 50, 100
- Whether training diverged (binary label for AUROC analysis)

**Step 3.3 — Measure final performance.**

For each configuration:
- Benchmark task accuracy / loss after 2,000 training steps
- Maximum stable context length: the largest T at which the model achieves > chance accuracy on the copying task

**Step 3.4 — Regression and correlation analysis.**

For each (contract metric, outcome metric) pair:
1. Compute Spearman ρ and p-value
2. Compute ΔSpearman ρ over best trivial baseline
3. For binary outcomes (divergence yes/no): compute AUROC
4. Produce a calibration curve: plot predicted risk (from contract metric) against observed failure rate across deciles

Report in `analysis/predictiveness_report.md` per schema §10.6.

**Step 3.5 — Generalization test (Gate G3.5).**

Repeat Steps 3.1–3.4 on a second SSM family independently. A metric that achieves SC-1 on S4-like models must also achieve Spearman ρ ≥ 0.55 on at least one other family to qualify as a generalizing contract.

---

### Workstream 4 — Tooling (Weeks 10–13)

**Deliverable:** `tool/` directory containing a working CLI

**Step 4.1 — Identify which metrics enter the tool.**

Only metrics that pass the generalization test (G3.5) and have cost ≤ O(N² · L) enter the tool. Metrics that are informative but expensive go into `approximations.py` with a warning label.

**Step 4.2 — Implement the CLI.**

```bash
# Basic usage
ssm-contracts check --config model_config.yaml

# Output
$ ssm-contracts check --config s4_experiment.yaml

  SSM Spectral Contracts v1.0
  ─────────────────────────────────────────────
  Model: S4-like, N=128, L=8
  Sequence length: 2048

  Contract metrics:
    C1  Transition condition growth:   847.3   [YELLOW — elevated risk]
    C2  Singular-value dispersion:     12.4    [GREEN — within safe range]
    C3  Pseudospectral sensitivity:    0.18    [RED — high non-normal amplification]
    C4  Controllability proxy:         2891.0  [RED — ill-conditioned Gramian]
    C5  Jacobian anisotropy growth:    3.2x/layer [YELLOW]

  Overall risk: RED
  Predicted failure mode: Non-normal amplification under long sequences.
  Recommendation: Apply LRU-style eigenvalue constraint or increase diagonal regularization.
  ─────────────────────────────────────────────
```

**Step 4.3 — Calibrate thresholds.**

Red/yellow/green thresholds are not set by intuition — they are calibrated on a held-out set of benchmark configurations. Procedure:

1. Hold out 20% of benchmark configurations as the calibration set (not used in WS3 regression)
2. For each metric, fit a threshold that maximizes F1 on the binary outcome (divergence yes/no)
3. Report calibration curves alongside thresholds — the tool must not claim accuracy the calibration does not support
4. Document the threshold, the calibration set size, and the calibration performance in `tool/thresholds.yaml`

**Step 4.4 — Integration tests.**

The CLI must pass integration tests on:
- An S4-like model that is known to be stable (all-green expected)
- An S4-like model known to be unstable (at least one red expected)
- A Mamba-like model (tests generalization claim)
- A model where trivial baseline says safe but contract says risky (Case A from §11)

---

## 7. Phase Gate Criteria

| Gate | After | Pass criterion | Fail action |
|------|-------|----------------|-------------|
| G1 | WS1 | `metric_inventory.md` complete with cost estimate and rigor tag for all 6 metrics; trivial baselines implemented | Revise missing entries |
| G2 | WS2 | Benchmark runs end-to-end; trivial baseline results complete; Case A and Case B verified empirically | Fix benchmark or revise Case A/B construction |
| G2.5 | Metric audit | PI checklist signed; cost estimates verified in practice; no metric's actual cost exceeds O(N³) without an approved approximation | Remove or approximate expensive metrics |
| G3 | WS3 single family | At least one contract metric achieves SC-1 on one SSM family; AUROC ≥ 0.75 for at least one metric on divergence prediction | Invoke §8 Track A if no metric passes |
| G3.5 | WS3 generalization | Passing metric achieves Spearman ρ ≥ 0.55 on ≥ 2 SSM families independently | Invoke §8 Track B if generalization fails |
| G4 | WS4 | CLI passes all 4 integration tests; thresholds calibrated on held-out set; calibration curves in report | Fix CLI bugs; recalibrate |
| G-kill | Week 6–8 | All three kill criteria (§2.3) hold | Write kill memo; pause project |

---

## 8. Negative Result and Pivot Plan

### Track A: No contract metric beats trivial baseline (fails G3)

Most likely cause: SSM stability is genuinely captured by spectral radius at the scales tested, and the richer metrics are only informative at larger scales or longer sequences.

| Option | Description | Trigger condition |
|--------|-------------|------------------|
| A1 | Scale up T: test at T ≥ 8,192 where non-normal effects accumulate | Metrics show trend toward significance with T but don't reach threshold at T ≤ 4,096 |
| A2 | Narrow to non-normal subcase: demonstrate that contracts work when A is specifically non-normal, and document that S4/DSS are in practice near-normal | At least one constructed non-normal case (Case A) shows strong signal even if natural SSMs don't |
| A3 | Publish the benchmark: the benchmark suite is the contribution; the null result (contracts don't outperform trivial baselines) is the finding | No metric reaches threshold at any T or scale |
| A4 | Reframe as a theory paper: characterize mathematically why spectral radius is sufficient for current SSM architectures | Only if a clear theoretical explanation exists |

### Track B: Contract metrics work but don't generalize (fails G3.5)

Most likely cause: SSM families have structurally different failure modes — non-normality matters for S4 but not for Hyena, which fails via different mechanism.

| Option | Description | Trigger condition |
|--------|-------------|------------------|
| B1 | Architecture-specific contracts: publish a paper with different contracts for different families, with a taxonomy of failure modes | Each family has at least one working metric, just different ones |
| B2 | Focus on S4-like recurrences only, explicitly scoped | S4-like contracts generalize within the family; Hyena/Mamba don't fit |
| B3 | The non-generalization is the finding: characterize why failure modes differ across SSM families | Clear structural explanation for the difference |

### Track C: Metrics work but are too expensive

Most likely cause: the informative metrics (pseudospectral sensitivity, controllability Gramian) require O(N³) computation that is impractical at N > 64.

| Option | Description | Trigger |
|--------|-------------|---------|
| C1 | Develop cheap approximations: randomized SVD, power iteration, low-rank Gramian | Approximation error < 20% relative to exact metric for Spearman ρ |
| C2 | Scope tool to small models only with explicit size limit | Contracts are still useful for architecture search at small N before scaling |
| C3 | Provide both: an exact version for offline analysis and a cheap proxy for CI use | Different use cases; both have value |

### Track D: Benchmark is the contribution (S-C survival)

If no contract beats trivial baselines but the benchmark suite is solid:
- Reframe the paper as "A Benchmark for Evaluating Pre-Training Stability Diagnostics in SSMs"
- Include the null result as a finding (current architectures are well-behaved enough that spectral radius is sufficient)
- Target NeurIPS Datasets & Benchmarks track or TMLR

---

## 9. Agent Prompts

### P0 — Agent Prompt Seed

```
Design a benchmark and metric suite for predicting long-horizon training stability in SSMs 
before full training. Metrics must go beyond spectral radius and be cheap enough for 
practical use. Compare predictive value across multiple SSM families and identify which 
diagnostics are robust across architectures.

For each candidate metric:
(a) State the formula or algorithm precisely.
(b) Estimate the computational cost in O(·) notation for state dimension N and depth L.
(c) State the theoretical motivation: why should this metric predict stability?
(d) Identify the failure mode it is designed to catch that spectral radius misses.
(e) State a rigor tag: PROVEN / MOTIVATED / EMPIRICAL / HEURISTIC.

Prioritize metrics that are: cheap (O(N²·L) or less), theoretically motivated, 
and target failure modes where the spectral radius gives incorrect predictions.
Output as a ranked table with costs, not prose.
```

---

### P1 — Metric Inventory Population Prompt

**Workstream:** WS1 | **Output:** entries in `metrics/metric_inventory.md`

```
You are building a catalog of pre-training spectral diagnostics for SSM stability.

You will evaluate the six candidate contract metrics (C1–C6) and any additional candidates 
you identify from the literature.

For each metric, produce a complete entry using the schema below.

After all entries, produce:
## Metric ranking
Sort metrics by: (1) theoretical rigor of motivation, (2) computational cost (cheapest first), 
(3) expected coverage of failure modes. Mark which metrics are likely to be redundant 
(measuring the same underlying quantity via different proxies).

## Trivial baseline definitions
Write out the exact formulas for all three trivial baselines with implementation notes.

## Implementation order recommendation
Which metrics should be implemented first (in order of highest expected 
information-per-compute-dollar)?

Schema for each metric entry: see §10.3.
```

---

### P2 — Benchmark Design Prompt

**Workstream:** WS2 | **Output:** `benchmarks/benchmark_spec.md`

```
You are designing a benchmark suite for evaluating pre-training stability diagnostics in SSMs.

The benchmark has one non-negotiable constraint:
It must include at least one configuration where the spectral radius predicts "safe" 
but the model fails, and at least one where it predicts "risky" but the model succeeds.
These cases are constructed, not hoped for. You must specify exactly how to construct them.

For each benchmark task, specify:
1. Task name and type (synthetic / real)
2. Input format and output format
3. Sequence lengths to test (list all T values)
4. SSM families to test on
5. What stability outcome is measured (divergence / accuracy / max stable T)
6. Expected difficulty: which metrics should predict failures on this task?

Then specify:
## Controlled instability sweep
The independent variables, the range of each, the fixed variables, 
the number of seeds, the outcome variable, and how the sweep is run.

## Case A: spectral radius says safe, model fails
Exact construction: what form does A take? What is the spectral radius? 
What property causes failure that spectral radius misses?
How to verify empirically that training diverges on this A?

## Case B: spectral radius says risky, model succeeds
Exact construction: what form does A take? What is the spectral radius?
Why does training succeed despite the spectral radius warning?
How to verify empirically that training converges?

## Calibration / holdout split
Which configurations are held out for CLI threshold calibration (not used in WS3 regression)?
How are they sampled to avoid leakage?
```

---

### P3 — Predictiveness Analysis Prompt

**Workstream:** WS3 | **Output:** sections of `analysis/predictiveness_report.md`

```
You are analyzing the predictive value of a set of pre-training contract metrics 
for SSM stability outcomes.

You will receive: the contract metric values (pre-training), the trivial baseline values, 
and the outcome metrics (training divergence, max stable T, final task accuracy).

For each (contract metric, outcome) pair:
1. Compute Spearman ρ and p-value.
2. Compute ΔSpearman ρ = ρ(contract) − ρ(best trivial baseline).
3. For binary outcomes: compute AUROC, precision, recall at the threshold that maximizes F1.
4. Plot and describe the calibration curve: does the metric's score correspond to 
   observed failure rates? (A metric that ranks models correctly but assigns wrong absolute 
   scores is not well-calibrated and should not drive a red/yellow/green tool.)

Produce a summary table:

| Metric | Outcome | Spearman ρ | ΔSpearman | AUROC | Calibrated? | Rigor tag |
|--------|---------|-----------|-----------|-------|-------------|-----------|

Flag any metric where:
- Spearman ρ is high (≥ 0.6) but calibration is poor: label RANK-ONLY
- Spearman ρ is low but AUROC is high: label THRESHOLD-ONLY  
- Both are high: label FULL CONTRACT (eligible for the tool)
- Both are low: label UNINFORMATIVE

For each SSM family, report results independently. Note whether the same metrics 
rank highest across families (cross-family robustness) or whether different metrics 
work for different families (architecture-specific signal).
```

---

### P4 — Generalization Test Prompt

**Workstream:** WS3 | **Output:** generalization section of `analysis/predictiveness_report.md`

```
You are testing whether spectral contract metrics generalize across SSM families.

You will receive predictiveness results from two or more SSM families tested independently.

For each metric that achieved FULL CONTRACT status on the primary family:
1. Report its Spearman ρ on each other family.
2. Classify: GENERALIZES (ρ ≥ 0.55 on ≥ 2 families) / FAMILY-SPECIFIC (ρ ≥ 0.55 on one family only) / UNINFORMATIVE (ρ < 0.55 on all families).
3. If FAMILY-SPECIFIC: characterize the structural difference between the families 
   that explains why the metric works for one but not the other.

Produce:
## Generalization summary table
| Metric | S4-like ρ | Mamba-like ρ | Hyena-like ρ | Status |

## Failure mode taxonomy
Group the SSM families by their primary failure mode 
(e.g., non-normal amplification, diagonal explosion, convolution bandwidth saturation).
Which contract metrics predict which failure mode?
Does any single metric predict all failure modes? If not, can a composite metric be designed?

## Composite contract recommendation
If no single metric generalizes, recommend a minimum set of metrics that together 
achieve GENERALIZES status — and estimate the combined computational cost.
```

---

### P5 — CLI Tool Specification Prompt

**Workstream:** WS4 | **Output:** `tool/cli_spec.md`

```
You are specifying a command-line tool for pre-training SSM stability contracts.

The tool takes a model config (YAML) and outputs a red/yellow/green risk report.

Specify:

## Input format
What does model_config.yaml contain? 
(model family, N, L, initialization parameters, sequence length, etc.)
What SSM families does it support? What if the family is not recognized?

## Contract computation
Which metrics are computed by default? 
Which are optional (with --expensive flag)?
How are approximations vs. exact versions selected?

## Threshold specification
For each metric that enters the tool (only FULL CONTRACT and GENERALIZES metrics):
- The numerical threshold for RED, YELLOW, GREEN
- How the threshold was calibrated (reference to calibration set)
- The F1 score on the calibration set at this threshold

## Output format
Describe the terminal output structure (see §6 WS4 Step 4.2 example).
What is written to a file vs. printed to terminal?
What figures are generated (plots/ directory)?

## Edge cases
What happens if: N > 256? L > 24? Family is Mamba (has input-dependent gate)?
What is the graceful degradation path for unsupported configurations?

## Integration test specifications
Write the four integration test cases (§6 WS4 Step 4.4) as pytest test stubs 
with assertions that would pass if the tool is working correctly.
```

---

### P6 — Paper Outline and Benchmark Tables Prompt

**After Gate G4 | Output:** `paper/outline.md`, `paper/benchmark_tables.md`

```
You are writing a benchmarking and systems paper on pre-training SSM stability diagnostics.

This is an empirical paper with theoretical motivation — not a theory paper.
The main contributions are: the benchmark suite, the predictiveness study, 
and the CLI tool. Theory backfills the empirical findings.

Write a detailed outline:

1. Title — three options; prefer titles that name the tool and the problem
2. Abstract — 200 words: the problem (costly training failures), the approach (pre-training contracts), the main finding (which metrics work, where trivial baselines fail), the artifact (the tool)
3. Introduction — establish: (a) cost of SSM training failures, (b) insufficiency of spectral radius alone (Case A and Case B), (c) the contract framework, (d) summary of findings
4. Related Work — stability diagnostics in deep learning; pseudospectra; SSM initialization papers; LRA benchmark
5. Contract Metrics — for each metric: formula, cost, motivation, rigor tag
6. Benchmark Suite — task descriptions; Case A and Case B; sweep design; held-out calibration split
7. Predictiveness Study — main regression table; AUROC results; calibration curves; generalization across families
8. The ssm-contracts Tool — CLI demo; threshold calibration; limitations
9. Discussion — which failure modes do current SSMs actually hit? Are contracts necessary for well-initialized models (LRU, HiPPO)?
10. Conclusion — what contracts work, for whom, under what conditions

For each section: estimated length, which deliverable it draws from, anticipated reviewer objection.

Also produce benchmark_tables.md:
The main results table in camera-ready format — all metrics × all families × all outcomes, 
with trivial baselines as the first rows. Mark statistically significant improvements 
with standard notation.
```

---

## 10. Deliverables and Schemas

### 10.1 Full Deliverable List

| File | Workstream | Description |
|------|-----------|-------------|
| `metrics/metric_inventory.md` | WS1 | Complete catalog of all candidate metrics with cost and rigor tags |
| `metrics/contracts.py` | WS1 | Exact implementations of all contract metrics |
| `metrics/approximations.py` | WS1 | Cheap approximation implementations |
| `benchmarks/benchmark_spec.md` | WS2 | Full benchmark task specs including Case A and Case B construction |
| `benchmarks/long_memory_tasks.py` | WS2 | Synthetic task implementations |
| `benchmarks/run_sweeps.py` | WS2 | Controlled instability sweep runner |
| `analysis/predictiveness_report.md` | WS3 | Full regression and correlation analysis |
| `analysis/regression_analysis.py` | WS3 | Spearman ρ, AUROC, calibration computation |
| `analysis/calibration_curves.py` | WS3 | Calibration curve generation |
| `results/contract_values.csv` | WS3 | Raw contract metric values per configuration |
| `results/outcomes.csv` | WS3 | Raw stability outcomes per configuration |
| `tool/cli.py` | WS4 | CLI implementation |
| `tool/thresholds.yaml` | WS4 | Calibrated red/yellow/green thresholds with documentation |
| `tool/report_templates/` | WS4 | Report output templates |
| `paper/outline.md` | Post-G4 | Paper outline from Prompt P6 |
| `paper/benchmark_tables.md` | Post-G4 | Camera-ready results tables |

### 10.2 Folder Scaffold

```
spectral_contracts/
  README.md                          ← links to this bundle document
  metrics/
    metric_inventory.md
    contracts.py
    approximations.py
  benchmarks/
    benchmark_spec.md
    long_memory_tasks.py
    run_sweeps.py
    configs/
      sweep_eigenvalue_radius.yaml
      sweep_depth.yaml
      sweep_sequence_length.yaml
  analysis/
    predictiveness_report.md
    regression_analysis.py
    calibration_curves.py
  results/
    contract_values.csv
    outcomes.csv
    figures/
    tables/
  tool/
    cli.py
    thresholds.yaml
    report_templates/
      terminal_report.txt
      json_report.json
  paper/
    outline.md
    benchmark_tables.md
```

### 10.3 metric_inventory.md Required Schema

Each entry must contain:

```markdown
## Metric [ID]: [name]

**Formula / algorithm:** [exact expression or pseudocode]
**Rigor tag:** [PROVEN / MOTIVATED / EMPIRICAL / HEURISTIC]
**Theoretical motivation:** [one paragraph — why does this predict stability?]
**Failure mode targeted:** [what does spectral radius miss that this catches?]
**Trivial baseline gap:** [in what regime does spectral radius give the wrong prediction?]

**Computational cost:**
- Exact: [O(·) in N and L]
- Approximation available: [YES / NO]
- Approximate cost: [O(·)] — [description of approximation method]
- Practical wall-clock (N=64, L=8): [measured, not estimated]

**Implementation:**
- Exact: `metrics/contracts.py :: [function_name]`
- Approximate: `metrics/approximations.py :: [function_name]` (if applicable)

**Expected Spearman ρ with stability outcomes:** [prior estimate, to be updated in WS3]
**Cross-family robustness expectation:** [why should / shouldn't this generalize?]
**Status after WS3:** [FULL CONTRACT / RANK-ONLY / THRESHOLD-ONLY / UNINFORMATIVE — filled in WS3]
```

Final section: `## Summary table` — all metrics sorted by expected cost, with rigor tags and failure modes.

### 10.4 benchmark_spec.md Required Schema

```markdown
## Task [ID]: [name]

**Type:** SYNTHETIC / REAL
**Description:** [one paragraph]
**Input:** [format, dimensions]
**Output:** [format]
**Sequence lengths T:** [list]
**SSM families tested:** [list]
**Stability outcome measured:** [divergence / accuracy / max stable T]
**Expected failure mode:** [which metrics should predict failures here?]
**Case A / Case B:** [CONTAINS CASE A / CONTAINS CASE B / NEITHER — with construction details if applicable]

---

## Case A: [full construction spec]
## Case B: [full construction spec]

---

## Sweep specification
[Table of sweep axes, values, fixed parameters, seeds, outcome variable]

## Calibration holdout
[Which configurations are held out; sampling procedure]
```

### 10.5 contract_values.csv Schema

```
config_id,          # e.g. CFG001
ssm_family,         # s4_like / mamba_like / hyena_like / hybrid
N,                  # state dimension
L,                  # depth
T,                  # sequence length
init_method,        # default / hippo / lru / custom
seed,               # int (1–5)
metric_id,          # C1 / C2 / C3 / C4 / C5 / C6 / trivial_radius / trivial_norm / trivial_grad
metric_value,       # float
compute_time_ms,    # wall-clock time to compute this metric
exact_or_approx,    # exact / approx
status              # computed / failed / skipped
```

### 10.6 predictiveness_report.md Required Schema

```markdown
## Analysis: [Metric ID] vs. [Outcome]

**Spearman ρ:** [value] (p = [value])
**ΔSpearman ρ over best trivial baseline:** [value]
**AUROC (if binary outcome):** [value]
**Calibration status:** WELL-CALIBRATED / RANK-ONLY / THRESHOLD-ONLY / UNINFORMATIVE
**SSM family:** [which family this analysis covers]

[Calibration curve figure reference]

---

## Summary table: all metrics × all outcomes × all families

[Full table — see paper/benchmark_tables.md for camera-ready version]

---

## Key finding: where does the trivial baseline fail?

[Description of Case A and Case B empirical results — 
which contract metrics correctly predicted the trivial baseline's error in each case?]
```

### 10.7 thresholds.yaml Required Schema

```yaml
metric_id: C3
name: "Pseudospectral sensitivity proxy"
threshold_red: 0.15
threshold_yellow: 0.08
calibration_set_size: 47
calibration_f1_at_red: 0.81
calibration_auc: 0.84
ssm_families_calibrated_on: [s4_like, hyena_like]
note: "Threshold not validated for Mamba-like models with input-dependent gating"
```

---

## 11. Benchmark Suite Specification

### 11.1 Task Descriptions

See §6 WS2 for full task specifications. Summary:

| Task | T range | Primary stability outcome | Case A/B | Required |
|------|---------|--------------------------|----------|----------|
| Copying task | 512–4096 | Memory retention accuracy | Case A and B constructed here | Yes |
| Selective recall | 1024 | Selective memory under distraction | Neither | Yes |
| Long-range parity | 2048–4096 | Long-range dependency accuracy | Neither | Yes |
| Controlled sweep | 256–4096 | Training divergence (binary) | N/A (sweep) | Yes |
| Short LM surrogate | 256 | Training stability (loss) | Neither | Recommended |
| Classification with distractors | 1024–2048 | Accuracy | Neither | Recommended |

### 11.2 Model Configurations

| Parameter | Values | Notes |
|-----------|--------|-------|
| N (state dim) | 32, 64, 128, 256 | N ≤ 256; exact metrics feasible |
| L (depth) | 2, 4, 8, 12, 16, 24 | |
| T (sequence length) | 256, 512, 1024, 2048, 4096 | |
| SSM family | s4_like, mamba_like, hyena_like, hybrid | |
| Init method | default, hippo, lru, case_A, case_B | case_A and case_B are engineered |
| Seeds | 5 per configuration | |

### 11.3 Statistical Reporting Requirements

- Spearman ρ and p-value for every (metric, outcome) pair
- ΔSpearman ρ over best trivial baseline — this is the primary comparison
- AUROC for binary outcomes (divergence yes/no) — threshold for tool calibration
- Calibration curves: predicted risk score vs. observed failure rate, across 10 deciles
- Wilcoxon signed-rank test (p < 0.05) for SC-1 claim (contract > trivial baseline)
- Report all results, including null — do not selectively report only significant metrics

### 11.4 Held-Out Calibration Set

20% of configurations are held out before any WS3 analysis begins. Sampling: stratified by SSM family and T. These configurations are used exclusively for CLI threshold calibration (§6 WS4 Step 4.3). They must never appear in the regression analysis.

---

## 12. Resources

| Resource | Quantity | Notes |
|----------|----------|-------|
| PI | 1.0 FTE equivalent | Required; WS3 analysis needs careful statistical interpretation |
| Engineer / RA | 0.5 FTE | Most valuable in WS1 (metric implementation) and WS4 (CLI packaging) |
| GPUs | 2–4 × A100 equivalent | Controlled sweeps at N=256 and T=4096; most WS1/WS2 work fits on CPU |
| Statistical analysis | scipy, statsmodels, scikit-learn | Standard Python stack |
| Experiment tracking | Weights & Biases (free tier) or MLflow | Required — sweep results must be logged automatically |
| CLI tooling | Click or Typer + Rich | Python packaging |

---

## 13. Timeline

| Week(s) | Active workstreams | Key milestone | Gate |
|---------|-------------------|--------------|------|
| 1–2 | WS1, WS2 (parallel) | Metric inventory complete with costs; trivial baselines running; benchmark tasks coded | G1 |
| 3 | WS2 finalize, WS2.5 audit | Benchmark runs end-to-end; Case A and Case B verified; metric audit checkpoint | G2, G2.5 |
| 4–5 | WS3 single-family | Contract values computed for S4-like family; regression analysis complete | G3 |
| 6–7 | WS3 generalization | Second family results; generalization test | G3.5 |
| **Week 6–8** | **Kill check** | **Invoke §2.3 if all kill criteria hold** | **G-kill** |
| 8–9 | WS3 full analysis | All families; calibration curves; predictiveness report complete | — |
| 10–11 | WS4 CLI | Threshold calibration; CLI implementation; integration tests | G4 |
| 12–13 | WS4 finalize + synthesis | CLI polished; paper outline; benchmark tables | — |
| 14 | Paper draft | First full draft | — |

**Critical path.** WS1 cost estimation (Step 1.1) is the highest-leverage early task. Getting the cost estimates wrong delays the entire project. Week 3 metric audit (G2.5) is designed to catch cost-estimate errors before the full sweep is committed.

**Checkpoint cadence.** Weekly PI check-in. End of Week 3: metric audit meeting. End of Week 5: memo "which metrics show signal on S4-like family?" End of Week 7: generalization decision meeting.

---

## 14. Main Traps

| Trap | Why it happens | Mitigation |
|------|---------------|-----------|
| Cherry-picking contracts that beat a weak baseline | Trivial baselines are implemented lazily (only one, not all three) | §3.1 constraint P2: all three trivial baselines run first and in full before any contract is evaluated |
| Missing the "trivial baseline failure" cases | They require engineering; it's easier to just run natural configurations | Case A and Case B are explicit benchmark requirements per G2; benchmark does not pass gate without them |
| Rank correlation without calibration | Spearman ρ sounds good but miscalibrated tool gives wrong red/yellow/green calls | §3.1 constraint P4: calibration is as important as correlation; calibration curves are required in WS3 |
| Thresholds set by intuition | It's tempting to pick "round number" thresholds | §6 WS4 Step 4.3: thresholds calibrated on held-out set with documented F1; never intuitive |
| Expensive metrics dominate the analysis | O(N³) metrics show strong signal; O(N²) metrics don't | §3.1 constraint P1: cost estimate before implementation; cost is a first-class gating criterion |
| Overfitting to S4-like family | It's the simplest to work with | G3.5 generalization gate requires ≥ 2 families before any metric enters the tool |
| Claiming generalization from weak signal | Spearman ρ = 0.50 on second family is reported as "generalizes" | §1.2 formal threshold: ρ ≥ 0.55 on ≥ 2 families independently, with p < 0.05 |
| Tool without documented limitations | CLI is shipped without noting where calibration fails | `thresholds.yaml` requires a `note` field; Mamba-like models explicitly flagged |
| **[ANTI-PATTERN 1] Jumping to full Mamba theory** | §1.1 sequences Mamba as Phase 3 only, after G3.5 on S4-like + Hyena. The input-dependent gate is not a complication to work around — it is a scope boundary. Any WS1/WS2 deliverable including Mamba configurations before G3.5 is out of sequence and flagged at G2.5. |
| **[ANTI-PATTERN 2] Pre-training contracts predicting trained-network outcomes** | The CLI says "stability risk," not "performance prediction." Claims that pre-training spectral quantities predict post-training accuracy, generalization, or final loss are prohibited unless a post-training experiment directly supports them. G2.5 audits all metric descriptions, README, and tool output for this scope error. |
| **[ANTI-PATTERN 3] "Free probability" labeling spectral-radius heuristics** | C6 is correctly tagged `[HEURISTIC]`. Any metric described as "free-probability-inspired" where the actual computation is eigenvalue magnitudes without invoking R-transform, free convolution, or S-transform is a labeling error. §3.2 requires the specific tool to be named; G2.5 audits every `[MOTIVATED]`-or-higher metric with "free probability" in its justification. |
| **[ANTI-PATTERN 4] `[MOTIVATED]` metrics untested for predicted mechanism direction** | Every metric with a theoretical story must have WS3 check that the predicted direction holds. If theory says "high condition number predicts instability" but correlation is near zero, this is a finding — report it explicitly rather than replacing the metric with one that happens to correlate. |
| **[ANTI-PATTERN 5] Drifting into a pure benchmark paper without declaring it** | S-C survival (benchmark-only) requires an explicit PI decision after Kill criteria fire — not a gradual reframing. If no contract beats the trivial baseline, §8 Track A3 requires a structural explanation for why spectral radius is sufficient. Publishing benchmark results without an explanatory thesis is a scope failure that must be named and pivoted cleanly. |

---

## 15. Target Outcome, Venues, and Commercial Path

### 15.1 Paper Shapes

**Best outcome (all of SC-1, SC-2, SC-3):**

> *"Spectral Contracts for Long-Horizon Stability in Structured State-Space Models"*

with: at least two contract metrics that outperform trivial baselines (ΔSpearman ρ ≥ 0.10), AUROC ≥ 0.75 for divergence prediction, and generalization across S4-like and at least one other family. Includes the working CLI tool as an artifact.

**Good outcome (SC-1 + SC-2, generalization partial):**

Same paper, with §11 narrowed to S4-like and one near-relative (e.g., DSS). Explicit statement that Hyena/Mamba require different contracts.

**Acceptable outcome (S-B or S-C: benchmark or null result):**

"On the Limits of Spectral Radius as a Stability Predictor for SSMs: A Benchmark Study" — the null finding that spectral radius is sufficient for current well-initialized SSMs, plus the benchmark suite as a community artifact.

### 15.2 Venue Targets

| Venue | When to target | Rationale |
|-------|---------------|-----------|
| ICLR | Best outcome; SC-1 + SC-2 + SC-3 | High-visibility; needs strong generalization and working tool |
| NeurIPS Benchmarking & Datasets | S-C outcome (benchmark is the contribution) | Appropriate for rigorous benchmark contribution |
| TMLR | Any outcome with careful, reusable study | No deadline pressure; appropriate for thorough empirical paper |
| ICML Systems / MLSys | SC-1 + SC-3 + working CLI | Systems angle; architecture screening before training |

### 15.3 Commercial and Product Path

This project has the clearest commercial story of the three directions. The path from paper to product:

**Architecture screening before training.** An engineer runs `ssm-contracts check --config experiment.yaml` before committing a GPU cluster to a multi-day training run. Red output → fix the architecture. This pays for itself after one avoided failed run.

**CI-style checks for model changes.** Add `ssm-contracts check` to the CI pipeline alongside unit tests. Any model config change that shifts risk from GREEN to YELLOW or RED fails CI and requires review. This is analogous to test coverage thresholds.

**Automated red/yellow/green stability report.** Integrated into model cards and experiment logs. Every trained model has an archived pre-training risk assessment.

**What the commercial path requires:**
- The CLI tool must be installable with `pip install ssm-contracts` (WS4 must produce a proper Python package)
- Documentation must cover at minimum: installation, the three trivial baselines, C3 and C4 (the most likely winners), and known limitations
- Threshold documentation must be honest about what SSM families the calibration covers
- A public benchmark so users can verify the tool's calibration claims on their own hardware

**Earliest commercial-viable milestone:** End of WS4, if at least two FULL CONTRACT metrics survive generalization testing and the CLI passes integration tests. The tool can be released as v0.1 before the paper is submitted.

---

*Document version 1.0 — Built from uploaded Spectral Contracts bundle. Added formal definitions with measurable thresholds for all three success criteria (§1.2, §2.1), operationalized kill criteria (§2.3), design philosophy as binding agent constraints (§3.1), full literature base spanning pseudospectra/controllability/Jacobian literature (§5), metric cost estimation as a pre-implementation step (§6 WS1), engineered Case A and Case B as hard benchmark requirements (§6 WS2), metric audit checkpoint (§6 WS2.5), calibration vs. correlation distinction (§3.1, §6 WS3), CLI threshold calibration on held-out set (§6 WS4), phase gate criteria with numeric thresholds (§7), four-track negative result plan (§8), six agent prompts (§9), deliverable schemas for all files including CSV schemas (§10), held-out calibration split (§11.4), commercial product path (§15.3).*
