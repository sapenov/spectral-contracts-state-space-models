## C6 Theoretical Motivation [MOTIVATED]

C6 computes the variance of the log-eigenvalue distribution of the composed operator under a diagonal approximation:

    spread(A^L) = Var({log|λ_i(A_1 · ... · A_L)|})
               ≈ Var({Σ_l log|λ_i(A_l)|})   [diagonal approximation]
               = L · Var({log|λ_i(A_l)|})    [i.i.d. layers, CLT]

Under the i.i.d. diagonal assumption, the log-eigenvalue distribution of the product concentrates around L times the per-layer variance by the strong law of large numbers. High variance means the composed operator has widely dispersed singular values — some dimensions amplify, others attenuate — which predicts both selective memory failure and anisotropic gradient flow.

The diagonal approximation makes this O(N·L) rather than O(N³). For non-diagonal matrices, the approximation underestimates spread when off-diagonal entries create additional variance — which explains why C6 shows ρ=0.422 on Hyena (where eigenvalue spread is structural) but less signal than C3 on S4-like (where non-normality, not eigenvalue spread, is the primary failure mode).

**Rigor tag**: [MOTIVATED] — the CLT argument holds exactly under i.i.d. diagonal assumptions; behavior on non-diagonal matrices is empirically validated but not formally proven.

**Connection to architectural differences**: Hyena's circulant structure naturally produces broad eigenvalue distributions (min≈0.006, max≈0.95), making eigenvalue spread the dominant stability factor. S4-like recurrence matrices have narrower eigenvalue distributions but higher susceptibility to non-normal transient effects, making pseudospectral sensitivity more informative than spectral spread.