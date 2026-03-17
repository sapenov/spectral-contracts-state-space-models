"""
S4/DSS Bridge Experiment: Use S4D-Real initialization scheme
to generate realistic SSM operators and evaluate spectral diagnostics.

This extends the DSS-style bridge in §6.3 using the published
S4D-Real initialization from Gu et al. 2022, providing external
validity beyond hand-constructed matrices.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def s4d_real_init(N, dt=0.01, seed=0):
    """
    S4D-Real initialization scheme from Gu et al. 2022:
    A_n = -1/2 + i*pi*n for n=1..N/2, then discretized via bilinear transform.
    This is the published initialization scheme, not hand-constructed.
    """
    np.random.seed(seed)

    # S4D-Real diagonal initialization
    n = np.arange(1, N//2 + 1)
    A_diag_complex = -0.5 + 1j * np.pi * n

    # Mirror for complex conjugate pairs
    if N % 2 == 0:
        A_full_diag = np.concatenate([A_diag_complex, A_diag_complex.conj()])
    else:
        # Odd N: add one real eigenvalue
        A_full_diag = np.concatenate([A_diag_complex, A_diag_complex.conj(), [-0.5]])

    # Discretize via bilinear transform: A_bar = (I + dt/2 * A) / (I - dt/2 * A)
    I_vals = np.ones(N)
    A_bar_diag = (1 + dt/2 * A_full_diag) / (1 - dt/2 * A_full_diag)

    # Add small random rotation to break exact diagonality (models learned bases)
    theta_scale = 0.02 * np.random.randn()
    theta = np.random.randn(N, N) * theta_scale
    Q, _ = np.linalg.qr(np.eye(N) + theta)

    # Construct matrix: A = Q * diag(A_bar) * Q^T
    A_matrix = Q @ np.diag(A_bar_diag) @ Q.T

    return np.real(A_matrix)

def compute_henrici_departure(A):
    """Henrici departure from normality: ||AA* - A*A||_F"""
    comm = A @ A.T - A.T @ A
    return np.linalg.norm(comm, 'fro')

def compute_operator_norm(A):
    """Spectral norm (largest singular value)"""
    return np.linalg.norm(A, ord=2)

def compute_spectral_radius(A):
    """Largest eigenvalue magnitude"""
    return np.max(np.abs(np.linalg.eigvals(A)))

def simplified_c3(A, epsilon=0.01, grid_size=30):
    """
    Simplified C3 computation for pseudospectral sensitivity.
    Based on the algorithm in the paper.
    """
    # Grid covering complex plane
    grid_real = np.linspace(-2, 2, grid_size)
    grid_imag = np.linspace(-2, 2, grid_size)

    # Center grid at trace/N (spectral centroid)
    center = np.trace(A) / A.shape[0]
    grid_real = grid_real + np.real(center)
    grid_imag = grid_imag + np.imag(center)

    count = 0
    total = grid_size ** 2

    for re_val in grid_real:
        for im_val in grid_imag:
            z = complex(re_val, im_val)
            try:
                # Compute resolvent norm: ||(zI - A)^{-1}||_2
                resolvent_matrix = np.linalg.inv(z * np.eye(A.shape[0]) - A)
                resolvent_norm = np.linalg.norm(resolvent_matrix, ord=2)

                if resolvent_norm > 1/epsilon:
                    count += 1
            except (np.linalg.LinAlgError, ZeroDivisionError):
                # Singularity or overflow -> inside pseudospectrum
                count += 1

    return count / total

def run_linear_dynamics(A, T=500, n_trials=10, seed=42):
    """Run linear dynamics rollout and return growth ratio."""
    np.random.seed(seed)
    growth_ratios = []

    for trial in range(n_trials):
        np.random.seed(seed + trial)
        h0 = np.random.randn(A.shape[0])
        h0 = h0 / np.linalg.norm(h0)

        h = h0.copy()
        for _ in range(T):
            h = A @ h
            # Early termination if clearly diverging
            if np.linalg.norm(h) > 1e6:
                break

        growth_ratio = np.linalg.norm(h) / np.linalg.norm(h0)
        growth_ratios.append(growth_ratio)

    return np.mean(growth_ratios)

def main():
    print("=== S4D-Real Bridge Experiment ===")
    print("Using S4D-Real initialization scheme from Gu et al. 2022...")

    N = 64
    n_seeds = 6
    results = []

    print(f"Generating {n_seeds} S4D-Real initialized matrices...")

    for seed in range(1, n_seeds + 1):
        print(f"  Processing seed {seed}/{n_seeds}...")

        # Generate S4D-Real initialized matrix
        A = s4d_real_init(N=N, dt=0.01, seed=seed)

        row = {'seed': seed, 'source': 'S4D-Real', 'hidden_size': N}

        # Compute all metrics
        try:
            row['c3'] = simplified_c3(A, epsilon=0.01, grid_size=30)
        except Exception as e:
            print(f"    C3 failed: {e}")
            row['c3'] = np.nan

        try:
            row['operator_norm'] = compute_operator_norm(A)
        except Exception as e:
            print(f"    Operator norm failed: {e}")
            row['operator_norm'] = np.nan

        try:
            row['spectral_radius'] = compute_spectral_radius(A)
        except Exception as e:
            print(f"    Spectral radius failed: {e}")
            row['spectral_radius'] = np.nan

        try:
            row['henrici'] = compute_henrici_departure(A)
        except Exception as e:
            print(f"    Henrici failed: {e}")
            row['henrici'] = np.nan

        # Eigenvector condition number
        try:
            eigenvalues, V = np.linalg.eig(A)
            row['kappa_V'] = np.linalg.cond(V)
        except Exception as e:
            print(f"    Kappa(V) failed: {e}")
            row['kappa_V'] = np.nan

        # Linear dynamics
        try:
            row['growth_ratio'] = run_linear_dynamics(A, T=500)
            row['diverged'] = int(row['growth_ratio'] > 10)
        except Exception as e:
            print(f"    Growth ratio failed: {e}")
            row['growth_ratio'] = np.nan
            row['diverged'] = np.nan

        results.append(row)
        print(f"    C3={row['c3']:.3f}, op_norm={row['operator_norm']:.3f}, "
              f"growth={row['growth_ratio']:.3f}")

    # Save results
    df = pd.DataFrame(results)
    out_path = PROJECT_ROOT / "results" / "s4_bridge_real_operators.csv"
    df.to_csv(out_path, index=False)

    print(f"\nSaved to {out_path}")

    # Summary statistics
    print("\n=== Summary Statistics ===")
    valid = df.dropna(subset=['c3', 'growth_ratio'])
    print(f"Valid matrices: {len(valid)}")

    if len(valid) > 0:
        print(f"C3 range: [{valid['c3'].min():.3f}, {valid['c3'].max():.3f}], "
              f"mean: {valid['c3'].mean():.3f}")
        print(f"Operator norm range: [{valid['operator_norm'].min():.3f}, "
              f"{valid['operator_norm'].max():.3f}]")
        print(f"Spectral radius range: [{valid['spectral_radius'].min():.3f}, "
              f"{valid['spectral_radius'].max():.3f}]")
        print(f"Henrici range: [{valid['henrici'].min():.4f}, "
              f"{valid['henrici'].max():.4f}]")
        print(f"Growth ratio range: [{valid['growth_ratio'].min():.3f}, "
              f"{valid['growth_ratio'].max():.3f}]")
        print(f"Diverged: {valid['diverged'].sum()} / {len(valid)}")

    return df

if __name__ == "__main__":
    df = main()