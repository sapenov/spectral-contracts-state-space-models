"""
Exact implementations of spectral contract metrics for SSM stability prediction.

This module implements the six candidate contract metrics (C1-C6) as defined
in the metric inventory. Each metric includes cost estimation and validation.

All metrics follow the schema from §10.3 and must outperform trivial baselines
per SC-1 criterion.
"""

import numpy as np
from scipy.linalg import svd, eigvals, solve
from typing import List, Dict, Any, Tuple, Optional
import time
import warnings


def condition_growth_exact(layer_matrices: List[np.ndarray], T: int = 100) -> Dict[str, Any]:
    """
    C1: Effective transition condition growth.

    Computes κ(A_1^T · A_2^T · ... · A_L^T) where κ is the condition number.

    Args:
        layer_matrices: List of transition matrices A_l for l=1,...,L
        T: Number of time steps for composition

    Returns:
        Dictionary with metric value and metadata

    Cost: O(N^3 * L) for full SVD of composed operator

    >>> A1 = np.array([[0.9, 0.1], [0.0, 0.8]])
    >>> A2 = np.array([[0.8, 0.2], [0.1, 0.7]])
    >>> result = condition_growth_exact([A1, A2], T=50)
    >>> result['value'] > 1.0  # Condition number always >= 1
    True
    """
    start_time = time.perf_counter()

    # Compute A_l^T for each layer
    powered_matrices = [np.linalg.matrix_power(A, T) for A in layer_matrices]

    # Compose: A_L^T · ... · A_2^T · A_1^T
    composed = powered_matrices[0]
    for A_powered in powered_matrices[1:]:
        composed = A_powered @ composed

    # Compute condition number via SVD with log-space computation
    try:
        s = svd(composed, compute_uv=False)
        s_pos = s[s > 0]
        if len(s_pos) < 2:
            condition_num = 1e8
        else:
            log_cond = np.log(s_pos[0]) - np.log(s_pos[-1])
            condition_num = min(np.exp(log_cond), 1e8)
    except np.linalg.LinAlgError:
        condition_num = 1e8

    compute_time_ms = (time.perf_counter() - start_time) * 1000

    return {
        'metric_id': 'C1',
        'name': 'Effective transition condition growth',
        'value': condition_num,
        'compute_time_ms': compute_time_ms,
        'cost_complexity': 'O(N^3 * L)',
        'rigor_tag': '[MOTIVATED]',
        'T': T,
        'L': len(layer_matrices),
        'N': layer_matrices[0].shape[0],
        'description': f'Condition number of composed operator A_L^{T} * ... * A_1^{T}'
    }


def sv_dispersion_exact(layer_matrices: List[np.ndarray]) -> Dict[str, Any]:
    """
    C2: Singular-value dispersion of stacked operator.

    Computes σ_max/σ_min of the composed operator A_L · ... · A_1.

    Args:
        layer_matrices: List of transition matrices A_l

    Returns:
        Dictionary with metric value and metadata

    Cost: O(N^3 * L) for full SVD of composed operator

    >>> A1 = np.array([[0.9, 0.1], [0.0, 0.8]])
    >>> A2 = np.array([[0.8, 0.2], [0.1, 0.7]])
    >>> result = sv_dispersion_exact([A1, A2])
    >>> result['value'] > 1.0  # Dispersion always >= 1
    True
    """
    start_time = time.perf_counter()

    # Compose matrices: A_L · ... · A_2 · A_1
    composed = layer_matrices[0]
    for A in layer_matrices[1:]:
        composed = A @ composed

    # Compute singular values
    try:
        s = svd(composed, compute_uv=False)
        if s[-1] < 1e-12:  # Numerical singularity
            dispersion = np.inf
        else:
            dispersion = s[0] / s[-1]

        # Also compute spectral spread alternative
        spectral_spread = (s[0] - s[-1]) / np.mean(s) if len(s) > 0 else 0.0

    except np.linalg.LinAlgError:
        dispersion = np.inf
        spectral_spread = np.inf

    compute_time_ms = (time.perf_counter() - start_time) * 1000

    return {
        'metric_id': 'C2',
        'name': 'Singular-value dispersion',
        'value': dispersion,
        'spectral_spread': spectral_spread,
        'compute_time_ms': compute_time_ms,
        'cost_complexity': 'O(N^3 * L)',
        'rigor_tag': '[MOTIVATED]',
        'L': len(layer_matrices),
        'N': layer_matrices[0].shape[0],
        'description': 'Ratio σ_max/σ_min of composed operator'
    }


def pseudospectral_radius_exact(A: np.ndarray, epsilon: float = 0.01,
                                grid_size: int = 50) -> Dict[str, Any]:
    """
    C3: Pseudospectral sensitivity proxy (exact version).

    Computes ε-pseudospectral radius: max{|z| : z ∈ Λ_ε(A)} where
    Λ_ε(A) = {z ∈ ℂ : σ_min(zI - A) ≤ ε}

    Args:
        A: Single transition matrix
        epsilon: Pseudospectral tolerance
        grid_size: Grid resolution for complex plane sampling

    Returns:
        Dictionary with metric value and metadata

    Cost: O(N^2 * grid_size^2) for ε-pseudospectrum on grid

    >>> A = np.array([[0.9, 0.1], [0.0, 0.8]])
    >>> result = pseudospectral_radius_exact(A, epsilon=0.1, grid_size=20)
    >>> result['value'] >= np.max(np.abs(np.linalg.eigvals(A)))
    True
    """
    start_time = time.perf_counter()

    # Get eigenvalue bounds for grid
    eigenvals = eigvals(A)
    max_eigval_mag = np.max(np.abs(eigenvals))

    # Create complex grid centered on the eigenvalue cluster
    center_real = np.mean(np.real(eigenvals))
    center_imag = np.mean(np.imag(eigenvals))
    spectral_r = np.max(np.abs(eigenvals))

    # Grid from (center - 1.2*spectral_r) to (center + 1.2*spectral_r)
    grid_radius = max(1.2 * spectral_r, 0.5)  # Minimum radius to avoid too small grids
    x = np.linspace(center_real - grid_radius, center_real + grid_radius, grid_size)
    y = np.linspace(center_imag - grid_radius, center_imag + grid_radius, grid_size)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Compute minimum singular value of (zI - A) for each z
    N = A.shape[0]
    I = np.eye(N)
    min_svs = np.zeros_like(Z, dtype=float)

    for i in range(grid_size):
        for j in range(grid_size):
            z = Z[i, j]
            try:
                s = svd(z * I - A, compute_uv=False)
                min_svs[i, j] = s[-1]  # Smallest singular value
            except np.linalg.LinAlgError:
                min_svs[i, j] = 0.0

    # Find ε-pseudospectrum: points where σ_min(zI - A) ≤ ε
    pseudospectrum_mask = min_svs <= epsilon

    if np.any(pseudospectrum_mask):
        pseudospectral_points = Z[pseudospectrum_mask]
        pseudospectral_radius = np.max(np.abs(pseudospectral_points))
    else:
        # Fallback: spectral radius if no points in ε-pseudospectrum
        pseudospectral_radius = max_eigval_mag

    compute_time_ms = (time.perf_counter() - start_time) * 1000

    return {
        'metric_id': 'C3',
        'name': 'Pseudospectral sensitivity proxy',
        'value': pseudospectral_radius,
        'spectral_radius': max_eigval_mag,
        'epsilon': epsilon,
        'grid_size': grid_size,
        'compute_time_ms': compute_time_ms,
        'cost_complexity': f'O(N^2 * {grid_size}^2)',
        'rigor_tag': '[PROVEN]',
        'N': A.shape[0],
        'description': f'ε-pseudospectral radius with ε={epsilon}'
    }


def controllability_condition_exact(A: np.ndarray, B: np.ndarray, T: int) -> Dict[str, Any]:
    """
    C4: Finite-horizon controllability proxy.

    Computes condition number of controllability Gramian:
    W_T = Σ_{t=0}^{T-1} A^t B B^T (A^T)^t

    Args:
        A: Transition matrix
        B: Input matrix
        T: Time horizon

    Returns:
        Dictionary with metric value and metadata

    Cost: O(N^2 * T + N^3) for Gramian construction and condition number

    >>> A = np.array([[0.9, 0.1], [0.0, 0.8]])
    >>> B = np.array([[1.0], [0.5]])
    >>> result = controllability_condition_exact(A, B, T=50)
    >>> result['value'] > 1.0  # Condition number always >= 1
    True
    """
    start_time = time.perf_counter()

    N = A.shape[0]

    # Build controllability Gramian W_T = Σ_{t=0}^{T-1} A^t B B^T (A^T)^t
    W = np.zeros((N, N))
    A_power = np.eye(N)

    for t in range(T):
        term = A_power @ B @ B.T @ A_power.T
        W += term
        A_power = A @ A_power

    # Compute condition number with log-space computation
    try:
        eigenvals = np.linalg.eigvals(W)
        eigenvals = eigenvals.real  # Should be real for symmetric W
        eigenvals_pos = eigenvals[eigenvals > 1e-12]  # Filter numerical zeros

        if len(eigenvals_pos) < 2:
            condition_num = 1e8
        else:
            log_cond = np.log(np.max(eigenvals_pos)) - np.log(np.min(eigenvals_pos))
            condition_num = min(np.exp(log_cond), 1e8)

    except np.linalg.LinAlgError:
        condition_num = 1e8

    compute_time_ms = (time.perf_counter() - start_time) * 1000

    return {
        'metric_id': 'C4',
        'name': 'Finite-horizon controllability proxy',
        'value': condition_num,
        'compute_time_ms': compute_time_ms,
        'cost_complexity': 'O(N^2 * T + N^3)',
        'rigor_tag': '[PROVEN]',
        'T': T,
        'N': A.shape[0],
        'description': f'Condition number of {T}-step controllability Gramian'
    }


def jacobian_anisotropy_exact(layer_matrices: List[np.ndarray], T: int) -> Dict[str, Any]:
    """
    C5: Jacobian anisotropy growth.

    Computes d/dT log(σ_max(J_T) / σ_min(J_T)) where J_T is the
    end-to-end Jacobian ∂h_T / ∂h_0 after T steps.

    Args:
        layer_matrices: List of transition matrices A_l
        T: Sequence length for anisotropy measurement

    Returns:
        Dictionary with metric value and metadata

    Cost: O(N^3 * T) for full Jacobian SVD at each step

    >>> A1 = np.array([[0.9, 0.1], [0.0, 0.8]])
    >>> A2 = np.array([[0.8, 0.2], [0.1, 0.7]])
    >>> result = jacobian_anisotropy_exact([A1, A2], T=20)
    >>> isinstance(result['value'], float)
    True
    """
    start_time = time.perf_counter()

    # For SSMs, the end-to-end Jacobian is approximately the composed transition matrix
    # In more complex models, this would require automatic differentiation

    anisotropy_values = []
    composed = np.eye(layer_matrices[0].shape[0])

    for t in range(1, T + 1):
        # Update composed Jacobian (simplified as matrix composition)
        for A in layer_matrices:
            composed = A @ composed

        # Compute condition number at step t
        try:
            s = svd(composed, compute_uv=False)
            if s[-1] < 1e-12:
                anisotropy = np.inf
            else:
                anisotropy = s[0] / s[-1]
            anisotropy_values.append(np.log(anisotropy))
        except (np.linalg.LinAlgError, RuntimeWarning):
            anisotropy_values.append(np.inf)
            break

    # Compute growth rate (slope of log-anisotropy vs. time)
    if len(anisotropy_values) > 1:
        times = np.arange(1, len(anisotropy_values) + 1)
        growth_rate = np.polyfit(times, anisotropy_values, 1)[0]  # Linear fit slope
    else:
        growth_rate = 0.0

    compute_time_ms = (time.perf_counter() - start_time) * 1000

    return {
        'metric_id': 'C5',
        'name': 'Jacobian anisotropy growth',
        'value': growth_rate,
        'anisotropy_values': anisotropy_values,
        'compute_time_ms': compute_time_ms,
        'cost_complexity': 'O(N^3 * T)',
        'rigor_tag': '[MOTIVATED]',
        'T': T,
        'L': len(layer_matrices),
        'N': layer_matrices[0].shape[0],
        'description': f'Growth rate of log-anisotropy over {T} steps'
    }


def free_prob_spectral_spread(layer_matrices: List[np.ndarray]) -> Dict[str, Any]:
    """
    C6: Free-probability-inspired composed spectral spread.

    Under diagonal approximation: A_l ≈ diag(a_l^{(1)}, ..., a_l^{(N)})
    Composed eigenvalues: λ_i^{(L)} = ∏_{l=1}^L a_l^{(i)}
    Spectral spread: max_i |λ_i^{(L)}| - min_i |λ_i^{(L)}|

    Args:
        layer_matrices: List of transition matrices A_l

    Returns:
        Dictionary with metric value and metadata

    Cost: O(N * L) for diagonal approximation

    >>> A1 = np.array([[0.9, 0.1], [0.0, 0.8]])
    >>> A2 = np.array([[0.8, 0.2], [0.1, 0.7]])
    >>> result = free_prob_spectral_spread([A1, A2])
    >>> result['value'] >= 0
    True
    """
    start_time = time.perf_counter()

    # Extract diagonal elements from each matrix
    N = layer_matrices[0].shape[0]
    diagonals = [np.diag(A) for A in layer_matrices]

    # Compute composed eigenvalues under diagonal approximation
    composed_eigenvals = np.ones(N, dtype=complex)
    for diag in diagonals:
        composed_eigenvals *= diag

    # Compute spectral spread
    eigenval_mags = np.abs(composed_eigenvals)
    spectral_spread = np.max(eigenval_mags) - np.min(eigenval_mags)

    # Alternative measure: ratio spread
    if np.min(eigenval_mags) > 1e-12:
        ratio_spread = np.max(eigenval_mags) / np.min(eigenval_mags)
    else:
        ratio_spread = np.inf

    compute_time_ms = (time.perf_counter() - start_time) * 1000

    return {
        'metric_id': 'C6',
        'name': 'Free-probability-inspired spectral spread',
        'value': spectral_spread,
        'ratio_spread': ratio_spread,
        'computed_eigenvals': composed_eigenvals,
        'compute_time_ms': compute_time_ms,
        'cost_complexity': 'O(N * L)',
        'rigor_tag': '[HEURISTIC]',
        'L': len(layer_matrices),
        'N': N,
        'description': 'Spectral spread under diagonal approximation',
        'note': 'Uses diagonal approximation - accuracy depends on off-diagonal magnitude'
    }


def compute_all_contracts(layer_matrices: List[np.ndarray],
                         B: Optional[np.ndarray] = None,
                         T: int = 100,
                         include_expensive: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Compute all contract metrics for a given SSM configuration.

    Args:
        layer_matrices: List of SSM transition matrices A_l
        B: Input matrix for controllability metric (optional)
        T: Time horizon for time-dependent metrics
        include_expensive: Whether to compute O(N^3) metrics

    Returns:
        Dictionary mapping metric IDs to their results
    """
    contracts = {}

    if include_expensive:
        # C1: Condition growth (expensive)
        contracts['C1'] = condition_growth_exact(layer_matrices, T)

        # C2: SV dispersion (expensive)
        contracts['C2'] = sv_dispersion_exact(layer_matrices)

        # C5: Jacobian anisotropy (expensive)
        contracts['C5'] = jacobian_anisotropy_exact(layer_matrices, T)

    # C3: Pseudospectral radius (moderate cost, per layer)
    contracts['C3_layers'] = {}
    for i, A in enumerate(layer_matrices):
        contracts['C3_layers'][f'layer_{i}'] = pseudospectral_radius_exact(A, epsilon=0.01, grid_size=30)

    # C3 aggregate: max across layers
    layer_values = [result['value'] for result in contracts['C3_layers'].values()]
    contracts['C3'] = {
        'metric_id': 'C3',
        'name': 'Pseudospectral sensitivity (max over layers)',
        'value': np.max(layer_values),
        'layer_values': layer_values,
        'cost_complexity': 'O(N^2 * grid^2 * L)',
        'rigor_tag': '[PROVEN]'
    }

    # C4: Controllability (if B provided)
    if B is not None:
        # Apply to first layer for simplicity
        contracts['C4'] = controllability_condition_exact(layer_matrices[0], B, T)

    # C6: Free-prob spread (always cheap)
    contracts['C6'] = free_prob_spectral_spread(layer_matrices)

    return contracts


def validate_contracts():
    """
    Validation tests for contract metric implementations.
    """
    print("Validating contract metrics...")

    # Test matrices: stable but with different properties
    A1 = np.array([[0.8, 0.1], [0.0, 0.7]])  # Upper triangular
    A2 = np.array([[0.6, 0.2], [0.1, 0.5]])  # Non-diagonal
    B = np.array([[1.0], [0.5]])

    # Test all metrics
    try:
        contracts = compute_all_contracts([A1, A2], B=B, T=20, include_expensive=True)

        # Basic validation
        for metric_id, result in contracts.items():
            if isinstance(result, dict) and 'value' in result:
                # Skip nested dictionaries (like C3_layers)
                if 'compute_time_ms' in result:
                    assert isinstance(result['compute_time_ms'], (int, float)), f"{metric_id}: Invalid compute time"
                    assert result['compute_time_ms'] >= 0, f"{metric_id}: Negative compute time"

        print("✓ All contract metrics computed successfully")

        # Display results
        for metric_id, result in contracts.items():
            if isinstance(result, dict) and 'value' in result:
                print(f"  {metric_id}: {result['value']:.4f} ({result.get('name', 'Unknown')})")

    except Exception as e:
        print(f"✗ Contract validation failed: {e}")
        raise


if __name__ == "__main__":
    validate_contracts()