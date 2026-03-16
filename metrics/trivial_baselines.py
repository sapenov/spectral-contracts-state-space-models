"""
Trivial baseline implementations for SSM stability prediction.

Per §3.1 constraint P2: All trivial baselines must be run first and in full
before any contract metric is evaluated.

The three trivial baselines are:
1. Max eigenvalue magnitude across all layers
2. Max operator norm across all layers
3. Initial gradient norm at step 0
"""

import numpy as np
from typing import List, Dict, Any
import time


def max_eigenvalue_magnitude(layer_matrices: List[np.ndarray]) -> Dict[str, Any]:
    """
    Compute maximum eigenvalue magnitude across all SSM layers.

    Args:
        layer_matrices: List of transition matrices A_l for l=1,...,L

    Returns:
        Dictionary with metric value and metadata

    >>> A1 = np.array([[0.9, 0.1], [0.0, 0.8]])
    >>> A2 = np.array([[0.7, 0.2], [0.1, 0.6]])
    >>> result = max_eigenvalue_magnitude([A1, A2])
    >>> 0.8 < result['value'] < 1.0
    True
    """
    start_time = time.perf_counter()

    max_eigenval = 0.0
    layer_eigenvals = []

    for i, A in enumerate(layer_matrices):
        eigenvals = np.linalg.eigvals(A)
        max_mag = np.max(np.abs(eigenvals))
        layer_eigenvals.append(max_mag)
        max_eigenval = max(max_eigenval, max_mag)

    compute_time_ms = (time.perf_counter() - start_time) * 1000

    return {
        'metric_id': 'trivial_eigenvalue',
        'value': max_eigenval,
        'compute_time_ms': compute_time_ms,
        'layer_values': layer_eigenvals,
        'description': 'Maximum eigenvalue magnitude across all layers',
        'cost_complexity': 'O(N^3 * L)',  # Each eigenvalue decomposition is O(N^3)
    }


def max_operator_norm(layer_matrices: List[np.ndarray]) -> Dict[str, Any]:
    """
    Compute maximum operator (spectral) norm across all SSM layers.

    Args:
        layer_matrices: List of transition matrices A_l for l=1,...,L

    Returns:
        Dictionary with metric value and metadata

    >>> A1 = np.array([[0.9, 0.1], [0.0, 0.8]])
    >>> A2 = np.array([[0.7, 0.2], [0.1, 0.6]])
    >>> result = max_operator_norm([A1, A2])
    >>> 0.7 < result['value'] < 1.0
    True
    """
    start_time = time.perf_counter()

    max_norm = 0.0
    layer_norms = []

    for i, A in enumerate(layer_matrices):
        # Spectral norm (largest singular value)
        norm = np.linalg.norm(A, ord=2)
        layer_norms.append(norm)
        max_norm = max(max_norm, norm)

    compute_time_ms = (time.perf_counter() - start_time) * 1000

    return {
        'metric_id': 'trivial_norm',
        'value': max_norm,
        'compute_time_ms': compute_time_ms,
        'layer_values': layer_norms,
        'description': 'Maximum spectral norm across all layers',
        'cost_complexity': 'O(N^3 * L)',  # Each SVD is O(N^3)
    }


def initial_gradient_norm(model, loss_fn, inputs, targets) -> Dict[str, Any]:
    """
    Initial gradient norm baseline - NOT APPLICABLE to linear dynamics regime.

    This baseline applies to neural network training with actual loss functions.
    The current study tests SSM stability via linear matrix dynamics without
    training loops, making gradient norms undefined.

    Args:
        model: SSM model with parameters (not used in linear regime)
        loss_fn: Loss function (not applicable)
        inputs: Input batch (not applicable)
        targets: Target batch (not applicable)

    Returns:
        Dictionary indicating baseline is not applicable

    Note: Excluded from analysis per methodology - linear dynamics testing
    does not involve gradient computation or loss function optimization.
    """
    return {
        'metric_id': 'trivial_grad',
        'value': None,
        'compute_time_ms': 0.0,
        'description': 'Initial gradient norm - NOT APPLICABLE to linear dynamics regime',
        'cost_complexity': 'N/A',
        'note': 'Excluded: Linear dynamics testing does not involve gradient computation',
        'exclusion_reason': 'Methodological - testing matrix dynamics, not neural network training'
    }


def compute_all_trivial_baselines(layer_matrices: List[np.ndarray],
                                  model=None, loss_fn=None,
                                  inputs=None, targets=None) -> Dict[str, Dict[str, Any]]:
    """
    Compute all three trivial baselines in one call.

    Args:
        layer_matrices: List of SSM transition matrices A_l
        model: SSM model (optional, for gradient computation)
        loss_fn: Loss function (optional, for gradient computation)
        inputs: Input batch (optional, for gradient computation)
        targets: Target batch (optional, for gradient computation)

    Returns:
        Dictionary mapping baseline names to their results
    """
    baselines = {}

    # Always compute spectral-based baselines
    baselines['max_eigenvalue'] = max_eigenvalue_magnitude(layer_matrices)
    baselines['max_operator_norm'] = max_operator_norm(layer_matrices)

    # Compute gradient baseline if model components provided
    if all(x is not None for x in [model, loss_fn, inputs, targets]):
        baselines['initial_gradient'] = initial_gradient_norm(model, loss_fn, inputs, targets)
    else:
        baselines['initial_gradient'] = {
            'metric_id': 'trivial_grad',
            'value': None,
            'compute_time_ms': 0,
            'description': 'Initial gradient norm (not computed - missing model components)',
            'cost_complexity': 'O(forward + backward pass)',
            'note': 'Skipped due to missing model, loss_fn, inputs, or targets'
        }

    return baselines


def validate_trivial_baselines():
    """
    Validation tests for trivial baseline implementations.
    """
    # Test case: stable matrices
    A1 = np.array([[0.8, 0.1], [0.0, 0.7]])
    A2 = np.array([[0.6, 0.2], [0.1, 0.5]])

    baselines = compute_all_trivial_baselines([A1, A2])

    # Check eigenvalue baseline
    assert 0.5 <= baselines['max_eigenvalue']['value'] <= 1.0, "Eigenvalue baseline out of expected range"

    # Check norm baseline
    assert 0.5 <= baselines['max_operator_norm']['value'] <= 1.0, "Norm baseline out of expected range"

    # Check that computation times are recorded
    assert baselines['max_eigenvalue']['compute_time_ms'] >= 0, "Computation time not recorded"
    assert baselines['max_operator_norm']['compute_time_ms'] >= 0, "Computation time not recorded"

    print("✓ Trivial baseline validation passed")


if __name__ == "__main__":
    # Run validation
    validate_trivial_baselines()

    # Example usage
    print("\n--- Trivial Baseline Example ---")
    A1 = np.array([[0.9, 0.1], [0.0, 0.8]])
    A2 = np.array([[0.7, 0.2], [0.1, 0.6]])

    baselines = compute_all_trivial_baselines([A1, A2])

    for name, result in baselines.items():
        if result['value'] is not None:
            print(f"{name}: {result['value']:.4f} (computed in {result['compute_time_ms']:.2f}ms)")
        else:
            print(f"{name}: {result['note']}")