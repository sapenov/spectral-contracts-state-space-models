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


def composed_jacobian_norm(layer_matrices: List[np.ndarray]) -> Dict[str, Any]:
    """
    Composed Jacobian norm baseline for linear SSM regime.

    In the linear dynamics regime, the gradient norm proxy is the Frobenius norm
    of the composed transition matrix (end-to-end Jacobian from input to output).
    This captures the overall amplification factor without requiring loss functions.

    Args:
        layer_matrices: List of SSM transition matrices A_l

    Returns:
        Dictionary with gradient proxy baseline

    >>> A1 = np.array([[0.9, 0.1], [0.0, 0.8]])
    >>> A2 = np.array([[0.7, 0.2], [0.1, 0.6]])
    >>> result = composed_jacobian_norm([A1, A2])
    >>> result['value'] > 0
    True
    """
    start_time = time.perf_counter()

    # Compose all layers: A_L @ ... @ A_2 @ A_1
    composed = layer_matrices[0]
    for A in layer_matrices[1:]:
        composed = A @ composed

    # Gradient norm proxy: Frobenius norm of composed Jacobian
    grad_norm_proxy = np.linalg.norm(composed, ord='fro')

    compute_time_ms = (time.perf_counter() - start_time) * 1000

    return {
        'metric_id': 'trivial_grad_proxy',
        'value': grad_norm_proxy,
        'compute_time_ms': compute_time_ms,
        'description': 'Composed Jacobian Frobenius norm (gradient proxy for linear regime)',
        'cost_complexity': 'O(N^3 * L)',
        'note': 'Linear regime proxy - Frobenius norm of composed transition matrix'
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
    baselines['composed_jacobian'] = composed_jacobian_norm(layer_matrices)

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