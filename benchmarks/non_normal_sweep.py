"""
Non-Normal Sweep: The Actual Test for Spectral Contracts

This implements the correct test regime: eigenvalue_radius × condition_V sweep
to verify if spectral contracts outperform trivial baselines in the non-normal regime.

This is what the contracts were designed for, unlike the diagonal-matrix sweep.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import sys
sys.path.append('..')

from metrics.trivial_baselines import compute_all_trivial_baselines
from metrics.contracts import compute_all_contracts
from benchmarks.long_memory_tasks import (
    create_non_normal_ssm_matrix, compute_linear_stability_outcome,
    compute_memory_retention_outcome
)


def run_non_normal_sweep(eigenvalue_radii: List[float] = [0.85, 0.90, 0.95, 0.99],
                        condition_V_values: List[float] = [1.0, 10.0, 100.0, 1000.0],
                        N: int = 32, L: int = 2, T_test: int = 200,
                        seeds: List[int] = [1, 2, 3]) -> List[Dict[str, Any]]:
    """
    Run two-axis sweep: eigenvalue_radius × condition_V.

    This is the regime test for spectral contracts:
    - condition_V = 1.0: Normal matrices (spectral radius should be sufficient)
    - condition_V = 1000.0: Non-normal matrices (contracts should outperform)

    Args:
        eigenvalue_radii: Range of spectral radii to test
        condition_V_values: Range of eigenvector conditioning to test
        N: State dimension (kept small for computational efficiency)
        L: Number of layers
        T_test: Time horizon for stability testing
        seeds: Random seeds

    Returns:
        List of experiment results with legitimate outcomes
    """
    results = []
    config_id = 0

    print(f"Running non-normal sweep: {len(eigenvalue_radii)}×{len(condition_V_values)}×{len(seeds)} = "
          f"{len(eigenvalue_radii)*len(condition_V_values)*len(seeds)} configurations")

    for r in eigenvalue_radii:
        for cond_V in condition_V_values:
            for seed in seeds:
                config_id += 1
                print(f"  Config {config_id:02d}: r={r:.3f}, κ(V)={cond_V:g}, seed={seed}")

                # Create non-normal matrices for this configuration
                layer_matrices = []
                for layer_idx in range(L):
                    A = create_non_normal_ssm_matrix(
                        N=N, eigenvalue_radius=r, condition_V=cond_V,
                        seed=seed + layer_idx  # Different seed per layer
                    )
                    layer_matrices.append(A)

                # Compute contract metrics
                B = np.random.RandomState(seed).randn(N, 1)

                # Trivial baselines
                baseline_results = compute_all_trivial_baselines(layer_matrices)

                # Contract metrics (use moderate T to avoid computational explosion)
                contract_results = compute_all_contracts(
                    layer_matrices, B=B, T=min(T_test, 50), include_expensive=True
                )

                # Compute legitimate training outcomes
                stability_outcome = compute_linear_stability_outcome(
                    layer_matrices, T_test=T_test, n_trials=3
                )
                memory_outcome = compute_memory_retention_outcome(
                    layer_matrices, T_test=T_test
                )

                # Record detailed matrix properties for analysis
                A_sample = layer_matrices[0]
                eigenvals, eigenvecs = np.linalg.eig(A_sample)
                matrix_properties = {
                    'actual_spectral_radius': np.max(np.abs(eigenvals)),
                    'actual_condition_V': np.linalg.cond(eigenvecs),
                    'is_nearly_diagonal': np.allclose(A_sample, np.diag(np.diag(A_sample)), atol=1e-3),
                    'off_diagonal_norm_ratio': np.linalg.norm(A_sample - np.diag(np.diag(A_sample))) / np.linalg.norm(A_sample)
                }

                # Store complete result
                result = {
                    'config_id': f'NN{config_id:02d}',
                    'sweep_type': 'non_normal',
                    'target_eigenvalue_radius': r,
                    'target_condition_V': cond_V,
                    'N': N,
                    'L': L,
                    'T_test': T_test,
                    'seed': seed,
                    'matrix_properties': matrix_properties,
                    'baseline_metrics': baseline_results,
                    'contract_metrics': contract_results,
                    'stability_outcome': stability_outcome,
                    'memory_outcome': memory_outcome
                }
                results.append(result)

    return results


def validate_non_normal_construction():
    """
    Validate that the non-normal matrix construction works as expected.
    """
    print("Validating non-normal matrix construction...")

    N = 16
    radius = 0.9

    for condition_target in [1.0, 10.0, 100.0]:
        print(f"\n--- Target κ(V) = {condition_target} ---")

        A = create_non_normal_ssm_matrix(N, radius, condition_target, seed=42)

        eigenvals, eigenvecs = np.linalg.eig(A)
        actual_radius = np.max(np.abs(eigenvals))
        actual_condition = np.linalg.cond(eigenvecs)

        print(f"Actual spectral radius: {actual_radius:.4f} (target: {radius})")
        print(f"Actual κ(V): {actual_condition:.1f} (target: {condition_target})")
        print(f"Is diagonal: {np.allclose(A, np.diag(np.diag(A)))}")

        # Test pseudospectral behavior
        from metrics.contracts import pseudospectral_radius_exact
        pseudo_result = pseudospectral_radius_exact(A, epsilon=0.01)
        pseudo_radius = pseudo_result['value']
        print(f"Pseudospectral radius: {pseudo_radius:.4f}")
        print(f"Pseudo/spectral ratio: {pseudo_radius/actual_radius:.4f} "
              f"({'NON-NORMAL' if pseudo_radius/actual_radius > 1.1 else 'normal-ish'})")

    print("\n✓ Non-normal construction validation complete")


if __name__ == "__main__":
    # Validate construction
    validate_non_normal_construction()

    # Run mini non-normal sweep
    print("\\n" + "="*60)
    print("Running Mini Non-Normal Sweep")
    print("="*60)

    results = run_non_normal_sweep(
        eigenvalue_radii=[0.90, 0.95],
        condition_V_values=[1.0, 100.0],  # Normal vs non-normal
        N=16, L=2, T_test=100,
        seeds=[1, 2]
    )

    print(f"\\nGenerated {len(results)} configurations")

    # Analyze results
    from analysis.regression_analysis import PredictivenesStudy
    study = PredictivenesStudy(results)
    study.compute_correlations()
    study.compute_auroc_analysis()
    summary_df = study.print_analysis_summary()

    print("\\n=== NON-NORMAL SWEEP RESULTS ===")
    print("Key question: Do contracts beat trivial baselines in non-normal regime?")

    # Export to CSV for WS3 analysis
    export_non_normal_to_csv(results, "results/non_normal_sweep.csv")
    print(f"Results exported to results/non_normal_sweep.csv")


def export_non_normal_to_csv(results: List[Dict[str, Any]], filename: str):
    """Export non-normal sweep results to CSV format for WS3 analysis."""
    rows = []

    for result in results:
        # Base configuration
        base_row = {
            'config_id': result['config_id'],
            'sweep_type': result['sweep_type'],
            'target_eigenvalue_radius': result['target_eigenvalue_radius'],
            'target_condition_V': result['target_condition_V'],
            'actual_spectral_radius': result['matrix_properties']['actual_spectral_radius'],
            'actual_condition_V': result['matrix_properties']['actual_condition_V'],
            'N': result['N'],
            'L': result['L'],
            'T_test': result['T_test'],
            'seed': result['seed']
        }

        # Add baseline metrics
        for metric_name, metric_data in result['baseline_metrics'].items():
            if isinstance(metric_data, dict) and 'value' in metric_data:
                if metric_data['value'] is not None:
                    base_row[f'trivial_{metric_name}'] = metric_data['value']

        # Add contract metrics
        for metric_name, metric_data in result['contract_metrics'].items():
            if isinstance(metric_data, dict) and 'value' in metric_data:
                base_row[f'contract_{metric_name}'] = metric_data['value']

        # Add outcomes
        stability = result['stability_outcome']
        memory = result['memory_outcome']
        base_row.update({
            'diverged': stability['diverged'],
            'growth_ratio': stability['growth_ratio'],
            'log_growth': stability['log_growth'],
            'memory_retention': memory['memory_retention'],
            'log_retention': memory['log_retention'],
            'final_norm': memory['final_norm']
        })

        rows.append(base_row)

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    return df