"""
Demo version of the benchmark sweep for testing purposes.

Runs a smaller version of the benchmark suite to validate implementation.
"""

import numpy as np
import sys
sys.path.append('..')

from metrics.trivial_baselines import compute_all_trivial_baselines
from metrics.contracts import compute_all_contracts
from benchmarks.long_memory_tasks import create_case_a_matrix, create_case_b_matrix


def demo_case_validation():
    """
    Test Case A and Case B with metrics to validate the benchmark design.
    """
    print("Testing Case A and Case B validation...")

    N = 8  # Smaller for faster computation

    # Case A: Should have spectral radius < 1 but high instability
    print("\n--- Case A: Hidden Instability ---")
    A_case_a = create_case_a_matrix(N, target_condition=100.0)

    # Compute trivial baselines
    baselines_a = compute_all_trivial_baselines([A_case_a])

    # Compute contracts (subset for speed)
    B = np.random.randn(N, 1)
    contracts_a = compute_all_contracts([A_case_a], B=B, T=50, include_expensive=False)  # Skip expensive ones

    print(f"Spectral radius (trivial): {baselines_a['max_eigenvalue']['value']:.4f}")
    print(f"Operator norm (trivial): {baselines_a['max_operator_norm']['value']:.4f}")
    if 'C3' in contracts_a:
        print(f"Pseudospectral sensitivity (C3): {contracts_a['C3']['value']:.4f}")
    if 'C6' in contracts_a:
        print(f"Free-prob spread (C6): {contracts_a['C6']['value']:.4f}")

    # Case B: Should have spectral radius ~1 but be stable
    print("\n--- Case B: Apparent Risk ---")
    A_case_b = create_case_b_matrix(N, max_eigenvalue=0.999)

    baselines_b = compute_all_trivial_baselines([A_case_b])
    contracts_b = compute_all_contracts([A_case_b], B=B, T=50, include_expensive=False)

    print(f"Spectral radius (trivial): {baselines_b['max_eigenvalue']['value']:.4f}")
    print(f"Operator norm (trivial): {baselines_b['max_operator_norm']['value']:.4f}")
    if 'C3' in contracts_b:
        print(f"Pseudospectral sensitivity (C3): {contracts_b['C3']['value']:.4f}")
    if 'C6' in contracts_b:
        print(f"Free-prob spread (C6): {contracts_b['C6']['value']:.4f}")

    # Analysis
    print("\n--- Analysis ---")
    spectral_a = baselines_a['max_eigenvalue']['value']
    spectral_b = baselines_b['max_eigenvalue']['value']

    print(f"Case A spectral radius: {spectral_a:.4f} ({'SAFE' if spectral_a < 1.0 else 'RISKY'})")
    print(f"Case B spectral radius: {spectral_b:.4f} ({'SAFE' if spectral_b < 1.0 else 'RISKY'})")

    if 'C3' in contracts_a and 'C3' in contracts_b:
        pseudo_a = contracts_a['C3']['value']
        pseudo_b = contracts_b['C3']['value']
        print(f"Case A pseudospectral: {pseudo_a:.4f} (should be HIGH)")
        print(f"Case B pseudospectral: {pseudo_b:.4f} (should be similar to spectral radius)")

        if pseudo_a > spectral_a * 1.5:
            print("✓ Case A shows hidden instability (pseudospectral >> spectral)")
        else:
            print("⚠ Case A may not show expected hidden instability")

        if abs(pseudo_b - spectral_b) < 0.1:
            print("✓ Case B shows no hidden instability (pseudospectral ≈ spectral)")
        else:
            print("⚠ Case B shows unexpected pseudospectral behavior")


def demo_mini_sweep():
    """
    Run a mini parameter sweep to test the analysis pipeline.
    """
    print("\n" + "="*50)
    print("Mini Parameter Sweep Demo")
    print("="*50)

    N = 6
    eigenvalue_radii = [0.8, 0.9, 0.95, 1.0]
    results = []

    for i, r in enumerate(eigenvalue_radii):
        print(f"\nConfiguration {i+1}: eigenvalue_radius = {r}")

        # Create simple diagonal matrix
        eigenvals = np.linspace(r * 0.7, r, N)
        A = np.diag(eigenvals)

        # Add small perturbation for non-normality
        if r > 0.95:
            perturbation = 0.05 * np.random.randn(N, N)
            A += perturbation

        # Compute metrics
        baselines = compute_all_trivial_baselines([A])

        B = np.random.randn(N, 1)
        contracts = compute_all_contracts([A], B=B, T=30, include_expensive=False)

        # Simulate instability based on spectral radius
        simulated_unstable = r >= 0.98

        result = {
            'config_id': i + 1,
            'eigenvalue_radius': r,
            'spectral_radius': baselines['max_eigenvalue']['value'],
            'operator_norm': baselines['max_operator_norm']['value'],
            'pseudospectral_c3': contracts.get('C3', {}).get('value', None),
            'free_prob_c6': contracts.get('C6', {}).get('value', None),
            'simulated_unstable': simulated_unstable
        }
        results.append(result)

        print(f"  Spectral radius: {result['spectral_radius']:.4f}")
        print(f"  Simulated unstable: {result['simulated_unstable']}")
        if result['pseudospectral_c3'] is not None:
            print(f"  Pseudospectral C3: {result['pseudospectral_c3']:.4f}")

    # Analysis
    print("\n--- Correlation Analysis Preview ---")
    spectral_values = [r['spectral_radius'] for r in results]
    unstable_values = [r['simulated_unstable'] for r in results]

    print(f"Spectral radii: {spectral_values}")
    print(f"Instabilities: {unstable_values}")

    # Simple correlation check
    from scipy.stats import spearmanr
    if len(set(unstable_values)) > 1:  # Need variation in outcomes
        correlation, p_value = spearmanr(spectral_values, unstable_values)
        print(f"Spectral radius vs instability: ρ = {correlation:.3f}, p = {p_value:.3f}")
    else:
        print("All configurations have same stability outcome - need more variation")

    return results


def main():
    """Run demo benchmark suite."""
    print("SSM Spectral Contracts - Demo Benchmark Suite")
    print("=" * 50)

    # Test Case A and Case B
    demo_case_validation()

    # Test mini sweep
    results = demo_mini_sweep()

    print(f"\n✓ Demo completed successfully!")
    print(f"Generated {len(results)} test configurations")
    print("\nNext steps:")
    print("1. Run full benchmark sweep with run_sweeps.py")
    print("2. Implement WS3 predictiveness analysis")
    print("3. Build CLI tool (WS4)")


if __name__ == "__main__":
    main()