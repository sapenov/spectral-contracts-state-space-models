#!/usr/bin/env python3
"""
Test script for non-normal regime analysis.
"""

import sys
sys.path.insert(0, '.')
from benchmarks.non_normal_sweep import run_non_normal_sweep, validate_non_normal_construction

if __name__ == "__main__":
    validate_non_normal_construction()

    # Run the actual test
    print("\n" + "="*60)
    print("Running Non-Normal Regime Test")
    print("="*60)

    results = run_non_normal_sweep(
        eigenvalue_radii=[0.90, 0.95],
        condition_V_values=[1.0, 100.0],  # Normal vs highly non-normal
        N=16, L=1, T_test=100,  # Single layer to isolate non-normality effect
        seeds=[1, 2]
    )

    print(f"\nGenerated {len(results)} configurations")

    # Quick analysis
    print("\nQuick correlation check:")
    for result in results:
        props = result['matrix_properties']
        outcomes = result['stability_outcome']

        print(f"κ(V)={props['actual_condition_V']:.1f}, "
              f"diverged={outcomes['diverged']}, "
              f"growth_ratio={outcomes['growth_ratio']:.2f}")