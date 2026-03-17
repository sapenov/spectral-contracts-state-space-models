#!/usr/bin/env python3
"""
Generate figures for TMLR paper submission.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import sys
import os

# Set up matplotlib for academic papers
try:
    plt.style.use('seaborn-v0_8-paper')
except:
    plt.style.use('default')  # Fallback if seaborn not available
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'figure.dpi': 300
})

def wilson_confidence_interval(successes, trials, confidence=0.95):
    """
    Wilson confidence interval for binomial proportion.
    """
    if trials == 0:
        return 0, 0

    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / trials

    denominator = 1 + z**2 / trials
    center = (p + z**2 / (2 * trials)) / denominator
    margin = z / denominator * np.sqrt(p * (1 - p) / trials + z**2 / (4 * trials**2))

    return max(0, center - margin), min(1, center + margin)


def generate_calibration_curve():
    """Generate Figure 1: Calibration curve for C3 vs operator norm."""
    print("Generating Figure 1: C3 Calibration Curve...")

    # Load S4-like data
    df = pd.read_csv('results/s4_like_nonormal_sweep_final.csv')

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    # C3 calibration curve
    c3_vals = df['contract_C3'].dropna()
    diverged = df.loc[c3_vals.index, 'diverged'].astype(int)

    # Create 8 bins (more stable than 10 for n=75)
    n_bins = 8
    bin_edges = np.percentile(c3_vals, np.linspace(0, 100, n_bins + 1))
    bin_centers = []
    observed_rates = []
    lower_bounds = []
    upper_bounds = []

    for i in range(n_bins):
        mask = (c3_vals >= bin_edges[i]) & (c3_vals < bin_edges[i + 1])
        if i == n_bins - 1:  # Include max value in last bin
            mask = (c3_vals >= bin_edges[i]) & (c3_vals <= bin_edges[i + 1])

        bin_diverged = diverged[mask]
        if len(bin_diverged) >= 3:  # Minimum sample size
            rate = bin_diverged.mean()
            center = (bin_edges[i] + bin_edges[i + 1]) / 2

            # Wilson confidence interval
            lower, upper = wilson_confidence_interval(bin_diverged.sum(), len(bin_diverged))

            bin_centers.append(center)
            observed_rates.append(rate)
            lower_bounds.append(lower)
            upper_bounds.append(upper)

    # Normalize bin centers to [0, 1] for plotting
    bin_centers = np.array(bin_centers)
    bin_centers_norm = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min())

    # Plot C3 calibration
    yerr_lower = np.maximum(0, np.array(observed_rates) - np.array(lower_bounds))
    yerr_upper = np.maximum(0, np.array(upper_bounds) - np.array(observed_rates))

    ax.errorbar(bin_centers_norm, observed_rates,
               yerr=[yerr_lower, yerr_upper],
               fmt='o-', label='C3 (Pseudospectral Sensitivity)',
               color='blue', capsize=4, linewidth=2, markersize=6)

    # Operator norm calibration curve
    norm_vals = df['trivial_max_operator_norm'].dropna()
    norm_diverged = df.loc[norm_vals.index, 'diverged'].astype(int)

    norm_bin_edges = np.percentile(norm_vals, np.linspace(0, 100, n_bins + 1))
    norm_bin_centers = []
    norm_observed_rates = []
    norm_lower_bounds = []
    norm_upper_bounds = []

    for i in range(n_bins):
        mask = (norm_vals >= norm_bin_edges[i]) & (norm_vals < norm_bin_edges[i + 1])
        if i == n_bins - 1:
            mask = (norm_vals >= norm_bin_edges[i]) & (norm_vals <= norm_bin_edges[i + 1])

        bin_diverged = norm_diverged[mask]
        if len(bin_diverged) >= 3:
            rate = bin_diverged.mean()
            center = (norm_bin_edges[i] + norm_bin_edges[i + 1]) / 2

            lower, upper = wilson_confidence_interval(bin_diverged.sum(), len(bin_diverged))

            norm_bin_centers.append(center)
            norm_observed_rates.append(rate)
            norm_lower_bounds.append(lower)
            norm_upper_bounds.append(upper)

    # Normalize operator norm bin centers
    norm_bin_centers = np.array(norm_bin_centers)
    norm_bin_centers_norm = (norm_bin_centers - norm_bin_centers.min()) / (norm_bin_centers.max() - norm_bin_centers.min())

    # Plot operator norm calibration
    norm_yerr_lower = np.maximum(0, np.array(norm_observed_rates) - np.array(norm_lower_bounds))
    norm_yerr_upper = np.maximum(0, np.array(norm_upper_bounds) - np.array(norm_observed_rates))

    ax.errorbar(norm_bin_centers_norm, norm_observed_rates,
               yerr=[norm_yerr_lower, norm_yerr_upper],
               fmt='s--', label='Operator Norm (Trivial Baseline)',
               color='red', capsize=4, linewidth=2, markersize=5)

    # Perfect calibration reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='Perfect Calibration')

    ax.set_xlabel('Predicted Risk (Normalized Metric Value)')
    ax.set_ylabel('Observed Divergence Rate')
    ax.set_title('Calibration: C3 vs. Operator Norm (Recurrence-Inspired Family)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig('paper/figures/fig1_calibration_c3.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/fig1_calibration_c3.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Figure 1 saved: paper/figures/fig1_calibration_c3.{pdf,png}")


def generate_taxonomy_contrast():
    """Generate Figure 2: Architecture-dependent diagnostic contrast."""
    print("Generating Figure 2: Taxonomy Contrast...")

    # Load both datasets
    s4_df = pd.read_csv('results/s4_like_nonormal_sweep_final.csv')
    hyena_df = pd.read_csv('results/hyena_like_powered_sweep.csv')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: Recurrence-inspired family
    s4_norm = s4_df['trivial_max_operator_norm']
    s4_c3 = s4_df['contract_C3']
    s4_diverged = s4_df['diverged'].astype(bool)

    # Add jitter to prevent overplotting
    np.random.seed(42)
    s4_norm_jitter = s4_norm + np.random.normal(0, 0.008 * (s4_norm.max() - s4_norm.min()), len(s4_norm))
    s4_c3_jitter = s4_c3 + np.random.normal(0, 0.008 * (s4_c3.max() - s4_c3.min()), len(s4_c3))

    # Plot stable (blue) and diverged (red) points
    ax1.scatter(s4_norm_jitter[~s4_diverged], s4_c3_jitter[~s4_diverged],
               c='steelblue', alpha=0.45, s=18, label='Stable', zorder=2)
    ax1.scatter(s4_norm_jitter[s4_diverged], s4_c3_jitter[s4_diverged],
               c='firebrick', alpha=0.55, s=22, label='Diverged', marker='*', zorder=3)

    ax1.set_xlabel('Max Operator Norm (Trivial Baseline)')
    ax1.set_ylabel('C3 Pseudospectral Sensitivity')
    ax1.set_title('Recurrence-inspired: C3 adds signal')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add correlation text
    s4_rho_c3, _ = stats.spearmanr(s4_c3, s4_df.loc[s4_c3.index, 'growth_ratio'])
    s4_rho_norm, _ = stats.spearmanr(s4_norm, s4_df.loc[s4_norm.index, 'growth_ratio'])
    ax1.text(0.05, 0.95, f'C3: ρ = {s4_rho_c3:.3f}\nOp. norm: ρ = {s4_rho_norm:.3f}',
            transform=ax1.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Right panel: Convolution-inspired family
    hyena_norm = hyena_df['trivial_max_operator_norm']
    hyena_c3 = hyena_df['contract_C3']
    hyena_diverged = hyena_df['diverged'].astype(bool)

    # Add jitter to prevent overplotting
    hyena_norm_jitter = hyena_norm + np.random.normal(0, 0.008 * (hyena_norm.max() - hyena_norm.min()), len(hyena_norm))
    hyena_c3_jitter = hyena_c3 + np.random.normal(0, 0.008 * (hyena_c3.max() - hyena_c3.min()), len(hyena_c3))

    ax2.scatter(hyena_norm_jitter[~hyena_diverged], hyena_c3_jitter[~hyena_diverged],
               c='steelblue', alpha=0.45, s=18, label='Stable', zorder=2)
    ax2.scatter(hyena_norm_jitter[hyena_diverged], hyena_c3_jitter[hyena_diverged],
               c='firebrick', alpha=0.55, s=22, label='Diverged', marker='*', zorder=3)

    ax2.set_xlabel('Max Operator Norm (Trivial Baseline)')
    ax2.set_ylabel('C3 Pseudospectral Sensitivity')
    ax2.set_title('Convolution-inspired: operator norm sufficient')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add correlation text
    hyena_rho_c3, _ = stats.spearmanr(hyena_c3, hyena_df.loc[hyena_c3.index, 'growth_ratio'])
    hyena_rho_norm, _ = stats.spearmanr(hyena_norm, hyena_df.loc[hyena_norm.index, 'growth_ratio'])
    ax2.text(0.05, 0.95, f'C3: ρ = {hyena_rho_c3:.3f}\nOp. norm: ρ = {hyena_rho_norm:.3f}',
            transform=ax2.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('paper/figures/fig2_taxonomy_contrast.pdf', dpi=300, bbox_inches='tight')
    plt.savefig('paper/figures/fig2_taxonomy_contrast.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Figure 2 saved: paper/figures/fig2_taxonomy_contrast.{pdf,png}")


if __name__ == "__main__":
    print("=== GENERATING TMLR PAPER FIGURES ===")

    # Set working directory
    os.chdir('/mnt/c/Users/Khazret/PycharmProjects/Spectral_Contracts')

    try:
        generate_calibration_curve()
        generate_taxonomy_contrast()

        # Verify files created
        print("\n=== VERIFICATION ===")
        figures = ['fig1_calibration_c3.pdf', 'fig1_calibration_c3.png',
                  'fig2_taxonomy_contrast.pdf', 'fig2_taxonomy_contrast.png']

        for fig in figures:
            path = f'paper/figures/{fig}'
            if os.path.exists(path):
                size_kb = os.path.getsize(path) / 1024
                print(f"✓ {fig}: {size_kb:.1f} KB")
            else:
                print(f"❌ {fig}: NOT FOUND")

    except Exception as e:
        print(f"❌ Figure generation failed: {e}")
        import traceback
        traceback.print_exc()