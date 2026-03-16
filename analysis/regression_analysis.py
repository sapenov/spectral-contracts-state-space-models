"""
WS3 - Predictiveness Study: Correlation and Regression Analysis

Implements statistical analysis of contract metric predictiveness per §6 WS3.
Computes Spearman correlations, AUROC, calibration curves, and generalization tests.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional dependency
from typing import Dict, List, Any, Tuple, Optional
import warnings


class PredictivenesStudy:
    """
    Analyzes predictive power of spectral contract metrics.
    """

    def __init__(self, results_data: List[Dict[str, Any]]):
        """
        Initialize with benchmark results.

        Args:
            results_data: List of experiment results from run_sweeps.py
        """
        self.results_data = results_data
        self.df = self._flatten_to_dataframe(results_data)

        # Storage for analysis results
        self.correlation_results = {}
        self.auroc_results = {}
        self.calibration_results = {}

    def _flatten_to_dataframe(self, results_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert nested results to flat DataFrame for analysis.

        Args:
            results_data: Raw benchmark results

        Returns:
            Flattened DataFrame with metrics and outcomes
        """
        rows = []

        for result in results_data:
            # Base configuration info
            base_row = {
                'config_id': result['config_id'],
                'sweep_type': result.get('sweep_type', 'unknown'),
                'ssm_family': result['ssm_family'],
                'N': result['N'],
                'L': result['L'],
                'eigenvalue_radius': result.get('eigenvalue_radius',
                                               result.get('r', 0.95)),
                'init_method': result['init_method'],
                'seed': result['seed']
            }

            # Add trivial baseline metrics
            baselines = result.get('baseline_metrics', {})
            for metric_name, metric_data in baselines.items():
                if isinstance(metric_data, dict) and 'value' in metric_data:
                    if metric_data['value'] is not None:
                        base_row[f'trivial_{metric_name}'] = metric_data['value']

            # Add contract metrics
            contracts = result.get('exact_contracts', {})
            for metric_name, metric_data in contracts.items():
                if isinstance(metric_data, dict) and 'value' in metric_data:
                    base_row[f'contract_{metric_name}'] = metric_data['value']

            # Add approximate contract metrics
            approx_contracts = result.get('approx_contracts', {})
            for metric_name, metric_data in approx_contracts.items():
                if isinstance(metric_data, dict) and 'value' in metric_data:
                    base_row[f'approx_{metric_name}'] = metric_data['value']

            # Add outcomes
            outcomes = result.get('outcomes', {})
            for outcome_name, outcome_value in outcomes.items():
                base_row[f'outcome_{outcome_name}'] = outcome_value

            rows.append(base_row)

        return pd.DataFrame(rows)

    def compute_correlations(self, metric_columns: Optional[List[str]] = None,
                           outcome_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compute Spearman correlations between metrics and outcomes.

        Args:
            metric_columns: List of metric column names (auto-detect if None)
            outcome_columns: List of outcome column names (auto-detect if None)

        Returns:
            Dictionary with correlation results
        """
        if metric_columns is None:
            # Auto-detect metric columns
            metric_columns = [col for col in self.df.columns
                            if col.startswith(('trivial_', 'contract_', 'approx_'))]

        if outcome_columns is None:
            # Auto-detect outcome columns
            outcome_columns = [col for col in self.df.columns
                             if col.startswith('outcome_')]

        print(f"Computing correlations for {len(metric_columns)} metrics "
              f"and {len(outcome_columns)} outcomes...")

        correlation_matrix = {}
        p_value_matrix = {}

        for metric_col in metric_columns:
            correlation_matrix[metric_col] = {}
            p_value_matrix[metric_col] = {}

            for outcome_col in outcome_columns:
                # Get non-null values for both metric and outcome
                metric_values = self.df[metric_col].dropna()
                outcome_values = self.df[outcome_col].dropna()

                # Find common indices
                common_indices = metric_values.index.intersection(outcome_values.index)

                if len(common_indices) > 3:  # Need at least 4 points
                    metric_common = metric_values.loc[common_indices]
                    outcome_common = outcome_values.loc[common_indices]

                    try:
                        # Compute Spearman correlation
                        rho, p_val = spearmanr(metric_common, outcome_common)
                        correlation_matrix[metric_col][outcome_col] = rho
                        p_value_matrix[metric_col][outcome_col] = p_val
                    except Exception as e:
                        print(f"Warning: Could not compute correlation for "
                              f"{metric_col} vs {outcome_col}: {e}")
                        correlation_matrix[metric_col][outcome_col] = np.nan
                        p_value_matrix[metric_col][outcome_col] = np.nan
                else:
                    correlation_matrix[metric_col][outcome_col] = np.nan
                    p_value_matrix[metric_col][outcome_col] = np.nan

        self.correlation_results = {
            'correlations': correlation_matrix,
            'p_values': p_value_matrix,
            'metric_columns': metric_columns,
            'outcome_columns': outcome_columns
        }

        return self.correlation_results

    def compute_delta_spearman(self, trivial_baseline_prefix: str = 'trivial_') -> Dict[str, Any]:
        """
        Compute ΔSpearman ρ over best trivial baseline per SC-1 criterion.

        Args:
            trivial_baseline_prefix: Prefix for trivial baseline columns

        Returns:
            Dictionary with ΔSpearman results
        """
        if not self.correlation_results:
            self.compute_correlations()

        correlations = self.correlation_results['correlations']
        outcome_columns = self.correlation_results['outcome_columns']

        # Identify trivial baseline metrics
        trivial_metrics = [col for col in correlations.keys()
                          if col.startswith(trivial_baseline_prefix)]

        # Identify contract metrics
        contract_metrics = [col for col in correlations.keys()
                           if col.startswith('contract_')]

        delta_spearman_results = {}

        for outcome_col in outcome_columns:
            delta_spearman_results[outcome_col] = {}

            # Find best trivial baseline for this outcome
            trivial_correlations = []
            for trivial_metric in trivial_metrics:
                if outcome_col in correlations[trivial_metric]:
                    rho = correlations[trivial_metric][outcome_col]
                    if not np.isnan(rho):
                        trivial_correlations.append(abs(rho))  # Use absolute value

            if len(trivial_correlations) > 0:
                best_trivial_rho = max(trivial_correlations)
            else:
                best_trivial_rho = 0.0

            # Compute ΔSpearman for each contract metric
            for contract_metric in contract_metrics:
                if outcome_col in correlations[contract_metric]:
                    contract_rho = correlations[contract_metric][outcome_col]
                    if not np.isnan(contract_rho):
                        delta_rho = abs(contract_rho) - best_trivial_rho
                        delta_spearman_results[outcome_col][contract_metric] = {
                            'contract_rho': contract_rho,
                            'best_trivial_rho': best_trivial_rho,
                            'delta_spearman': delta_rho,
                            'meets_sc1': delta_rho >= 0.10  # SC-1 threshold
                        }

        return delta_spearman_results

    def compute_auroc_analysis(self, binary_outcomes: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compute AUROC for binary outcome prediction per SC-2 criterion.

        Args:
            binary_outcomes: List of binary outcome column names

        Returns:
            Dictionary with AUROC results
        """
        if binary_outcomes is None:
            # Auto-detect binary outcomes (boolean or 0/1 values)
            binary_outcomes = []
            for col in self.df.columns:
                if col.startswith('outcome_'):
                    unique_vals = self.df[col].dropna().unique()
                    if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, True, False}):
                        binary_outcomes.append(col)

        print(f"Computing AUROC for binary outcomes: {binary_outcomes}")

        metric_columns = [col for col in self.df.columns
                         if col.startswith(('trivial_', 'contract_', 'approx_'))]

        auroc_results = {}

        for outcome_col in binary_outcomes:
            auroc_results[outcome_col] = {}

            # Get outcome values
            outcome_values = self.df[outcome_col].dropna()

            for metric_col in metric_columns:
                metric_values = self.df[metric_col].dropna()

                # Find common indices
                common_indices = metric_values.index.intersection(outcome_values.index)

                if len(common_indices) > 3:
                    metric_common = metric_values.loc[common_indices]
                    outcome_common = outcome_values.loc[common_indices]

                    # Ensure binary outcomes are 0/1
                    outcome_binary = outcome_common.astype(int)

                    # Check if we have both classes
                    if len(outcome_binary.unique()) == 2:
                        try:
                            auroc = roc_auc_score(outcome_binary, metric_common)
                            auroc_results[outcome_col][metric_col] = {
                                'auroc': auroc,
                                'n_samples': len(common_indices),
                                'meets_sc2': auroc >= 0.75  # SC-2 threshold
                            }
                        except Exception as e:
                            print(f"Warning: Could not compute AUROC for "
                                  f"{metric_col} vs {outcome_col}: {e}")

        self.auroc_results = auroc_results
        return auroc_results

    def compute_calibration_analysis(self, binary_outcomes: Optional[List[str]] = None,
                                   n_bins: int = 5) -> Dict[str, Any]:
        """
        Compute calibration curves for risk assessment.

        Args:
            binary_outcomes: List of binary outcome column names
            n_bins: Number of bins for calibration curve

        Returns:
            Dictionary with calibration results
        """
        if binary_outcomes is None:
            # Use same auto-detection as AUROC
            binary_outcomes = []
            for col in self.df.columns:
                if col.startswith('outcome_'):
                    unique_vals = self.df[col].dropna().unique()
                    if len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, True, False}):
                        binary_outcomes.append(col)

        metric_columns = [col for col in self.df.columns
                         if col.startswith(('trivial_', 'contract_'))]

        calibration_results = {}

        for outcome_col in binary_outcomes:
            calibration_results[outcome_col] = {}

            outcome_values = self.df[outcome_col].dropna()

            for metric_col in metric_columns:
                metric_values = self.df[metric_col].dropna()
                common_indices = metric_values.index.intersection(outcome_values.index)

                if len(common_indices) >= n_bins * 2:  # Need enough samples per bin
                    metric_common = metric_values.loc[common_indices]
                    outcome_common = outcome_values.loc[common_indices].astype(int)

                    if len(outcome_common.unique()) == 2:
                        try:
                            # Normalize metric to [0,1] for calibration
                            metric_normalized = (metric_common - metric_common.min()) / (
                                metric_common.max() - metric_common.min() + 1e-8)

                            frac_positive, mean_pred = calibration_curve(
                                outcome_common, metric_normalized, n_bins=n_bins)

                            # Compute calibration error (reliability)
                            calibration_error = np.mean(np.abs(frac_positive - mean_pred))

                            calibration_results[outcome_col][metric_col] = {
                                'frac_positive': frac_positive,
                                'mean_predicted': mean_pred,
                                'calibration_error': calibration_error,
                                'well_calibrated': calibration_error < 0.1,  # Threshold
                                'n_bins': n_bins
                            }
                        except Exception as e:
                            print(f"Warning: Could not compute calibration for "
                                  f"{metric_col} vs {outcome_col}: {e}")

        self.calibration_results = calibration_results
        return calibration_results

    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate the main results table per §10.6 schema.

        Returns:
            DataFrame with summary of all metric performance
        """
        if not self.correlation_results:
            self.compute_correlations()
        if not self.auroc_results:
            self.compute_auroc_analysis()

        delta_spearman = self.compute_delta_spearman()

        # Build summary table
        rows = []

        metric_columns = self.correlation_results['metric_columns']
        for metric_col in metric_columns:
            # Skip if no meaningful data
            if metric_col not in self.correlation_results['correlations']:
                continue

            # Focus on training divergence as primary outcome
            primary_outcome = 'outcome_training_diverged'

            base_row = {
                'Metric': metric_col,
                'Type': self._classify_metric_type(metric_col)
            }

            # Spearman correlation
            if (primary_outcome in self.correlation_results['correlations'][metric_col]):
                rho = self.correlation_results['correlations'][metric_col][primary_outcome]
                p_val = self.correlation_results['p_values'][metric_col][primary_outcome]
                base_row['Spearman_rho'] = rho
                base_row['p_value'] = p_val
            else:
                base_row['Spearman_rho'] = np.nan
                base_row['p_value'] = np.nan

            # ΔSpearman
            if (primary_outcome in delta_spearman and
                metric_col in delta_spearman[primary_outcome]):
                base_row['Delta_Spearman'] = delta_spearman[primary_outcome][metric_col]['delta_spearman']
                base_row['Meets_SC1'] = delta_spearman[primary_outcome][metric_col]['meets_sc1']
            else:
                base_row['Delta_Spearman'] = np.nan
                base_row['Meets_SC1'] = False

            # AUROC
            if (primary_outcome in self.auroc_results and
                metric_col in self.auroc_results[primary_outcome]):
                auroc_data = self.auroc_results[primary_outcome][metric_col]
                base_row['AUROC'] = auroc_data['auroc']
                base_row['Meets_SC2'] = auroc_data['meets_sc2']
            else:
                base_row['AUROC'] = np.nan
                base_row['Meets_SC2'] = False

            # Calibration status
            if (primary_outcome in self.calibration_results and
                metric_col in self.calibration_results[primary_outcome]):
                cal_data = self.calibration_results[primary_outcome][metric_col]
                base_row['Calibrated'] = cal_data['well_calibrated']
                base_row['Calibration_Error'] = cal_data['calibration_error']
            else:
                base_row['Calibrated'] = False
                base_row['Calibration_Error'] = np.nan

            # Overall classification
            base_row['Status'] = self._classify_metric_status(base_row)

            rows.append(base_row)

        summary_df = pd.DataFrame(rows)
        return summary_df

    def _classify_metric_type(self, metric_name: str) -> str:
        """Classify metric into type for summary table."""
        if metric_name.startswith('trivial_'):
            return 'TRIVIAL'
        elif metric_name.startswith('contract_'):
            return 'CONTRACT'
        elif metric_name.startswith('approx_'):
            return 'APPROXIMATE'
        else:
            return 'UNKNOWN'

    def _classify_metric_status(self, metric_row: Dict[str, Any]) -> str:
        """Classify metric status based on performance criteria."""
        spearman_good = not pd.isna(metric_row['Spearman_rho']) and abs(metric_row['Spearman_rho']) >= 0.6
        auroc_good = not pd.isna(metric_row['AUROC']) and metric_row['AUROC'] >= 0.75
        calibrated = metric_row['Calibrated']

        if spearman_good and auroc_good and calibrated:
            return 'FULL_CONTRACT'
        elif spearman_good and not auroc_good:
            return 'RANK_ONLY'
        elif not spearman_good and auroc_good:
            return 'THRESHOLD_ONLY'
        else:
            return 'UNINFORMATIVE'

    def plot_correlation_heatmap(self, save_path: Optional[str] = None):
        """Plot heatmap of metric-outcome correlations."""
        if not self.correlation_results:
            self.compute_correlations()

        correlations = self.correlation_results['correlations']

        # Convert to DataFrame for plotting
        corr_df = pd.DataFrame(correlations).T

        # Simple heatmap without seaborn
        plt.figure(figsize=(12, 8))
        im = plt.imshow(corr_df.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        plt.colorbar(im, label='Spearman ρ')

        # Add text annotations
        for i in range(len(corr_df.index)):
            for j in range(len(corr_df.columns)):
                value = corr_df.iloc[i, j]
                if not np.isnan(value):
                    plt.text(j, i, f'{value:.3f}', ha='center', va='center',
                            color='white' if abs(value) > 0.5 else 'black')

        plt.title('Metric-Outcome Correlations')
        plt.xlabel('Outcomes')
        plt.ylabel('Metrics')
        plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha='right')
        plt.yticks(range(len(corr_df.index)), corr_df.index, rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation heatmap saved to {save_path}")
        else:
            plt.show()

    def print_analysis_summary(self):
        """Print comprehensive analysis summary."""
        print("="*60)
        print("SPECTRAL CONTRACTS PREDICTIVENESS STUDY - WS3 RESULTS")
        print("="*60)

        # Basic dataset info
        print(f"Dataset size: {len(self.df)} configurations")
        print(f"SSM families: {sorted(self.df['ssm_family'].unique())}")
        print(f"Unique outcomes analyzed: {len([c for c in self.df.columns if c.startswith('outcome_')])}")

        # Correlation analysis
        if self.correlation_results:
            print(f"\nCorrelation analysis completed:")
            print(f"  Metrics analyzed: {len(self.correlation_results['metric_columns'])}")
            print(f"  Outcomes analyzed: {len(self.correlation_results['outcome_columns'])}")

        # Summary table
        summary_df = self.generate_summary_table()
        print(f"\n{'Metric Performance Summary':<50}")
        print("-"*50)

        for _, row in summary_df.iterrows():
            metric_name = row['Metric'].replace('contract_', '').replace('trivial_', '')
            print(f"{metric_name:<25} | ρ={row['Spearman_rho']:.3f} | AUROC={row['AUROC']:.3f} | {row['Status']}")

        # SC-1 and SC-2 criteria check
        sc1_passing = summary_df[summary_df['Meets_SC1'] == True]
        sc2_passing = summary_df[summary_df['Meets_SC2'] == True]

        print(f"\nSuccess Criteria Analysis:")
        print(f"  SC-1 (ΔSpearman ≥ 0.10): {len(sc1_passing)} metrics pass")
        print(f"  SC-2 (AUROC ≥ 0.75): {len(sc2_passing)} metrics pass")

        if len(sc1_passing) > 0:
            print("  ✓ Strong claim threshold achievable (SC-1 satisfied)")
        else:
            print("  ⚠ Strong claim threshold not met - consider survival paths")

        return summary_df


def demo_predictiveness_analysis():
    """
    Demo the predictiveness analysis on synthetic data.
    """
    print("Generating demo data for predictiveness analysis...")

    # Generate synthetic benchmark results
    demo_results = []
    np.random.seed(42)

    for i in range(20):
        # Create configuration
        config = {
            'config_id': f'DEMO{i:03d}',
            'sweep_type': 'demo',
            'ssm_family': 's4_like',
            'N': 64,
            'L': 8,
            'eigenvalue_radius': 0.7 + 0.4 * np.random.random(),
            'init_method': 'default',
            'seed': i
        }

        # Generate synthetic metric values
        eigenval_radius = config['eigenvalue_radius']

        baseline_metrics = {
            'max_eigenvalue': {'value': eigenval_radius + np.random.normal(0, 0.05)},
            'max_operator_norm': {'value': eigenval_radius + np.random.normal(0, 0.03)},
            'initial_gradient': {'value': np.random.uniform(0.1, 2.0)}
        }

        exact_contracts = {
            'C1': {'value': max(1.0, 10**(eigenval_radius * 3 + np.random.normal(0, 0.5)))},
            'C2': {'value': 1.0 + eigenval_radius * 2 + np.random.normal(0, 0.3)},
            'C3': {'value': eigenval_radius + max(0, np.random.normal(eigenval_radius - 0.9, 0.1))},
            'C6': {'value': eigenval_radius * 0.5 + np.random.normal(0, 0.1)}
        }

        # Compute non-circular training outcomes via actual dynamics
        # Create simple SSM matrices for this configuration
        layer_matrices = []
        np.random.seed(i)  # Ensure reproducibility per configuration
        for l in range(3):  # Small number of layers for demo
            A = np.diag(np.linspace(eigenval_radius * 0.8, eigenval_radius, 32))
            # Add small off-diagonal for non-triviality
            A += 0.02 * np.random.randn(32, 32)
            # Rescale to maintain eigenvalue bounds
            current_max = np.max(np.abs(np.linalg.eigvals(A)))
            if current_max > 0:
                A = A * (eigenval_radius / current_max)
            layer_matrices.append(A)

        # Import the non-circular outcome functions
        import sys
        sys.path.insert(0, '.')
        from benchmarks.long_memory_tasks import compute_linear_stability_outcome, compute_memory_retention_outcome

        # Compute legitimate outcomes
        stability_result = compute_linear_stability_outcome(layer_matrices, T_test=200, n_trials=3)
        memory_result = compute_memory_retention_outcome(layer_matrices, T_test=200)

        outcomes = {
            'training_diverged': stability_result['diverged'],
            'growth_ratio': stability_result['growth_ratio'],
            'log_growth': stability_result['log_growth'],
            'memory_retention': memory_result['memory_retention'],
            'log_retention': memory_result['log_retention'],
            'gradient_exploded': stability_result['growth_ratio'] > 100.0
        }

        demo_results.append({
            **config,
            'baseline_metrics': baseline_metrics,
            'exact_contracts': exact_contracts,
            'outcomes': outcomes
        })

    # Run predictiveness analysis
    study = PredictivenesStudy(demo_results)

    print(f"\nDemo dataset created: {len(demo_results)} configurations")
    print(f"Training diverged: {sum(r['outcomes']['training_diverged'] for r in demo_results)} configurations")

    # Run analysis
    study.compute_correlations()
    study.compute_auroc_analysis()
    study.compute_calibration_analysis()

    # Print results
    summary_df = study.print_analysis_summary()

    return study, summary_df


if __name__ == "__main__":
    # Run demo analysis
    demo_predictiveness_analysis()