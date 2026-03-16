"""
WS4 - CLI Tool Implementation

SSM Spectral Contracts command-line interface with red/yellow/green risk assessment.
Implements the CLI specification from §6 WS4.
"""

import click
import numpy as np
import yaml
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from metrics.trivial_baselines import compute_all_trivial_baselines
from metrics.contracts import compute_all_contracts
from metrics.approximations import compute_all_approximations


class SSMContractsEvaluator:
    """
    Core evaluator class for SSM stability contracts.
    """

    def __init__(self, thresholds_file: str = "thresholds.yaml"):
        """
        Initialize evaluator with calibrated thresholds.

        Args:
            thresholds_file: Path to threshold configuration file
        """
        self.thresholds_file = Path(__file__).parent / thresholds_file
        self.thresholds = self._load_thresholds()

    def _load_thresholds(self) -> Dict[str, Any]:
        """Load calibrated thresholds from YAML file."""
        if self.thresholds_file.exists():
            with open(self.thresholds_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default thresholds (placeholder - should be calibrated)
            return {
                'C3': {
                    'metric_id': 'C3',
                    'name': 'Pseudospectral sensitivity proxy',
                    'threshold_red': 1.2,
                    'threshold_yellow': 1.05,
                    'calibrated_on': ['s4_like'],
                    'note': 'Default thresholds - not calibrated on holdout set'
                },
                'C6': {
                    'metric_id': 'C6',
                    'name': 'Free-probability-inspired spectral spread',
                    'threshold_red': 0.8,
                    'threshold_yellow': 0.5,
                    'calibrated_on': ['s4_like'],
                    'note': 'Default thresholds - not calibrated on holdout set'
                }
            }

    def create_layer_matrices_from_config(self, config: Dict[str, Any]) -> List[np.ndarray]:
        """
        Create SSM layer matrices from configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            List of transition matrices A_l
        """
        # Extract parameters
        N = config.get('state_dimension', 64)
        L = config.get('depth', 8)
        ssm_family = config.get('ssm_family', 's4_like')
        eigenvalue_radius = config.get('eigenvalue_radius', 0.95)
        init_method = config.get('init_method', 'default')
        seed = config.get('seed', 42)

        np.random.seed(seed)

        # Generate layer matrices based on family type
        layer_matrices = []

        for layer_idx in range(L):
            if init_method == 'hippo':
                A = self._create_hippo_matrix(N, eigenvalue_radius)
            elif init_method == 'lru':
                A = self._create_lru_matrix(N, eigenvalue_radius)
            else:  # default
                A = self._create_default_matrix(N, eigenvalue_radius, ssm_family)

            layer_matrices.append(A)

        return layer_matrices

    def _create_default_matrix(self, N: int, r: float, ssm_family: str) -> np.ndarray:
        """Create default SSM matrix based on family type."""
        if ssm_family == 's4_like':
            # S4-like: diagonal with small off-diagonal coupling
            eigenvals = np.linspace(r * 0.7, r, N)
            A = np.diag(eigenvals)
            A += 0.01 * np.random.randn(N, N)
            # Rescale to maintain eigenvalue bounds
            current_max = np.max(np.abs(np.linalg.eigvals(A)))
            A = A * (r / current_max)

        elif ssm_family == 'mamba_like':
            # Mamba-like: selective with input dependence (simplified)
            A = np.diag(np.random.uniform(r * 0.8, r, N))
            A += 0.02 * np.random.randn(N, N)  # Input-dependent noise

        else:  # hybrid or unknown
            # Generic recurrence
            A = np.random.randn(N, N)
            A = A / np.max(np.abs(np.linalg.eigvals(A))) * r

        return A

    def _create_hippo_matrix(self, N: int, r: float) -> np.ndarray:
        """Create HiPPO-style matrix."""
        A = np.zeros((N, N))
        for i in range(N):
            A[i, i] = r * (2 * i + 1) / N
            for j in range(i + 1, N):
                A[i, j] = r * np.sqrt((2*i + 1) * (2*j + 1)) / N

        # Ensure spectral radius <= r
        eigvals = np.linalg.eigvals(A)
        max_eig = np.max(np.abs(eigvals))
        if max_eig > r:
            A = A * (r / max_eig)

        return A

    def _create_lru_matrix(self, N: int, r: float) -> np.ndarray:
        """Create LRU-style matrix."""
        eigenvals = r * np.exp(-np.arange(N) / (N / 3))
        return np.diag(eigenvals)

    def evaluate_configuration(self, config: Dict[str, Any],
                             use_approximations: bool = False) -> Dict[str, Any]:
        """
        Evaluate SSM configuration and compute risk assessment.

        Args:
            config: Model configuration
            use_approximations: Whether to use fast approximations

        Returns:
            Dictionary with evaluation results
        """
        # Create layer matrices
        layer_matrices = self.create_layer_matrices_from_config(config)

        # Create input matrix for controllability
        N = config.get('state_dimension', 64)
        B = np.random.randn(N, 1)

        # Sequence length for evaluation
        T = config.get('sequence_length', 1024)

        # Compute metrics
        baseline_results = compute_all_trivial_baselines(layer_matrices)

        if use_approximations:
            contract_results = compute_all_approximations(layer_matrices, B=B, T=T)
        else:
            # Use realistic T for meaningful transient amplification detection
            T_c1 = min(T, 100)  # Allow C1 to detect transient effects at meaningful timescales
            contract_results = compute_all_contracts(
                layer_matrices, B=B, T=T_c1, include_expensive=True
            )

        # Assess risk for each metric
        risk_assessments = {}
        overall_risks = []

        for metric_id, threshold_config in self.thresholds.items():
            if metric_id in contract_results:
                metric_value = contract_results[metric_id].get('value', None)
                if metric_value is not None:
                    risk_level = self._assess_risk_level(metric_value, threshold_config)
                    risk_assessments[metric_id] = {
                        'value': metric_value,
                        'risk_level': risk_level,
                        'threshold_config': threshold_config
                    }
                    overall_risks.append(risk_level)

        # Determine overall risk
        if 'RED' in overall_risks:
            overall_risk = 'RED'
        elif 'YELLOW' in overall_risks:
            overall_risk = 'YELLOW'
        else:
            overall_risk = 'GREEN'

        return {
            'config': config,
            'baseline_metrics': baseline_results,
            'contract_metrics': contract_results,
            'risk_assessments': risk_assessments,
            'overall_risk': overall_risk,
            'layer_matrices_shape': [(A.shape, np.max(np.abs(np.linalg.eigvals(A))))
                                   for A in layer_matrices[:3]]  # First 3 layers
        }

    def _assess_risk_level(self, value: float, threshold_config: Dict[str, Any]) -> str:
        """
        Assess risk level based on calibrated thresholds.

        Args:
            value: Metric value
            threshold_config: Threshold configuration

        Returns:
            Risk level: 'RED', 'YELLOW', or 'GREEN'
        """
        red_threshold = threshold_config.get('threshold_red', np.inf)
        yellow_threshold = threshold_config.get('threshold_yellow', 0.0)

        if value >= red_threshold:
            return 'RED'
        elif value >= yellow_threshold:
            return 'YELLOW'
        else:
            return 'GREEN'

    def generate_report(self, evaluation_result: Dict[str, Any]) -> str:
        """
        Generate formatted risk assessment report.

        Args:
            evaluation_result: Result from evaluate_configuration

        Returns:
            Formatted report string
        """
        config = evaluation_result['config']
        risk_assessments = evaluation_result['risk_assessments']
        overall_risk = evaluation_result['overall_risk']

        # Risk level colors (for terminal display)
        color_map = {
            'RED': '\033[91m',    # Red
            'YELLOW': '\033[93m', # Yellow
            'GREEN': '\033[92m',  # Green
            'END': '\033[0m'      # End coloring
        }

        report_lines = [
            "",
            "  SSM Spectral Contracts v1.0",
            "  " + "─" * 60,
        ]

        # Configuration summary
        report_lines.extend([
            f"  Model: {config.get('ssm_family', 'unknown')}, "
            f"N={config.get('state_dimension', 64)}, L={config.get('depth', 8)}",
            f"  Sequence length: {config.get('sequence_length', 1024)}",
            "",
            "  Contract metrics:"
        ])

        # Individual metric assessments
        for metric_id, assessment in risk_assessments.items():
            value = assessment['value']
            risk_level = assessment['risk_level']
            name = assessment['threshold_config'].get('name', metric_id)

            color_start = color_map.get(risk_level, '')
            color_end = color_map['END']

            risk_description = {
                'RED': 'HIGH RISK — training likely to fail',
                'YELLOW': 'ELEVATED RISK — monitor closely',
                'GREEN': 'WITHIN SAFE RANGE'
            }.get(risk_level, 'UNKNOWN')

            report_lines.append(
                f"    {metric_id:<4} {name:<30}: {value:8.3f}   "
                f"{color_start}[{risk_level}]{color_end} — {risk_description}"
            )

        # Overall assessment
        overall_color = color_map.get(overall_risk, '')
        overall_end = color_map['END']

        report_lines.extend([
            "",
            f"  Overall risk: {overall_color}{overall_risk}{overall_end}",
        ])

        # Risk interpretation and recommendations
        if overall_risk == 'RED':
            report_lines.extend([
                "  Predicted failure mode: High instability detected.",
                "  Recommendation: Reduce eigenvalue radius or apply regularization.",
            ])
        elif overall_risk == 'YELLOW':
            report_lines.extend([
                "  Predicted failure mode: Moderate instability risk.",
                "  Recommendation: Monitor training closely; consider checkpointing.",
            ])
        else:
            report_lines.extend([
                "  Predicted status: Training should proceed stably.",
                "  Recommendation: Configuration appears safe to train.",
            ])

        report_lines.extend([
            "  " + "─" * 60,
            ""
        ])

        return "\n".join(report_lines)


@click.group()
def cli():
    """SSM Spectral Contracts - Pre-training stability assessment tool."""
    pass


@cli.command()
@click.option('--config', '-c', required=True, type=click.Path(exists=True),
              help='Path to model configuration YAML file')
@click.option('--output', '-o', type=click.Path(),
              help='Output file for JSON results (optional)')
@click.option('--approximations', '-a', is_flag=True,
              help='Use fast approximations instead of exact metrics')
@click.option('--quiet', '-q', is_flag=True,
              help='Suppress detailed output')
def check(config: str, output: Optional[str], approximations: bool, quiet: bool):
    """
    Check SSM configuration for pre-training stability risks.

    Example:
        ssm-contracts check --config model.yaml
        ssm-contracts check --config model.yaml --approximations --output results.json
    """
    # Load configuration
    try:
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        click.echo(f"Error loading config file: {e}", err=True)
        return

    # Initialize evaluator
    try:
        evaluator = SSMContractsEvaluator()
    except Exception as e:
        click.echo(f"Error initializing evaluator: {e}", err=True)
        return

    # Evaluate configuration
    if not quiet:
        click.echo("Computing spectral contracts...")

    try:
        result = evaluator.evaluate_configuration(config_data, use_approximations=approximations)
    except Exception as e:
        click.echo(f"Error during evaluation: {e}", err=True)
        return

    # Generate and display report
    if not quiet:
        report = evaluator.generate_report(result)
        click.echo(report)

    # Save JSON output if requested
    if output:
        try:
            # Convert numpy arrays to lists for JSON serialization
            json_result = _make_json_serializable(result)
            with open(output, 'w') as f:
                json.dump(json_result, f, indent=2)
            if not quiet:
                click.echo(f"Results saved to {output}")
        except Exception as e:
            click.echo(f"Error saving output: {e}", err=True)

    # Exit with error code if high risk
    if result['overall_risk'] == 'RED':
        sys.exit(1)


@cli.command()
@click.option('--model', '-m', required=True,
              help='SSM family: s4_like, mamba_like, or hybrid')
@click.option('--N', '-n', default=64, help='State dimension')
@click.option('--L', '-l', default=8, help='Number of layers')
@click.option('--T', '-t', default=1024, help='Sequence length')
@click.option('--radius', '-r', default=0.95, help='Eigenvalue radius')
def demo(model: str, n: int, l: int, t: int, radius: float):
    """
    Run demo evaluation with specified parameters.

    Example:
        ssm-contracts demo --model s4_like --N 64 --L 8 --T 1024 --radius 0.95
    """
    # Create demo configuration
    config_data = {
        'ssm_family': model,
        'state_dimension': n,
        'depth': l,
        'sequence_length': t,
        'eigenvalue_radius': radius,
        'init_method': 'default',
        'seed': 42
    }

    # Run evaluation
    evaluator = SSMContractsEvaluator()
    result = evaluator.evaluate_configuration(config_data)

    # Display report
    report = evaluator.generate_report(result)
    click.echo(report)


@cli.command()
def validate():
    """
    Run validation tests on Case A and Case B configurations.
    """
    from benchmarks.long_memory_tasks import create_case_a_matrix, create_case_b_matrix

    click.echo("Running Case A and Case B validation tests...")

    evaluator = SSMContractsEvaluator()
    N = 32  # Smaller for faster validation

    # Test Case A: Should show high risk despite safe spectral radius
    click.echo("\n--- Case A: Hidden Instability Test ---")
    A_case_a = create_case_a_matrix(N, target_condition=500.0)
    spectral_radius_a = np.max(np.abs(np.linalg.eigvals(A_case_a)))

    config_a = {
        'ssm_family': 's4_like',
        'state_dimension': N,
        'depth': 1,  # Single layer for clear analysis
        'sequence_length': 512,
        'eigenvalue_radius': spectral_radius_a,
        'init_method': 'case_a',
        'seed': 42
    }

    # Override layer matrices to use the specific Case A matrix
    original_create = evaluator.create_layer_matrices_from_config
    evaluator.create_layer_matrices_from_config = lambda config: [A_case_a]

    result_a = evaluator.evaluate_configuration(config_a)
    click.echo(f"Spectral radius: {spectral_radius_a:.4f} (appears safe)")
    click.echo(f"Overall risk: {result_a['overall_risk']}")

    # Debug: show individual risk assessments for Case A
    click.echo("  Individual assessments:")
    for metric_id, assessment in result_a['risk_assessments'].items():
        click.echo(f"    {metric_id}: {assessment['value']:.3f} → {assessment['risk_level']}")

    if result_a['overall_risk'] in ['RED', 'YELLOW']:
        click.echo("✓ Case A correctly identified as risky")
    else:
        click.echo("⚠ Case A not identified as risky - may need threshold adjustment")

    # Test Case B: Should show low risk despite high spectral radius
    click.echo("\n--- Case B: Apparent Risk Test ---")
    A_case_b = create_case_b_matrix(N, max_eigenvalue=0.999)
    spectral_radius_b = np.max(np.abs(np.linalg.eigvals(A_case_b)))

    config_b = {
        'ssm_family': 's4_like',
        'state_dimension': N,
        'depth': 1,
        'sequence_length': 50,  # Reduced T to avoid 0.95^T underflow
        'eigenvalue_radius': spectral_radius_b,
        'init_method': 'case_b',
        'seed': 42
    }

    evaluator.create_layer_matrices_from_config = lambda config: [A_case_b]

    result_b = evaluator.evaluate_configuration(config_b)
    click.echo(f"Spectral radius: {spectral_radius_b:.4f} (appears risky)")
    click.echo(f"Overall risk: {result_b['overall_risk']}")

    # Debug: show individual risk assessments
    click.echo("  Individual assessments:")
    for metric_id, assessment in result_b['risk_assessments'].items():
        click.echo(f"    {metric_id}: {assessment['value']:.3f} → {assessment['risk_level']}")

    if result_b['overall_risk'] == 'GREEN':
        click.echo("✓ Case B correctly identified as safe")
    else:
        click.echo("⚠ Case B identified as risky - contracts may be conservative")

    # Restore original method
    evaluator.create_layer_matrices_from_config = original_create

    click.echo("\nValidation complete!")


def _make_json_serializable(obj):
    """Convert numpy arrays and other objects to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    else:
        return obj


if __name__ == '__main__':
    cli()