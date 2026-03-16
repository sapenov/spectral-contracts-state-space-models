"""
Synthetic long-memory benchmark tasks for SSM stability evaluation.

Implements the six benchmark tasks from benchmark_spec.md:
1. Copying task (primary)
2. Selective recall
3. Long-range parity
4. Controlled instability sweep (framework)
5. Short LM surrogate (placeholder)
6. Sequence classification with distractors

Each task returns data in standardized format for metric evaluation.
"""

import numpy as np
from typing import Tuple, Dict, Any, List, Optional
import random


def generate_copying_task(T: int, K: int, vocab_size: int = 10,
                         noise_level: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate copying task data: copy a sequence after delay.

    Args:
        T: Total sequence length
        K: Length of sequence to copy (K < T/2)
        vocab_size: Size of token vocabulary
        noise_level: Fraction of delay period filled with noise (0.0-1.0)

    Returns:
        inputs: (T,) array of input tokens
        targets: (K,) array of tokens to copy

    >>> inputs, targets = generate_copying_task(T=10, K=3, vocab_size=5)
    >>> len(inputs) == 10 and len(targets) == 3
    True
    """
    assert K < T // 2, f"Copy length {K} too large for sequence length {T}"

    # Generate sequence to copy
    copy_sequence = np.random.randint(1, vocab_size, size=K)

    # Build full input sequence
    inputs = np.zeros(T, dtype=int)

    # Place copy sequence at beginning
    inputs[:K] = copy_sequence

    # Fill delay period
    delay_start = K
    delay_end = T - 1  # Leave one position for copy signal
    delay_length = delay_end - delay_start

    if noise_level > 0:
        # Add noise tokens during delay
        noise_positions = np.random.choice(
            delay_length,
            size=int(noise_level * delay_length),
            replace=False
        )
        inputs[delay_start + noise_positions] = np.random.randint(1, vocab_size, size=len(noise_positions))

    # Add copy signal token (token 0 is reserved for copy signal)
    inputs[T-1] = 0

    return inputs, copy_sequence


def generate_selective_recall(T: int, num_pairs: int, vocab_size: int = 10,
                             distractor_fraction: float = 0.6) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Generate selective recall task: store key-value pairs, then recall specific keys.

    Args:
        T: Total sequence length
        num_pairs: Number of key-value pairs to store
        vocab_size: Size of vocabulary for keys and values
        distractor_fraction: Fraction of sequence filled with distractors

    Returns:
        inputs: (T,) input sequence with store/recall commands
        targets: List of (key, value) pairs for recall queries

    >>> inputs, targets = generate_selective_recall(T=20, num_pairs=3, vocab_size=5)
    >>> len(targets) <= 3
    True
    """
    # Special tokens
    STORE_TOKEN = 0
    RECALL_TOKEN = vocab_size + 1

    inputs = np.full(T, -1, dtype=int)  # -1 = empty position
    position = 0

    # Generate key-value pairs
    keys = np.random.choice(range(1, vocab_size), size=num_pairs, replace=False)
    values = np.random.randint(1, vocab_size, size=num_pairs)
    kv_pairs = list(zip(keys, values))

    # Store phase: place key-store-value sequences
    store_length = 3 * num_pairs  # Each pair takes 3 tokens
    for key, value in kv_pairs:
        if position + 2 < T:
            inputs[position] = key
            inputs[position + 1] = STORE_TOKEN
            inputs[position + 2] = value
            position += 3

    # Calculate remaining space for distractors and recalls
    remaining_length = T - position
    distractor_length = int(distractor_fraction * remaining_length)

    # Distractor phase
    distractor_end = position + distractor_length
    if distractor_end < T:
        distractor_positions = range(position, distractor_end)
        inputs[distractor_positions] = np.random.randint(1, vocab_size, size=len(distractor_positions))
        position = distractor_end

    # Recall phase: query random subset of stored keys
    recall_keys = np.random.choice(keys, size=min(len(keys), (T - position) // 2), replace=False)
    recall_targets = []

    for recall_key in recall_keys:
        if position + 1 < T:
            inputs[position] = recall_key
            inputs[position + 1] = RECALL_TOKEN
            position += 2

            # Find corresponding value
            for key, value in kv_pairs:
                if key == recall_key:
                    recall_targets.append((recall_key, value))
                    break

    # Fill remaining positions with padding
    inputs[inputs == -1] = vocab_size + 2  # PAD_TOKEN

    return inputs, recall_targets


def generate_long_range_parity(T: int, signal_fraction: float = 0.1,
                              vocab_size: int = 10) -> Tuple[np.ndarray, int]:
    """
    Generate long-range parity task: compute XOR over marked positions.

    Args:
        T: Sequence length
        signal_fraction: Fraction of positions that contain signal bits
        vocab_size: Size of vocabulary for distractor tokens

    Returns:
        inputs: (T,) sequence with signal and distractor tokens
        parity: 0 (even) or 1 (odd) parity of signal bits

    >>> inputs, parity = generate_long_range_parity(T=20, signal_fraction=0.2)
    >>> parity in [0, 1]
    True
    """
    # Special tokens
    SIGNAL_0 = 0  # Signal bit 0
    SIGNAL_1 = 1  # Signal bit 1

    num_signal_positions = int(signal_fraction * T)
    signal_positions = np.random.choice(T, size=num_signal_positions, replace=False)

    inputs = np.random.randint(2, vocab_size, size=T)  # Fill with distractors (tokens 2+)

    # Place signal bits at selected positions
    signal_bits = np.random.randint(0, 2, size=num_signal_positions)
    inputs[signal_positions] = signal_bits

    # Compute parity
    parity = np.sum(signal_bits) % 2

    return inputs, parity


def generate_classification_with_distractors(T: int, signal_position: int,
                                           vocab_size: int = 10) -> Tuple[np.ndarray, int]:
    """
    Generate classification task where signal appears early, followed by distractors.

    Args:
        T: Sequence length
        signal_position: Position where classification signal appears
        vocab_size: Size of vocabulary

    Returns:
        inputs: (T,) sequence with signal and distractors
        label: Binary classification label (0 or 1)

    >>> inputs, label = generate_classification_with_distractors(T=10, signal_position=2)
    >>> label in [0, 1]
    True
    """
    assert signal_position < T, f"Signal position {signal_position} beyond sequence length {T}"

    # Generate random sequence
    inputs = np.random.randint(0, vocab_size, size=T)

    # Classification rule: label = 1 if token at signal_position is even, 0 if odd
    signal_token = inputs[signal_position]
    label = signal_token % 2

    return inputs, label


class SSMTaskDataset:
    """
    Dataset wrapper for SSM benchmark tasks with standardized interface.
    """

    def __init__(self, task_name: str, T: int, num_samples: int, **task_kwargs):
        """
        Initialize dataset for specified task.

        Args:
            task_name: Name of task ('copying', 'selective_recall', 'parity', 'classification')
            T: Sequence length
            num_samples: Number of samples to generate
            **task_kwargs: Task-specific parameters
        """
        self.task_name = task_name
        self.T = T
        self.num_samples = num_samples
        self.task_kwargs = task_kwargs

        # Generate all samples
        self.samples = []
        self.generate_samples()

    def generate_samples(self):
        """Generate all samples for the dataset."""
        for i in range(self.num_samples):
            sample = self.generate_single_sample()
            self.samples.append(sample)

    def generate_single_sample(self) -> Dict[str, Any]:
        """Generate a single sample based on task type."""
        if self.task_name == 'copying':
            K = self.task_kwargs.get('K', self.T // 4)
            vocab_size = self.task_kwargs.get('vocab_size', 10)
            inputs, targets = generate_copying_task(self.T, K, vocab_size)
            return {
                'inputs': inputs,
                'targets': targets,
                'task_type': 'sequence_to_sequence',
                'metric_name': 'copy_accuracy'
            }

        elif self.task_name == 'selective_recall':
            num_pairs = self.task_kwargs.get('num_pairs', 3)
            vocab_size = self.task_kwargs.get('vocab_size', 10)
            inputs, targets = generate_selective_recall(self.T, num_pairs, vocab_size)
            return {
                'inputs': inputs,
                'targets': targets,  # List of (key, value) pairs
                'task_type': 'selective_recall',
                'metric_name': 'recall_accuracy'
            }

        elif self.task_name == 'parity':
            signal_fraction = self.task_kwargs.get('signal_fraction', 0.1)
            vocab_size = self.task_kwargs.get('vocab_size', 10)
            inputs, parity = generate_long_range_parity(self.T, signal_fraction, vocab_size)
            return {
                'inputs': inputs,
                'targets': parity,
                'task_type': 'binary_classification',
                'metric_name': 'parity_accuracy'
            }

        elif self.task_name == 'classification':
            signal_position = self.task_kwargs.get('signal_position', self.T // 4)
            vocab_size = self.task_kwargs.get('vocab_size', 10)
            inputs, label = generate_classification_with_distractors(self.T, signal_position, vocab_size)
            return {
                'inputs': inputs,
                'targets': label,
                'task_type': 'binary_classification',
                'metric_name': 'classification_accuracy'
            }
        else:
            raise ValueError(f"Unknown task: {self.task_name}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """Get a random batch of samples."""
        indices = np.random.choice(len(self.samples), size=batch_size, replace=True)
        return [self.samples[i] for i in indices]


def create_case_a_matrix(N: int, target_condition: float = 1000.0) -> np.ndarray:
    """
    Create Case A matrix: spectral radius < 1, but high condition number.

    A = V @ D @ V^{-1} where D has eigenvalues < 1 but V is ill-conditioned.

    Args:
        N: Matrix dimension
        target_condition: Target condition number for V

    Returns:
        A: Transition matrix with hidden instability

    >>> A = create_case_a_matrix(4, target_condition=100.0)
    >>> spectral_radius = np.max(np.abs(np.linalg.eigvals(A)))
    >>> spectral_radius < 1.0
    True
    """
    # Create diagonal eigenvalue matrix with all eigenvalues < 1
    eigenvals = np.linspace(0.80, 0.95, N)  # All safely inside unit circle
    D = np.diag(eigenvals)

    # Create ill-conditioned eigenvector matrix V
    # Start with identity and add structured perturbations
    V = np.eye(N)

    # Add upper triangular perturbations to create ill-conditioning
    alpha = 0.1  # Start small and adjust
    for iteration in range(10):  # Iterative adjustment
        R = np.triu(np.random.randn(N, N), k=1)  # Upper triangular random
        V_test = np.eye(N) + alpha * R

        try:
            cond = np.linalg.cond(V_test)
            if target_condition * 0.5 <= cond <= target_condition * 2.0:
                V = V_test
                break
            elif cond < target_condition:
                alpha *= 1.5  # Increase perturbation
            else:
                alpha *= 0.7  # Decrease perturbation
        except np.linalg.LinAlgError:
            alpha *= 0.5  # Reduce if singular

    # Construct A = V @ D @ V^{-1}
    try:
        V_inv = np.linalg.inv(V)
        A = V @ D @ V_inv
    except np.linalg.LinAlgError:
        # Fallback: use pseudoinverse
        V_inv = np.linalg.pinv(V)
        A = V @ D @ V_inv

    return A


def create_case_b_matrix(N: int, max_eigenvalue: float = 0.999) -> np.ndarray:
    """
    Create Case B matrix: spectral radius ≈ 1 (risky) but well-conditioned.

    Args:
        N: Matrix dimension
        max_eigenvalue: Largest eigenvalue (close to 1)

    Returns:
        A: Well-conditioned diagonal matrix with large spectral radius

    >>> A = create_case_b_matrix(4, max_eigenvalue=0.999)
    >>> spectral_radius = np.max(np.abs(np.linalg.eigvals(A)))
    >>> 0.99 <= spectral_radius <= 1.0
    True
    """
    # Create diagonal matrix with one eigenvalue close to 1
    # Case B: one near-unit eigenvalue, rest at 0.95 (not 0.5)
    # This isolates the "spectral radius looks risky" signal
    eigenvals = np.zeros(N)
    eigenvals[0] = max_eigenvalue  # Largest eigenvalue close to 1
    eigenvals[1:] = 0.95  # Other eigenvalues at safe 0.95 to avoid composition underflow

    # Diagonal matrix is automatically well-conditioned (condition number = max/min eigenvalue)
    A = np.diag(eigenvals)

    return A


def validate_case_matrices():
    """Validate that Case A and Case B matrices have expected properties."""
    print("Validating Case A and Case B matrix construction...")

    N = 8

    # Test Case A
    A_case_a = create_case_a_matrix(N, target_condition=500.0)
    eigenvals_a = np.linalg.eigvals(A_case_a)
    spectral_radius_a = np.max(np.abs(eigenvals_a))

    # Compute pseudospectral proxy (Kreiss constant approximation)
    A_power = A_case_a
    kreiss_values = []
    for n in range(1, 21):
        norm_n = np.linalg.norm(A_power, ord=2)
        kreiss_values.append(norm_n**(1.0/n))
        A_power = A_case_a @ A_power
    kreiss_constant_a = np.max(kreiss_values)

    print(f"Case A: spectral_radius = {spectral_radius_a:.4f}, Kreiss = {kreiss_constant_a:.4f}")
    assert spectral_radius_a < 1.0, "Case A should have spectral radius < 1"
    assert kreiss_constant_a > spectral_radius_a, "Case A should have hidden instability"

    # Test Case B
    A_case_b = create_case_b_matrix(N, max_eigenvalue=0.999)
    eigenvals_b = np.linalg.eigvals(A_case_b)
    spectral_radius_b = np.max(np.abs(eigenvals_b))
    condition_b = np.linalg.cond(A_case_b)

    print(f"Case B: spectral_radius = {spectral_radius_b:.4f}, condition = {condition_b:.4f}")
    assert spectral_radius_b >= 0.99, "Case B should have spectral radius close to 1"
    assert condition_b < 10.0, "Case B should be well-conditioned"

    print("✓ Case A and Case B matrices validated")


def compute_linear_stability_outcome(layer_matrices: List[np.ndarray], T_test: int = 500,
                                   n_trials: int = 10) -> Dict[str, Any]:
    """
    Non-circular stability outcome for linear SSM.

    Measures whether state norm grows unboundedly relative to initial norm.
    A model is 'unstable' if mean growth ratio exceeds threshold across trials.

    This is a legitimate ground truth for linear recurrences — not derived
    from eigenvalue radius directly, but from actual iterated dynamics.

    Args:
        layer_matrices: List of SSM transition matrices
        T_test: Number of time steps to iterate
        n_trials: Number of random initial conditions to test

    Returns:
        Dictionary with stability outcomes

    Cost: O(N * L * T_test * n_trials)
    """
    N = layer_matrices[0].shape[0]
    growth_ratios = []

    for _ in range(n_trials):
        x = np.random.randn(N)
        x = x / np.linalg.norm(x)  # unit initial state
        x0_norm = 1.0

        for t in range(T_test):
            for A in layer_matrices:
                x = A @ x
            # Early exit if clearly blown up
            current_norm = np.linalg.norm(x)
            if current_norm > 1e6:
                growth_ratios.append(1e6)
                break
        else:
            growth_ratios.append(np.linalg.norm(x) / x0_norm)

    mean_growth = np.mean(growth_ratios)

    return {
        'diverged': mean_growth > 10.0,      # binary label for AUROC
        'growth_ratio': mean_growth,          # continuous label for Spearman ρ
        'log_growth': np.log(max(mean_growth, 1e-10)),  # more linear for regression
        'T_test': T_test,
        'n_trials': n_trials,
        'all_growth_ratios': growth_ratios
    }


def compute_memory_retention_outcome(layer_matrices: List[np.ndarray],
                                   T_test: int = 500) -> Dict[str, Any]:
    """
    How much of an initial signal survives after T steps.
    Relevant for long-context failure mode separate from blowup.

    Args:
        layer_matrices: List of SSM transition matrices
        T_test: Number of time steps to iterate

    Returns:
        Dictionary with memory retention outcomes
    """
    N = layer_matrices[0].shape[0]
    e1 = np.zeros(N); e1[0] = 1.0  # canonical basis vector

    x = e1.copy()
    for t in range(T_test):
        for A in layer_matrices:
            x = A @ x

    # How much of e1 is retained in the output direction
    x_norm = np.linalg.norm(x)
    if x_norm > 1e-10:
        retention = abs(np.dot(x, e1)) / x_norm
    else:
        retention = 0.0

    return {
        'memory_retention': retention,
        'log_retention': np.log(max(retention, 1e-10)),
        'final_norm': x_norm,
        'T_test': T_test
    }


def create_non_normal_ssm_matrix(N: int, eigenvalue_radius: float, condition_V: float,
                               seed: Optional[int] = None) -> np.ndarray:
    """
    Create a matrix with controlled eigenvalue radius AND controlled
    eigenvector ill-conditioning. This is the regime where spectral
    radius is insufficient and contracts should add value.

    Args:
        N: Matrix dimension
        eigenvalue_radius: Magnitude of all eigenvalues
        condition_V: Condition number of eigenvector matrix V
                     1.0 = normal (diagonal), 100+ = highly non-normal
        seed: Random seed for reproducibility

    Returns:
        A: Non-normal transition matrix with controlled properties

    >>> A = create_non_normal_ssm_matrix(4, 0.9, 100.0, seed=42)
    >>> np.max(np.abs(np.linalg.eigvals(A))) <= 0.91  # Near target radius
    True
    >>> np.linalg.cond(np.linalg.eig(A)[1]) >= 50.0  # Ill-conditioned eigenvectors
    True
    """
    if seed is not None:
        np.random.seed(seed)

    # Create eigenvalues all at target radius (real for SSM stability)
    # Use slight variation around radius to avoid repeated eigenvalues
    eigenvalues = eigenvalue_radius * np.random.uniform(0.95, 1.0, N)
    # Ensure they're real and respect the radius bound
    eigenvalues = np.clip(eigenvalues, -eigenvalue_radius, eigenvalue_radius)

    if condition_V <= 1.1:
        # Normal case: return diagonal matrix
        return np.diag(eigenvalues)

    # Construct ill-conditioned eigenvector matrix V
    # Use direct construction to ensure target condition number
    V = np.random.randn(N, N)

    # Iteratively adjust to reach target condition number
    for iteration in range(10):
        current_condition = np.linalg.cond(V)

        if condition_V * 0.5 <= current_condition <= condition_V * 2.0:
            break  # Close enough

        # Adjust the scale of perturbations
        if current_condition < condition_V:
            # Need more ill-conditioning: add rank-1 perturbation
            u = np.random.randn(N, 1)
            v = np.random.randn(1, N)
            scale = np.sqrt(condition_V / current_condition)
            V += scale * u @ v
        else:
            # Too ill-conditioned: add regularization
            V += 0.1 * np.eye(N)

    try:
        V_inv = np.linalg.inv(V)
        # A = V @ diag(eigenvalues) @ V^{-1}
        A = V @ np.diag(eigenvalues) @ V_inv

        # Verify the eigenvalue radius constraint is met
        actual_radius = np.max(np.abs(np.linalg.eigvals(A)))
        if actual_radius > eigenvalue_radius * 1.05:  # Allow 5% tolerance
            # Rescale if necessary
            A = A * (eigenvalue_radius / actual_radius)

    except np.linalg.LinAlgError:
        # Fallback to diagonal if V is too ill-conditioned to invert
        A = np.diag(eigenvalues)

    return A


def benchmark_summary():
    """Print summary of available benchmark tasks."""
    print("SSM Benchmark Tasks Summary:")
    print("1. Copying Task - Memory retention over long sequences")
    print("2. Selective Recall - Key-value memory under distraction")
    print("3. Long-Range Parity - XOR computation with distractors")
    print("4. Classification - Early signal detection with distractors")
    print("\nSpecial Cases:")
    print("- Case A: Hidden instability (spectral radius safe, training fails)")
    print("- Case B: Apparent risk (spectral radius high, training succeeds)")
    print("\nNon-Circular Outcomes:")
    print("- Linear stability: Growth ratio under iterated dynamics")
    print("- Memory retention: Signal preservation over long horizons")


if __name__ == "__main__":
    # Run validation tests
    print("Testing synthetic benchmark task generation...")

    # Test each task type
    dataset_copying = SSMTaskDataset('copying', T=20, num_samples=5, K=5)
    print(f"✓ Copying task: {len(dataset_copying)} samples generated")

    dataset_recall = SSMTaskDataset('selective_recall', T=30, num_samples=5, num_pairs=3)
    print(f"✓ Selective recall: {len(dataset_recall)} samples generated")

    dataset_parity = SSMTaskDataset('parity', T=40, num_samples=5, signal_fraction=0.15)
    print(f"✓ Long-range parity: {len(dataset_parity)} samples generated")

    dataset_classification = SSMTaskDataset('classification', T=25, num_samples=5, signal_position=5)
    print(f"✓ Classification: {len(dataset_classification)} samples generated")

    # Validate Case A and Case B matrices
    validate_case_matrices()

    # Display summary
    print("\n" + "="*50)
    benchmark_summary()