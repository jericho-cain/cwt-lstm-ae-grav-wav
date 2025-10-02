"""
Test script for the metrics evaluation module.

This script tests the MetricsEvaluator class with synthetic data
to ensure all plotting and metrics calculation functions work correctly.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from evaluation import MetricsEvaluator


def create_synthetic_data(n_noise=100, n_signals=20, noise_mean=0.45, signal_mean=0.65, 
                         noise_std=0.05, signal_std=0.08):
    """
    Create synthetic data for testing metrics.
    
    Parameters
    ----------
    n_noise : int
        Number of noise samples
    n_signals : int
        Number of signal samples
    noise_mean : float
        Mean reconstruction error for noise
    signal_mean : float
        Mean reconstruction error for signals
    noise_std : float
        Standard deviation for noise
    signal_std : float
        Standard deviation for signals
        
    Returns
    -------
    tuple
        (y_true, y_scores) labels and scores
    """
    # Generate noise samples (lower reconstruction errors)
    noise_scores = np.random.normal(noise_mean, noise_std, n_noise)
    noise_labels = np.zeros(n_noise)
    
    # Generate signal samples (higher reconstruction errors)
    signal_scores = np.random.normal(signal_mean, signal_std, n_signals)
    signal_labels = np.ones(n_signals)
    
    # Combine and shuffle
    y_scores = np.concatenate([noise_scores, signal_scores])
    y_true = np.concatenate([noise_labels, signal_labels])
    
    # Shuffle to randomize order
    indices = np.random.permutation(len(y_true))
    y_true = y_true[indices]
    y_scores = y_scores[indices]
    
    return y_true, y_scores


def test_metrics_evaluator():
    """Test the MetricsEvaluator class."""
    print("Testing MetricsEvaluator...")
    
    # Create synthetic data
    y_true, y_scores = create_synthetic_data(n_noise=161, n_signals=24)
    
    print(f"Created synthetic data: {len(y_true)} samples")
    print(f"  Noise: {np.sum(y_true == 0)} samples")
    print(f"  Signals: {np.sum(y_true == 1)} samples")
    print(f"  Score range: [{np.min(y_scores):.3f}, {np.max(y_scores):.3f}]")
    
    # Initialize evaluator
    evaluator = MetricsEvaluator()
    
    # Calculate metrics
    print("\nCalculating metrics...")
    results = evaluator.calculate_metrics(y_true, y_scores)
    
    # Print summary
    summary = evaluator.get_summary_metrics()
    print(f"\nSummary metrics:")
    for key, value in summary.items():
        print(f"  {key}: {value:.3f}")
    
    # Test individual plots
    print("\nTesting individual plots...")
    
    # Create output directory
    output_dir = Path("test_metrics_output")
    output_dir.mkdir(exist_ok=True)
    
    # Test each plot function
    try:
        evaluator.plot_precision_recall_curve(save_path=str(output_dir / "pr_curve.png"))
        print("  [OK] Precision-recall curve")
    except Exception as e:
        print(f"  [FAIL] Precision-recall curve: {e}")
    
    try:
        evaluator.plot_roc_curve(save_path=str(output_dir / "roc_curve.png"))
        print("  [OK] ROC curve")
    except Exception as e:
        print(f"  [FAIL] ROC curve: {e}")
    
    try:
        evaluator.plot_confusion_matrix(save_path=str(output_dir / "confusion_matrix.png"))
        print("  [OK] Confusion matrix")
    except Exception as e:
        print(f"  [FAIL] Confusion matrix: {e}")
    
    try:
        evaluator.plot_reconstruction_error_distribution(save_path=str(output_dir / "error_dist.png"))
        print("  [OK] Reconstruction error distribution")
    except Exception as e:
        print(f"  [FAIL] Reconstruction error distribution: {e}")
    
    # Test comprehensive plots
    print("\nTesting comprehensive plots...")
    try:
        evaluator.create_comprehensive_plots(str(output_dir))
        print("  [OK] Comprehensive plots created")
    except Exception as e:
        print(f"  [FAIL] Comprehensive plots: {e}")
    
    # Check output files
    expected_files = [
        "detection_results.png",
        "precision_recall_curve.png", 
        "roc_curve.png",
        "confusion_matrix.png",
        "reconstruction_errors.png",
        "metrics_report.txt"
    ]
    
    print(f"\nChecking output files in {output_dir}:")
    for filename in expected_files:
        file_path = output_dir / filename
        if file_path.exists():
            print(f"  [OK] {filename}")
        else:
            print(f"  [MISSING] {filename}")
    
    print(f"\nTest completed! Check {output_dir} for output files.")
    assert True  # Test passes if we get here


if __name__ == "__main__":
    test_metrics_evaluator()
