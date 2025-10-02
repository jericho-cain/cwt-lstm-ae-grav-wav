"""
Metrics and Plotting Module for Gravitational Wave Detection

This module provides comprehensive evaluation metrics and visualization
for anomaly detection in gravitational wave data. It generates precision-recall
curves, ROC curves, confusion matrices, and reconstruction error distributions.

Author: Gravitational Wave Hunter v2.0
Date: October 2, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, roc_auc_score, 
    average_precision_score, confusion_matrix, classification_report
)
from typing import Dict, Any, Tuple, Optional, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricsEvaluator:
    """
    Comprehensive metrics evaluation for gravitational wave detection.
    
    This class calculates and visualizes performance metrics for anomaly
    detection models, including precision-recall curves, ROC curves,
    confusion matrices, and reconstruction error distributions.
    """
    
    def __init__(self, sample_rate: int = 4096):
        """
        Initialize the metrics evaluator.
        
        Parameters
        ----------
        sample_rate : int, default=4096
            Sample rate in Hz for timing calculations
        """
        self.sample_rate = sample_rate
        self.results = {}
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def calculate_metrics(self, y_true: np.ndarray, y_scores: np.ndarray, 
                         y_pred: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive evaluation metrics.
        
        Parameters
        ----------
        y_true : np.ndarray
            True binary labels (0=noise, 1=signal)
        y_scores : np.ndarray
            Anomaly scores (reconstruction errors)
        y_pred : np.ndarray, optional
            Binary predictions (0=noise, 1=signal)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all calculated metrics
        """
        logger.info("Calculating evaluation metrics...")
        
        # Ensure inputs are numpy arrays
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores)
        
        # Calculate precision-recall curve
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)
        avg_precision = average_precision_score(y_true, y_scores)
        
        # Calculate ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        auc_score = roc_auc_score(y_true, y_scores)
        
        # Find optimal operating points
        # 1. Maximum F1 score
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        optimal_f1_idx = np.argmax(f1_scores)
        optimal_f1_threshold = pr_thresholds[optimal_f1_idx] if optimal_f1_idx < len(pr_thresholds) else pr_thresholds[-1]
        
        # 2. Maximum precision (for confusion matrix)
        max_precision_idx = np.argmax(precision)
        max_precision_threshold = pr_thresholds[max_precision_idx] if max_precision_idx < len(pr_thresholds) else pr_thresholds[-1]
        
        # 3. Find threshold that maximizes precision while maintaining reasonable recall
        precision_90_plus = precision >= 0.90
        if np.any(precision_90_plus):
            high_precision_indices = np.where(precision_90_plus)[0]
            best_recall_idx = high_precision_indices[np.argmax(recall[high_precision_indices])]
            high_precision_threshold = pr_thresholds[best_recall_idx] if best_recall_idx < len(pr_thresholds) else pr_thresholds[-1]
        else:
            high_precision_threshold = max_precision_threshold
        
        # Calculate confusion matrix at maximum precision threshold
        y_pred_max_precision = (y_scores >= max_precision_threshold).astype(int)
        cm_max_precision = confusion_matrix(y_true, y_pred_max_precision)
        
        # Calculate metrics at maximum precision
        tn, fp, fn, tp = cm_max_precision.ravel()
        precision_max = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_max = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_max = 2 * (precision_max * recall_max) / (precision_max + recall_max) if (precision_max + recall_max) > 0 else 0
        
        # Calculate baseline (random classifier)
        baseline_precision = np.mean(y_true)
        
        # Store results
        self.results = {
            'precision': precision,
            'recall': recall,
            'pr_thresholds': pr_thresholds,
            'avg_precision': avg_precision,
            'fpr': fpr,
            'tpr': tpr,
            'roc_thresholds': roc_thresholds,
            'auc': auc_score,
            'optimal_f1_threshold': optimal_f1_threshold,
            'optimal_f1': f1_scores[optimal_f1_idx] if len(f1_scores) > optimal_f1_idx else 0,
            'max_precision_threshold': max_precision_threshold,
            'max_precision': precision[max_precision_idx],
            'max_precision_recall': recall[max_precision_idx],
            'high_precision_threshold': high_precision_threshold,
            'confusion_matrix_max_precision': cm_max_precision,
            'precision_max_precision': precision_max,
            'recall_max_precision': recall_max,
            'f1_max_precision': f1_max,
            'baseline_precision': baseline_precision,
            'y_true': y_true,
            'y_scores': y_scores,
            'y_pred_max_precision': y_pred_max_precision
        }
        
        logger.info(f"Metrics calculated - AUC: {auc_score:.3f}, AP: {avg_precision:.3f}")
        logger.info(f"Max Precision: {precision_max:.3f}, Recall: {recall_max:.3f}, F1: {f1_max:.3f}")
        
        return self.results
    
    def plot_precision_recall_curve(self, ax: Optional[plt.Axes] = None, 
                                   save_path: Optional[str] = None) -> plt.Axes:
        """
        Plot precision-recall curve.
        
        Parameters
        ----------
        ax : plt.Axes, optional
            Matplotlib axes to plot on
        save_path : str, optional
            Path to save the plot
            
        Returns
        -------
        plt.Axes
            The axes object
        """
        if not self.results:
            raise ValueError("No metrics calculated. Call calculate_metrics() first.")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot precision-recall curve
        ax.plot(self.results['recall'], self.results['precision'], 'b-', linewidth=2,
                label=f'CWT-LSTM Autoencoder (AP={self.results["avg_precision"]:.3f})')
        
        # Add baseline
        ax.axhline(y=self.results['baseline_precision'], color='gray', linestyle='--', alpha=0.8,
                   label=f'Random (AP={self.results["baseline_precision"]:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Precision-recall curve saved to {save_path}")
        
        return ax
    
    def plot_roc_curve(self, ax: Optional[plt.Axes] = None, 
                      save_path: Optional[str] = None) -> plt.Axes:
        """
        Plot ROC curve.
        
        Parameters
        ----------
        ax : plt.Axes, optional
            Matplotlib axes to plot on
        save_path : str, optional
            Path to save the plot
            
        Returns
        -------
        plt.Axes
            The axes object
        """
        if not self.results:
            raise ValueError("No metrics calculated. Call calculate_metrics() first.")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot ROC curve
        ax.plot(self.results['fpr'], self.results['tpr'], 'b-', linewidth=2,
                label=f'CWT-LSTM Autoencoder (AUC={self.results["auc"]:.3f})')
        
        # Add random baseline
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.8, label='Random')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        return ax
    
    def plot_confusion_matrix(self, ax: Optional[plt.Axes] = None, 
                             save_path: Optional[str] = None) -> plt.Axes:
        """
        Plot confusion matrix at maximum precision threshold.
        
        Parameters
        ----------
        ax : plt.Axes, optional
            Matplotlib axes to plot on
        save_path : str, optional
            Path to save the plot
            
        Returns
        -------
        plt.Axes
            The axes object
        """
        if not self.results:
            raise ValueError("No metrics calculated. Call calculate_metrics() first.")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        cm = self.results['confusion_matrix_max_precision']
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Pred Noise', 'Pred Signal'],
                    yticklabels=['True Noise', 'True Signal'])
        
        # Add metrics text
        metrics_text = (f'P={self.results["precision_max_precision"]:.3f}\n'
                       f'R={self.results["recall_max_precision"]:.3f}\n'
                       f'F1={self.results["f1_max_precision"]:.3f}')
        
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        ax.set_title('Confusion Matrix (Max Precision)')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return ax
    
    def plot_reconstruction_error_distribution(self, ax: Optional[plt.Axes] = None, 
                                             save_path: Optional[str] = None) -> plt.Axes:
        """
        Plot reconstruction error distribution for noise and signals.
        
        Parameters
        ----------
        ax : plt.Axes, optional
            Matplotlib axes to plot on
        save_path : str, optional
            Path to save the plot
            
        Returns
        -------
        plt.Axes
            The axes object
        """
        if not self.results:
            raise ValueError("No metrics calculated. Call calculate_metrics() first.")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        y_true = self.results['y_true']
        y_scores = self.results['y_scores']
        
        # Separate scores by class
        noise_scores = y_scores[y_true == 0]
        signal_scores = y_scores[y_true == 1]
        
        # Plot histograms
        ax.hist(noise_scores, bins=30, alpha=0.7, label=f'Noise (n={len(noise_scores)})', 
                color='blue', density=True)
        ax.hist(signal_scores, bins=30, alpha=0.7, label=f'Signals (n={len(signal_scores)})', 
                color='red', density=True)
        
        # Add optimal threshold line
        ax.axvline(x=self.results['max_precision_threshold'], color='green', 
                   linestyle='--', linewidth=2, label='Optimal Threshold')
        
        # Calculate and display statistics
        noise_mean = np.mean(noise_scores)
        signal_mean = np.mean(signal_scores)
        separation = abs(signal_mean - noise_mean)
        
        stats_text = (f'Noise μ={noise_mean:.3f}\n'
                     f'Signal μ={signal_mean:.3f}\n'
                     f'Separation={separation:.3f}')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        ax.set_xlabel('Reconstruction Error (Anomaly Score)')
        ax.set_ylabel('Density')
        ax.set_title('Reconstruction Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Reconstruction error distribution saved to {save_path}")
        
        return ax
    
    def create_comprehensive_plots(self, save_dir: str) -> None:
        """
        Create all evaluation plots and save them.
        
        Parameters
        ----------
        save_dir : str
            Directory to save all plots
        """
        if not self.results:
            raise ValueError("No metrics calculated. Call calculate_metrics() first.")
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Creating comprehensive evaluation plots...")
        
        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Gravitational Wave Detection Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Precision-Recall Curve
        self.plot_precision_recall_curve(ax=axes[0, 0])
        
        # Plot 2: ROC Curve
        self.plot_roc_curve(ax=axes[0, 1])
        
        # Plot 3: Confusion Matrix
        self.plot_confusion_matrix(ax=axes[1, 0])
        
        # Plot 4: Reconstruction Error Distribution
        self.plot_reconstruction_error_distribution(ax=axes[1, 1])
        
        # Save comprehensive plot
        comprehensive_path = save_path / 'detection_results.png'
        plt.tight_layout()
        plt.savefig(comprehensive_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comprehensive plots saved to {comprehensive_path}")
        
        # Save individual plots
        self.plot_precision_recall_curve(save_path=str(save_path / 'precision_recall_curve.png'))
        self.plot_roc_curve(save_path=str(save_path / 'roc_curve.png'))
        self.plot_confusion_matrix(save_path=str(save_path / 'confusion_matrix.png'))
        self.plot_reconstruction_error_distribution(save_path=str(save_path / 'reconstruction_errors.png'))
        
        # Generate metrics report
        self.generate_metrics_report(save_path)
    
    def generate_metrics_report(self, save_dir: Path) -> None:
        """
        Generate a comprehensive metrics report.
        
        Parameters
        ----------
        save_dir : Path
            Directory to save the report
        """
        if not self.results:
            raise ValueError("No metrics calculated. Call calculate_metrics() first.")
        
        report_path = save_dir / 'metrics_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("GRAVITATIONAL WAVE DETECTION METRICS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall performance
            f.write("OVERALL PERFORMANCE:\n")
            f.write(f"  AUC-ROC: {self.results['auc']:.3f}\n")
            f.write(f"  Average Precision: {self.results['avg_precision']:.3f}\n")
            f.write(f"  Baseline (Random): {self.results['baseline_precision']:.3f}\n\n")
            
            # Optimal operating points
            f.write("OPTIMAL OPERATING POINTS:\n")
            f.write(f"  Max F1 Score: {self.results['optimal_f1']:.3f}\n")
            f.write(f"  Max F1 Threshold: {self.results['optimal_f1_threshold']:.3f}\n")
            f.write(f"  Max Precision: {self.results['max_precision']:.3f}\n")
            f.write(f"  Max Precision Recall: {self.results['max_precision_recall']:.3f}\n")
            f.write(f"  Max Precision Threshold: {self.results['max_precision_threshold']:.3f}\n\n")
            
            # Confusion matrix at max precision
            f.write("CONFUSION MATRIX (MAX PRECISION):\n")
            cm = self.results['confusion_matrix_max_precision']
            f.write(f"  True Negatives: {cm[0, 0]}\n")
            f.write(f"  False Positives: {cm[0, 1]}\n")
            f.write(f"  False Negatives: {cm[1, 0]}\n")
            f.write(f"  True Positives: {cm[1, 1]}\n")
            f.write(f"  Precision: {self.results['precision_max_precision']:.3f}\n")
            f.write(f"  Recall: {self.results['recall_max_precision']:.3f}\n")
            f.write(f"  F1-Score: {self.results['f1_max_precision']:.3f}\n\n")
            
            # Data statistics
            y_true = self.results['y_true']
            y_scores = self.results['y_scores']
            f.write("DATA STATISTICS:\n")
            f.write(f"  Total Samples: {len(y_true)}\n")
            f.write(f"  Noise Samples: {np.sum(y_true == 0)}\n")
            f.write(f"  Signal Samples: {np.sum(y_true == 1)}\n")
            f.write(f"  Signal Ratio: {np.mean(y_true):.3f}\n")
            f.write(f"  Score Range: [{np.min(y_scores):.3f}, {np.max(y_scores):.3f}]\n")
            f.write(f"  Score Mean: {np.mean(y_scores):.3f}\n")
            f.write(f"  Score Std: {np.std(y_scores):.3f}\n")
        
        logger.info(f"Metrics report saved to {report_path}")
    
    def get_summary_metrics(self) -> Dict[str, float]:
        """
        Get a summary of key metrics.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of key metrics
        """
        if not self.results:
            raise ValueError("No metrics calculated. Call calculate_metrics() first.")
        
        return {
            'auc': self.results['auc'],
            'avg_precision': self.results['avg_precision'],
            'max_precision': self.results['max_precision'],
            'max_precision_recall': self.results['max_precision_recall'],
            'f1_max_precision': self.results['f1_max_precision'],
            'optimal_f1': self.results['optimal_f1']
        }
