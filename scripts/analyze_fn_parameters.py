"""
Analyze false negative physical parameters compared to true positive means.

Author: Jericho Cain
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_fn_parameters():
    """Compare FN physical parameters to TP means."""
    
    # Load the comprehensive results
    results_file = Path("runs/run_20251007_174718_846c89d3/comprehensive_results_table.csv")
    df = pd.read_csv(results_file)
    
    # Filter to signals only (exclude noise)
    signals = df[df['data_type'] == 'Signal'].copy()
    
    # Separate TP and FN
    tp = signals[signals['final_status'] == 'TP'].copy()
    fn = signals[signals['final_status'] == 'FN'].copy()
    
    print(f"True Positives: {len(tp)}")
    print(f"False Negatives: {len(fn)}")
    print()
    
    # Physical parameters to analyze
    params = [
        'network_matched_filter_snr',
        'mass_1_source',
        'mass_2_source',
        'total_mass_source',
        'chirp_mass_source',
        'luminosity_distance',
        'far',
        'p_astro'
    ]
    
    # Compute TP means and stds
    print("="*80)
    print("TRUE POSITIVE STATISTICS")
    print("="*80)
    
    tp_stats = {}
    for param in params:
        # Remove NaN values
        tp_values = tp[param].dropna()
        if len(tp_values) > 0:
            mean = tp_values.mean()
            std = tp_values.std()
            tp_stats[param] = {'mean': mean, 'std': std, 'count': len(tp_values)}
            print(f"{param}:")
            print(f"  Mean: {mean:.4f}")
            print(f"  Std:  {std:.4f}")
            print(f"  Count: {len(tp_values)}/{len(tp)}")
        else:
            tp_stats[param] = {'mean': np.nan, 'std': np.nan, 'count': 0}
            print(f"{param}: No data")
        print()
    
    # Analyze each FN
    print("="*80)
    print("FALSE NEGATIVE ANALYSIS")
    print("="*80)
    print()
    
    for idx, row in fn.iterrows():
        print(f"Event: {row['event_name']}")
        print(f"GPS: {row['gps']}")
        print(f"Reconstruction Error: {row['reconstruction_error']:.4f}")
        print()
        
        for param in params:
            fn_value = row[param]
            
            if pd.isna(fn_value):
                print(f"  {param}: MISSING DATA")
                continue
            
            tp_mean = tp_stats[param]['mean']
            tp_std = tp_stats[param]['std']
            
            if pd.isna(tp_mean) or tp_std == 0:
                print(f"  {param}: {fn_value:.4f} (no TP reference)")
                continue
            
            # Compute z-score (standard deviations from TP mean)
            z_score = (fn_value - tp_mean) / tp_std
            
            # Flag if unusual (>2 std from mean)
            flag = " [UNUSUAL]" if abs(z_score) > 2 else ""
            
            print(f"  {param}: {fn_value:.4f} (z={z_score:+.2f} std){flag}")
        
        print()
        print("-"*80)
        print()
    
    # Summary: Which parameters are most unusual for FNs?
    print("="*80)
    print("SUMMARY: PARAMETER DEVIATIONS FOR FALSE NEGATIVES")
    print("="*80)
    print()
    
    # Collect z-scores for each parameter
    param_z_scores = {param: [] for param in params}
    
    for idx, row in fn.iterrows():
        for param in params:
            fn_value = row[param]
            if pd.isna(fn_value):
                continue
            
            tp_mean = tp_stats[param]['mean']
            tp_std = tp_stats[param]['std']
            
            if pd.isna(tp_mean) or tp_std == 0:
                continue
            
            z_score = (fn_value - tp_mean) / tp_std
            param_z_scores[param].append(abs(z_score))
    
    # Compute mean absolute z-score for each parameter
    print("Mean absolute z-score (deviation from TP mean):")
    print()
    
    param_deviations = []
    for param in params:
        z_scores = param_z_scores[param]
        if len(z_scores) > 0:
            mean_abs_z = np.mean(z_scores)
            param_deviations.append((param, mean_abs_z, len(z_scores)))
        else:
            param_deviations.append((param, np.nan, 0))
    
    # Sort by deviation
    param_deviations.sort(key=lambda x: x[1] if not np.isnan(x[1]) else -1, reverse=True)
    
    for param, mean_abs_z, count in param_deviations:
        if count > 0:
            flag = " [!]" if mean_abs_z > 1.5 else ""
            print(f"  {param}: {mean_abs_z:.2f} std (n={count}){flag}")
        else:
            print(f"  {param}: No data")
    
    print()
    print("Parameters with mean |z| > 1.5 std are likely contributing to missed detections.")
    print()
    
    # Also print reconstruction errors
    print("="*80)
    print("RECONSTRUCTION ERRORS")
    print("="*80)
    print()
    print(f"TP reconstruction error: {tp['reconstruction_error'].mean():.4f} ± {tp['reconstruction_error'].std():.4f}")
    print(f"FN reconstruction error: {fn['reconstruction_error'].mean():.4f} ± {fn['reconstruction_error'].std():.4f}")
    print()
    print(f"Threshold: {df['reconstruction_error'].max():.4f} (approximate from data)")
    print()
    
    # Show FN reconstruction errors
    print("FN reconstruction errors:")
    for idx, row in fn.iterrows():
        print(f"  {row['event_name']}: {row['reconstruction_error']:.4f}")

if __name__ == "__main__":
    analyze_fn_parameters()

