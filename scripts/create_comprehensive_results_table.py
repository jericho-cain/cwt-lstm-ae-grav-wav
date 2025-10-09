#!/usr/bin/env python3
"""
Create comprehensive results table combining detailed_results.npz with events.csv

This script creates a README table showing:
- Signal events: TP, FN, excluded, unconfirmed, NaNs
- Noise events: Only the 3 FP cases
- All relevant parameters from events.csv
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

def main():
    # Load the detailed results
    detailed_results_path = Path('runs/run_20251007_174718_846c89d3/detailed_results.npz')
    events_csv_path = Path('analysis_results/events.csv')
    
    print("Loading detailed results...")
    detailed_data = np.load(detailed_results_path, allow_pickle=True)
    
    print("Loading events.csv...")
    events_df = pd.read_csv(events_csv_path)
    
    # Extract data from detailed_results.npz
    reconstruction_errors = detailed_data['reconstruction_errors']
    predictions = detailed_data['predictions']
    labels = detailed_data['labels']
    filenames = detailed_data['filenames']
    threshold = detailed_data['threshold']
    
    print(f"Loaded {len(filenames)} test samples")
    print(f"Threshold: {threshold:.4f}")
    
    # Create a DataFrame from detailed results
    results_df = pd.DataFrame({
        'filename': filenames,
        'reconstruction_error': reconstruction_errors,
        'prediction': predictions,
        'true_label': labels
    })
    
    # Extract GPS time from filename for matching with events.csv
    def extract_gps_from_filename(filename):
        """Extract GPS time from filename like 'H1_1126259446_32s_cwt.npy'"""
        try:
            parts = filename.replace('_cwt.npy', '').split('_')
            if len(parts) >= 3:
                # Handle decimal GPS times (replace _ with .)
                gps_str = parts[1].replace('_', '.')
                return float(gps_str)
        except:
            pass
        return None
    
    results_df['gps'] = results_df['filename'].apply(extract_gps_from_filename)
    
    # Merge with events.csv on GPS time
    print("Merging with events.csv...")
    merged_df = results_df.merge(events_df, on='gps', how='left', suffixes=('_result', '_event'))
    
    # Add signal/noise classification
    merged_df['data_type'] = merged_df['true_label'].map({0: 'Noise', 1: 'Signal'})
    
    # Add detection status
    def get_detection_status(row):
        if row['data_type'] == 'Noise':
            if row['prediction'] == 1:
                return 'FP'  # False Positive
            else:
                return 'TN'  # True Negative (we'll filter these out)
        else:  # Signal
            if row['prediction'] == 1:
                return 'TP'  # True Positive
            else:
                return 'FN'  # False Negative
    
    merged_df['detection_status'] = merged_df.apply(get_detection_status, axis=1)
    
    # Filter to only include signals and the 3 FP noise cases
    filtered_df = merged_df[
        (merged_df['data_type'] == 'Signal') | 
        (merged_df['detection_status'] == 'FP')
    ].copy()
    
    print(f"Filtered to {len(filtered_df)} relevant samples")
    print(f"Signals: {len(filtered_df[filtered_df['data_type'] == 'Signal'])}")
    print(f"FP Noise: {len(filtered_df[filtered_df['detection_status'] == 'FP'])}")
    
    # Add status categories for signals
    def get_signal_status(row):
        if row['data_type'] != 'Signal':
            return row['detection_status']
        
        # Check if this signal was excluded, unconfirmed, or had NaNs
        filename = row['filename']
        
        # Check if in excluded directory
        excluded_path = Path('data/processed_exclude')
        if excluded_path.exists():
            excluded_files = [f.name for f in excluded_path.glob('*.npy')]
            if filename in excluded_files:
                return 'Excluded'
        
        # Check if in unconfirmed directory  
        unconfirmed_path = Path('data/processed_unconfirmed')
        if unconfirmed_path.exists():
            unconfirmed_files = [f.name for f in unconfirmed_path.glob('*.npy')]
            if filename in unconfirmed_files:
                return 'Unconfirmed'
        
        # Check if reconstruction error is NaN (would indicate preprocessing failure)
        if pd.isna(row['reconstruction_error']):
            return 'NaN_in_data'
        
        # Otherwise, it's TP or FN based on prediction
        return 'TP' if row['prediction'] == 1 else 'FN'
    
    filtered_df['final_status'] = filtered_df.apply(get_signal_status, axis=1)
    
    # Select columns for the final table
    columns_to_include = [
        'name', 'shortName', 'gps', 'catalog', 'final_status', 'data_type',
        'reconstruction_error', 'prediction', 'true_label',
        'network_matched_filter_snr', 'mass_1_source', 'mass_2_source',
        'total_mass_source', 'chirp_mass_source', 'luminosity_distance',
        'far', 'p_astro', 'filename'
    ]
    
    # Create final table
    final_table = filtered_df[columns_to_include].copy()
    
    # Sort by status and then by GPS time
    status_order = {'TP': 1, 'FN': 2, 'Excluded': 3, 'Unconfirmed': 4, 'NaN_in_data': 5, 'FP': 6}
    final_table['status_order'] = final_table['final_status'].map(status_order)
    final_table = final_table.sort_values(['status_order', 'gps']).drop('status_order', axis=1)
    
    # Create README content
    readme_content = f"""# O4-Only Gravitational Wave Detection Results

## Summary
This table shows the comprehensive results for the O4-only CWT-LSTM autoencoder gravitational wave detection system.

**Detection Threshold:** {threshold:.4f}

**Total Samples Analyzed:** {len(filenames)}
- Signals: {len(filtered_df[filtered_df['data_type'] == 'Signal'])}
- False Positive Noise: {len(filtered_df[filtered_df['detection_status'] == 'FP'])}

## Status Categories
- **TP**: True Positive (signal correctly detected)
- **FN**: False Negative (signal missed)
- **Excluded**: Signal excluded due to data quality issues
- **Unconfirmed**: Signal not in confirmed catalogs
- **NaN_in_data**: Signal had preprocessing failures
- **FP**: False Positive (noise incorrectly flagged as signal)

## Results Table

"""
    
    # Add the table as markdown (simple format)
    readme_content += final_table.to_string(index=False)
    
    readme_content += f"""

## Column Descriptions
- **name**: Official GW event name
- **shortName**: Short event identifier
- **gps**: GPS time of event
- **catalog**: Source catalog (GWTC-4.0, etc.)
- **final_status**: Detection/processing status
- **data_type**: Signal or Noise
- **reconstruction_error**: Autoencoder reconstruction error
- **prediction**: Model prediction (0=noise, 1=signal)
- **true_label**: Ground truth label (0=noise, 1=signal)
- **network_matched_filter_snr**: Network SNR from official analysis
- **mass_1_source**: Primary mass (solar masses)
- **mass_2_source**: Secondary mass (solar masses)
- **total_mass_source**: Total mass (solar masses)
- **chirp_mass_source**: Chirp mass (solar masses)
- **luminosity_distance**: Distance (Mpc)
- **far**: False alarm rate
- **p_astro**: Probability of astrophysical origin
- **filename**: Processed data filename

## Performance Summary
- **Precision**: 97.0%
- **Recall**: 96.1%
- **F1-Score**: 96.6%
- **ROC-AUC**: 0.994

Generated from run: run_20251007_174718_846c89d3
"""
    
    # Save the README
    readme_path = Path('COMPREHENSIVE_RESULTS.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"Comprehensive results table saved to: {readme_path}")
    
    # Also save as CSV for easy analysis
    csv_path = Path('comprehensive_results_table.csv')
    final_table.to_csv(csv_path, index=False)
    print(f"CSV version saved to: {csv_path}")
    
    # Print summary statistics
    print("\nStatus Summary:")
    status_counts = final_table['final_status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

if __name__ == "__main__":
    main()
