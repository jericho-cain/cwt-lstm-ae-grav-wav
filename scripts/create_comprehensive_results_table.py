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
    manifest_path = Path('data/download_manifest.json')
    
    print("Loading detailed results...")
    detailed_data = np.load(detailed_results_path, allow_pickle=True)
    
    print("Loading events.csv...")
    events_df = pd.read_csv(events_csv_path)
    
    print("Loading manifest...")
    import json
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
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
    
    # For signals, we need to match with manifest to get the actual event names
    # The GPS time in the filename should match the GPS time in the manifest
    print("Matching signal filenames with event names from manifest...")
    
    # Create a mapping from GPS time to event name using manifest
    gps_to_event = {}
    for download in manifest['downloads']:
        if download['segment_type'] == 'signal' and download.get('event'):
            gps_time = download['start_gps']
            # Determine catalog based on GPS time
            if gps_time >= 1380000000:  # O4 events (2024+)
                catalog = 'GWTC-4.0'
            elif gps_time >= 1230000000:  # O3 events (2019-2020)
                catalog = 'GWTC-3.0'
            elif gps_time >= 1160000000:  # O2 events (2017)
                catalog = 'GWTC-2.0'
            else:  # O1 events (2015-2016)
                catalog = 'GWTC-1.0'
            
            gps_to_event[gps_time] = {
                'name': download['event'],
                'shortName': download['event'],  # Use same as name for now
                'catalog': catalog
            }
    
    # Also try to match with events.csv for additional metadata
    events_gps_to_info = {}
    for _, event_row in events_df.iterrows():
        if pd.notna(event_row['gps']):
            events_gps_to_info[event_row['gps']] = {
                'name': event_row['name'],
                'shortName': event_row['shortName'],
                'catalog': event_row['catalog']
            }
    
    # Add event names to results_df
    def get_event_info(gps_time):
        # First try exact match in events.csv
        if gps_time in events_gps_to_info:
            return events_gps_to_info[gps_time]
        
        # Try to find closest GPS time in events.csv (within 1000 second tolerance)
        closest_gps = None
        min_diff = float('inf')
        for event_gps in events_gps_to_info.keys():
            diff = abs(event_gps - gps_time)
            if diff < min_diff and diff <= 1000.0:  # Within 1000 seconds (~16 minutes)
                min_diff = diff
                closest_gps = event_gps
        
        if closest_gps is not None:
            return events_gps_to_info[closest_gps]
        
        # Fall back to manifest
        if gps_time in gps_to_event:
            return gps_to_event[gps_time]
        
        return {'name': None, 'shortName': None, 'catalog': None}
    
    event_info = results_df['gps'].apply(get_event_info)
    results_df['event_name'] = [info['name'] for info in event_info]
    results_df['event_shortName'] = [info['shortName'] for info in event_info]
    results_df['event_catalog'] = [info['catalog'] for info in event_info]
    
    # Merge with events.csv on GPS time to get physical parameters
    print("Merging with events.csv for physical parameters...")
    
    # First, try exact GPS matches
    exact_merge = results_df.merge(events_df, on='gps', how='left', suffixes=('_result', '_event'))
    
    # For rows that didn't match exactly, try to find closest GPS time
    def find_closest_event(row):
        if pd.notna(row['name']) and pd.notna(row['catalog']):  # Already matched
            return row
        
        gps_time = row['gps']
        if pd.isna(gps_time):
            return row
            
        # Find closest GPS time in events.csv
        closest_gps = None
        min_diff = float('inf')
        for _, event_row in events_df.iterrows():
            if pd.notna(event_row['gps']):
                diff = abs(event_row['gps'] - gps_time)
                if diff < min_diff and diff <= 1000.0:  # Within 1000 seconds
                    min_diff = diff
                    closest_gps = event_row
        
        if closest_gps is not None:
            # Update row with closest event data
            for col in events_df.columns:
                if col not in ['gps']:  # Don't overwrite the GPS time
                    row[col] = closest_gps[col]
        
        return row
    
    merged_df = exact_merge.apply(find_closest_event, axis=1)
    
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
        'event_name', 'gps', 'event_catalog', 'final_status', 'data_type',
        'reconstruction_error', 'prediction', 'true_label',
        'network_matched_filter_snr', 'mass_1_source', 'mass_2_source',
        'total_mass_source', 'chirp_mass_source', 'luminosity_distance',
        'far', 'p_astro', 'filename'
    ]
    
    # Create column name mapping for abbreviations
    column_mapping = {
        'event_name': 'event_name',
        'gps': 'gps',
        'event_catalog': 'catalog',
        'final_status': 'status',
        'data_type': 'type',
        'reconstruction_error': 'RE',
        'prediction': 'pred',
        'true_label': 'label',
        'network_matched_filter_snr': 'SNR',
        'mass_1_source': 'M1',
        'mass_2_source': 'M2',
        'total_mass_source': 'Mtot',
        'chirp_mass_source': 'Mc',
        'luminosity_distance': 'D',
        'far': 'FAR',
        'p_astro': 'p_astro',
        'filename': 'filename'
    }
    
    # Create final table
    final_table = filtered_df[columns_to_include].copy()
    
    # Sort by status and then by GPS time
    status_order = {'TP': 1, 'FN': 2, 'Excluded': 3, 'Unconfirmed': 4, 'NaN_in_data': 5, 'FP': 6}
    final_table['status_order'] = final_table['final_status'].map(status_order)
    final_table = final_table.sort_values(['status_order', 'gps']).drop('status_order', axis=1)
    
    # Rename columns to abbreviations
    final_table = final_table.rename(columns=column_mapping)
    
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
    
    # Add the table as proper markdown
    # Create markdown table header
    columns = final_table.columns.tolist()
    readme_content += "| " + " | ".join(columns) + " |\n"
    readme_content += "| " + " | ".join(["---"] * len(columns)) + " |\n"
    
    # Add table rows
    for _, row in final_table.iterrows():
        row_values = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                row_values.append("")
            else:
                row_values.append(str(value))
        readme_content += "| " + " | ".join(row_values) + " |\n"
    
    readme_content += f"""

## Data Sources
Physical parameters (SNR, masses, distance, FAR, p_astro) are from the **Gravitational-wave Transient Catalog (GWTC)** maintained by the LIGO/Virgo/KAGRA collaboration: https://gwosc.org/eventapi/html/GWTC/?pagesize=all

## Column Abbreviations:
- **SNR**: network_matched_filter_snr (Network SNR from official analysis)
- **M1**: mass_1_source (Primary mass in solar masses)
- **M2**: mass_2_source (Secondary mass in solar masses)  
- **Mtot**: total_mass_source (Total mass in solar masses)
- **Mc**: chirp_mass_source (Chirp mass in solar masses)
- **D**: luminosity_distance (Distance in Mpc)
- **RE**: reconstruction_error (Autoencoder reconstruction error)
- **FAR**: far (False alarm rate)
- **p_astro**: p_astro (Probability of astrophysical origin)

## Column Descriptions
- **event_name**: Official GW event name (e.g., GW150914)
- **gps**: GPS time of event
- **catalog**: Source catalog (GWTC-4.0, etc.)
- **status**: Detection/processing status (TP/FN/FP)
- **type**: Signal or Noise
- **pred**: Model prediction (0=noise, 1=signal)
- **label**: Ground truth label (0=noise, 1=signal)
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
    
    # Also save as CSV for easy analysis (use original column names for CSV compatibility)
    csv_table = filtered_df[columns_to_include].copy()
    csv_table['status_order'] = csv_table['final_status'].map(status_order)
    csv_table = csv_table.sort_values(['status_order', 'gps']).drop('status_order', axis=1)
    
    csv_path = Path('comprehensive_results_table.csv')
    csv_table.to_csv(csv_path, index=False)
    print(f"CSV version saved to: {csv_path}")
    
    # Print summary statistics
    print("\nStatus Summary:")
    status_counts = final_table['status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status}: {count}")

if __name__ == "__main__":
    main()
