#!/usr/bin/env python3
"""
Quick test of legacy CWT preprocessing on 500 noise files to check loss values.
This allows fast iteration to verify the legacy approach works before full pipeline.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import logging
import sys
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing.legacy_cwt import legacy_cwt_clean
from models.cwtlstm import CWT_LSTM_Autoencoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_manifest():
    """Load the download manifest to identify noise files"""
    manifest_path = Path("data/download_manifest.json")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Create mapping from GPS time to segment type
    gps_to_type = {}
    for download in manifest['downloads']:
        if download.get('successful', False):
            gps_time = download.get('start_gps')
            segment_type = download.get('segment_type')
            if gps_time and segment_type:
                gps_to_type[gps_time] = segment_type
    
    return gps_to_type

def find_noise_files(raw_dir, gps_to_type, max_files=100):
    """Find noise files from raw data"""
    noise_files = []
    
    for file_path in raw_dir.glob("*.npz"):
        try:
            # Extract GPS time from filename
            filename_parts = file_path.stem.split('_')
            if len(filename_parts) >= 2:
                gps_time = int(filename_parts[1])
                
                # Check if this is a noise file
                if gps_to_type.get(gps_time) == 'noise':
                    noise_files.append(file_path)
                    
                if len(noise_files) >= max_files:
                    break
        except (ValueError, IndexError):
            continue
    
    logger.info(f"Found {len(noise_files)} noise files")
    return noise_files

def load_raw_data(file_path):
    """Load raw strain data from file"""
    try:
        if file_path.suffix == '.npz':
            data_dict = np.load(file_path)
            data = data_dict[list(data_dict.keys())[0]]
        else:
            data = np.load(file_path)
        return data
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None

def process_files_with_legacy_cwt(noise_files, sample_rate=4096):
    """Process files using legacy CWT preprocessing"""
    processed_data = []
    
    logger.info(f"Processing {len(noise_files)} files with legacy CWT...")
    
    for i, file_path in enumerate(noise_files):
        try:
            # Load raw data
            raw_data = load_raw_data(file_path)
            if raw_data is None:
                continue
            
            # Apply legacy CWT preprocessing
            cwt_data, freqs, scales, coi = legacy_cwt_clean(
                raw_data, fs=sample_rate, fmin=20.0, fmax=512.0, n_scales=64
            )
            
            processed_data.append(cwt_data)
            
            if (i + 1) % 50 == 0:
                logger.info(f"Processed {i + 1}/{len(noise_files)} files")
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    logger.info(f"Successfully processed {len(processed_data)} files")
    return processed_data

def test_model_training(cwt_data_list, num_epochs=3):
    """Test model training on legacy CWT data"""
    
    if not cwt_data_list:
        logger.error("No CWT data to test")
        return
    
    # Convert to numpy array
    cwt_data = np.array(cwt_data_list)
    logger.info(f"CWT data shape: {cwt_data.shape}")
    logger.info(f"CWT data range: {cwt_data.min():.6e} to {cwt_data.max():.6e}")
    logger.info(f"CWT data mean: {cwt_data.mean():.6e}, std: {cwt_data.std():.6e}")
    
    # Convert to PyTorch tensors
    cwt_tensors = torch.tensor(cwt_data, dtype=torch.float32)
    cwt_tensors = cwt_tensors.unsqueeze(1)  # Add channel dimension: (N, 1, H, W)
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(cwt_tensors)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Create model
    model = CWT_LSTM_Autoencoder(
        input_height=cwt_data.shape[1],
        input_width=cwt_data.shape[2],
        latent_dim=32,
        lstm_hidden=64
    )
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
    
    logger.info(f"Starting training test with {num_epochs} epochs...")
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data,) in enumerate(data_loader):
            # Forward pass
            reconstructed, latent = model(data)
            
            # Compute loss
            loss = criterion(reconstructed, data)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")
        
        # Check if we're getting tiny losses (bad sign)
        if avg_loss < 0.001:
            logger.warning(f"⚠️  TINY LOSS DETECTED: {avg_loss:.6f} - Legacy CWT might not be working correctly")
        elif avg_loss > 0.1:
            logger.info(f"✅ GOOD LOSS: {avg_loss:.6f} - Legacy CWT appears to be working correctly")

def main():
    """Main test function"""
    
    logger.info("=== LEGACY CWT SMALL SCALE TEST ===")
    
    # Load manifest
    gps_to_type = load_manifest()
    
    # Find noise files
    raw_dir = Path("data/raw")
    noise_files = find_noise_files(raw_dir, gps_to_type, max_files=500)
    
    if not noise_files:
        logger.error("No noise files found!")
        return
    
    # Process files with legacy CWT
    cwt_data_list = process_files_with_legacy_cwt(noise_files)
    
    if not cwt_data_list:
        logger.error("No data processed!")
        return
    
    # Test model training
    test_model_training(cwt_data_list, num_epochs=3)
    
    logger.info("=== TEST COMPLETE ===")
    logger.info("Key indicator: If loss is > 0.1, legacy CWT is working correctly")
    logger.info("If loss is < 0.001, there's still an issue with the preprocessing")

if __name__ == "__main__":
    main()
