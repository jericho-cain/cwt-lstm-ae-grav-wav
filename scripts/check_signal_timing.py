#!/usr/bin/env python3
"""
Check Signal Timing

Verify if the 15-second mark in the CWT spectrogram corresponds to the actual LIGO detection time.
"""

import numpy as np
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_manifest():
    """Load the download manifest to get event information."""
    manifest_path = Path("data/download_manifest.json")
    if not manifest_path.exists():
        logger.error("Manifest file not found!")
        return {}
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    return manifest

def check_gw150914_timing():
    """Check the timing for GW150914 specifically."""
    # GW150914 detection time from LIGO
    gw150914_detection_time = 1126259462.4  # GPS time of detection
    
    # Our signal file GPS time
    signal_gps_time = 1126259462  # From filename H1_1126259462_32s_cwt.npy
    
    # Segment duration
    segment_duration = 32  # seconds
    
    # Calculate the time range covered by our segment
    segment_start = signal_gps_time
    segment_end = signal_gps_time + segment_duration
    
    logger.info("GW150914 Timing Analysis:")
    logger.info(f"  LIGO detection time: {gw150914_detection_time}")
    logger.info(f"  Our segment start:   {segment_start}")
    logger.info(f"  Our segment end:     {segment_end}")
    logger.info(f"  Segment duration:    {segment_duration} seconds")
    
    # Check if detection time is within our segment
    if segment_start <= gw150914_detection_time <= segment_end:
        # Calculate the time within our segment
        time_within_segment = gw150914_detection_time - segment_start
        logger.info(f"  Detection time within segment: {time_within_segment:.1f} seconds")
        
        # Check if this matches the ~15 seconds we see in the spectrogram
        if 14 <= time_within_segment <= 16:
            logger.info("✅ PERFECT MATCH! The ~15s mark in the spectrogram corresponds to the actual LIGO detection time!")
        else:
            logger.info(f"⚠️  Detection time is at {time_within_segment:.1f}s, not ~15s")
    else:
        logger.error("❌ Detection time is outside our segment range!")
    
    return time_within_segment if segment_start <= gw150914_detection_time <= segment_end else None

def analyze_other_signals():
    """Analyze timing for other signals in our dataset."""
    manifest = load_manifest()
    
    logger.info("\nOther Signal Timing Analysis:")
    
    signal_events = []
    for download in manifest['downloads']:
        if download.get('successful', False) and download.get('segment_type') == 'signal':
            gps_time = download.get('start_gps')
            event_name = download.get('event', 'Unknown')
            if gps_time:
                signal_events.append((event_name, gps_time))
    
    # Sort by GPS time
    signal_events.sort(key=lambda x: x[1])
    
    logger.info(f"Found {len(signal_events)} signal events:")
    for i, (event, gps_time) in enumerate(signal_events[:10]):  # Show first 10
        logger.info(f"  {i+1:2d}. {event}: {gps_time}")
    
    return signal_events

def main():
    """Main analysis function."""
    logger.info("Signal Timing Verification")
    logger.info("=" * 50)
    
    # Check GW150914 timing
    detection_time_offset = check_gw150914_timing()
    
    # Analyze other signals
    signal_events = analyze_other_signals()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("SUMMARY")
    logger.info("="*50)
    
    if detection_time_offset is not None:
        if 14 <= detection_time_offset <= 16:
            logger.info("✅ CONFIRMED: The visible signal feature at ~15s corresponds to the actual LIGO detection time!")
            logger.info("   This proves our CWT preprocessing is correctly preserving signal timing.")
        else:
            logger.info(f"⚠️  Detection time offset: {detection_time_offset:.1f}s (not ~15s)")
    else:
        logger.error("❌ Could not verify timing")

if __name__ == "__main__":
    main()
