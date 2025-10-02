#!/usr/bin/env python3
"""
Standalone Data Download Script for Gravitational Wave Hunter v2.0

This script downloads gravitational wave data based on configuration files.
It runs independently of the training pipeline and creates a manifest of all downloads.

Usage:
    python scripts/download_data.py --config config/download_config.yaml
    python scripts/download_data.py --config config/download_config.yaml --no-confirm
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from downloader.data_downloader import GWOSCDownloader
from downloader.config_validator import ConfigValidator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main function for standalone data download.
    
    Parses command line arguments, validates configuration, and executes
    the download process with proper error handling and user feedback.
    
    Command Line Arguments
    ---------------------
    --config : str
        Path to YAML configuration file (required)
    --no-confirm : bool
        Skip confirmation prompt (optional)
    --validate-only : bool
        Only validate configuration, don't download (optional)
    --schema : str
        Path to schema file for validation (default: config/schema.yaml)
    --verbose : bool
        Enable verbose logging (optional)
        
    Raises
    ------
    SystemExit
        Exits with code 0 on success, 1 on failure
    """
    parser = argparse.ArgumentParser(
        description="Download gravitational wave data from GWOSC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download with confirmation prompt
  python scripts/download_data.py --config config/download_config.yaml
  
  # Download without confirmation
  python scripts/download_data.py --config config/download_config.yaml --no-confirm
  
  # Validate config only
  python scripts/download_data.py --config config/download_config.yaml --validate-only
        """
    )
    
    parser.add_argument(
        "--config", 
        required=True, 
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--no-confirm", 
        action="store_true", 
        help="Skip confirmation prompt"
    )
    parser.add_argument(
        "--validate-only", 
        action="store_true", 
        help="Only validate configuration, don't download"
    )
    parser.add_argument(
        "--schema", 
        default="config/schema.yaml", 
        help="Path to schema file for validation"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Check if schema file exists
    if not os.path.exists(args.schema):
        logger.error(f"Schema file not found: {args.schema}")
        sys.exit(1)
    
    try:
        # Validate configuration
        logger.info("Validating configuration...")
        validator = ConfigValidator(args.schema)
        validation_result = validator.validate_downloader_config(args.config)
        
        if not validation_result['valid']:
            logger.error("Configuration validation failed:")
            for error in validation_result['errors']:
                logger.error(f"   - {error}")
            sys.exit(1)
        
        logger.info("Configuration validation passed")
        
        # Show warnings if any
        if validation_result['warnings']:
            logger.warning("Configuration warnings:")
            for warning in validation_result['warnings']:
                logger.warning(f"   - {warning}")
        
        # If only validating, exit here
        if args.validate_only:
            logger.info("Configuration is valid (validation-only mode)")
            return
        
        # Initialize downloader
        logger.info("Initializing downloader...")
        downloader = GWOSCDownloader(args.config)
        
        # Run download
        logger.info("Starting download process...")
        results = downloader.download_all_segments(require_confirmation=not args.no_confirm)
        
        # Print results
        if results["status"] == "cancelled":
            logger.info("Download cancelled by user")
            sys.exit(0)
        elif results["status"] == "no_segments":
            logger.warning("No segments configured for download")
            sys.exit(0)
        
        # Success - print summary
        print("\n" + "="*60)
        print("DOWNLOAD SUMMARY")
        print("="*60)
        print(f"Total segments:     {results['total']}")
        print(f"Successful:         {results['successful']}")
        print(f"Failed:             {results['failed']}")
        print(f"Skipped:             {results['skipped']}")
        
        if results['failed'] > 0:
            print(f"\nFailed downloads:")
            for result in results['results']:
                if result['status'] == 'failed':
                    print(f"   - {result['segment_id']}: {result.get('reason', 'Unknown error')}")
        
        print(f"\nData saved to: {downloader.raw_data_dir}")
        print(f"Manifest:       {downloader.manifest_path}")
        print("="*60)
        
        # Exit with error code if any downloads failed
        if results['failed'] > 0:
            logger.warning(f"{results['failed']} downloads failed")
            sys.exit(1)
        else:
            logger.info("All downloads completed successfully!")
    
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
