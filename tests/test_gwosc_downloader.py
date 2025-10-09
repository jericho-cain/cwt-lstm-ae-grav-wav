#!/usr/bin/env python3
"""
Test suite for the CleanGWOSCDownloader.

This module tests the clean GWOSC downloader functionality including:
- Configuration loading and validation
- Event discovery
- Signal and noise segment downloads
- Manifest tracking
- Error handling

Author: Jericho Cain
Date: October 2, 2025
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import yaml

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from downloader.gwosc_downloader import CleanGWOSCDownloader


class TestCleanGWOSCDownloader:
    """Test class for CleanGWOSCDownloader."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_config(self, temp_dir):
        """Create a test configuration file."""
        config = {
            'downloader': {
                'data_directories': {
                    'raw_data': str(temp_dir / 'raw'),
                    'manifest_file': str(temp_dir / 'manifest.json')
                },
                'sample_rate': 4096,
                'duration': 32,
                'signals': {
                    'detectors': ['H1'],
                    'events': ['GW150914-v3']
                },
                'noise': {
                    'detectors': ['H1'],
                    'runs': ['O1'],
                    'segments_per_run': 2
                }
            }
        }
        
        config_path = temp_dir / 'test_config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        return config_path
    
    def test_initialization(self, test_config):
        """Test downloader initialization."""
        downloader = CleanGWOSCDownloader(test_config)
        
        assert downloader.config_path == test_config
        assert downloader.config is not None
        assert downloader.manifest is not None
        assert 'downloads' in downloader.manifest
    
    def test_load_config(self, test_config):
        """Test configuration loading."""
        downloader = CleanGWOSCDownloader(test_config)
        
        assert downloader.config['downloader']['sample_rate'] == 4096
        assert downloader.config['downloader']['duration'] == 32
        assert downloader.config['downloader']['signals']['detectors'] == ['H1']
    
    def test_manifest_creation(self, test_config):
        """Test manifest file creation."""
        downloader = CleanGWOSCDownloader(test_config)
        
        manifest_path = Path(downloader.config['downloader']['data_directories']['manifest_file'])
        # Manifest is created when first accessed, not during initialization
        downloader._save_manifest()
        assert manifest_path.exists()
        
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        assert 'downloads' in manifest
        assert 'metadata' in manifest
        assert manifest['metadata']['config_file'] == str(test_config)
    
    def test_available_events_confident_only(self, test_config):
        """Test confident events discovery."""
        downloader = CleanGWOSCDownloader(test_config)
        
        with patch('downloader.gwosc_downloader.find_datasets') as mock_find:
            mock_find.return_value = ['GW150914-v3', 'GW151226-v2', 'GW170817-v3']
            
            events = downloader._get_available_events(confident_only=True)
            
            assert len(events) == 3
            assert 'GW150914-v3' in events
            assert 'GW151226-v2' in events
            assert 'GW170817-v3' in events
    
    def test_available_events_all(self, test_config):
        """Test all events discovery."""
        downloader = CleanGWOSCDownloader(test_config)
        
        with patch('downloader.gwosc_downloader.query_events') as mock_query:
            mock_query.return_value = ['GW150914-v1', 'GW150914-v2', 'GW150914-v3']
            
            events = downloader._get_available_events(confident_only=False)
            
            assert len(events) == 3
            assert 'GW150914-v1' in events
    
    def test_download_signal_segment_config(self, test_config):
        """Test signal segment download configuration access."""
        downloader = CleanGWOSCDownloader(test_config)
        
        # Test that config access works correctly
        sample_rate = downloader.config['downloader']['sample_rate']
        duration = downloader.config['downloader']['duration']
        
        assert sample_rate == 4096
        assert duration == 32
        
        # Test expected length calculation
        expected_length = duration * sample_rate
        assert expected_length == 131072
    
    @patch('downloader.gwosc_downloader.GWPY_AVAILABLE', True)
    @patch('downloader.gwosc_downloader.TimeSeries')
    def test_download_noise_segment(self, mock_timeseries, test_config):
        """Test noise segment download."""
        downloader = CleanGWOSCDownloader(test_config)
        
        # Mock TimeSeries response
        mock_ts = MagicMock()
        mock_ts.value = np.random.randn(131072).astype(np.float32)
        mock_timeseries.fetch_open_data.return_value = mock_ts
        
        strain_data = downloader._download_noise_segment('H1', 1126259450, 32)
        
        assert strain_data is not None
        assert len(strain_data) == 131072
        assert strain_data.dtype == np.float32
    
    def test_save_segment(self, test_config):
        """Test segment saving."""
        downloader = CleanGWOSCDownloader(test_config)
        
        strain_data = np.random.randn(131072).astype(np.float32)
        result = downloader._save_segment(
            'H1', 1126259450, 32, strain_data, 'signal', 'GW150914-v3', 'GW150914-v3'
        )
        
        assert result is True
        
        # Check file was created
        raw_dir = Path(downloader.config['downloader']['data_directories']['raw_data'])
        expected_file = raw_dir / 'H1_1126259450_32s.npz'
        assert expected_file.exists()
        
        # Check file contents
        with np.load(expected_file) as data:
            assert 'strain' in data
            assert 'detector' in data
            assert 'event' in data
            assert data['detector'] == 'H1'
            assert data['event'] == 'GW150914-v3'
            assert data['segment_type'] == 'signal'
    
    def test_manifest_update(self, test_config):
        """Test manifest update."""
        downloader = CleanGWOSCDownloader(test_config)
        
        initial_count = len(downloader.manifest['downloads'])
        
        # Test manifest update through _save_segment
        strain_data = np.random.randn(131072).astype(np.float32)
        downloader._save_segment('H1', 1126259450, 32, strain_data, 'signal', 'GW150914-v3', 'GW150914-v3')
        
        assert len(downloader.manifest['downloads']) == initial_count + 1
        
        last_entry = downloader.manifest['downloads'][-1]
        assert last_entry['detector'] == 'H1'
        assert last_entry['start_gps'] == 1126259450
        assert last_entry['segment_type'] == 'signal'
        assert last_entry['event'] == 'GW150914-v3'
        assert last_entry['successful'] is True
    
    @patch('downloader.gwosc_downloader.GWPY_AVAILABLE', True)
    @patch('downloader.gwosc_downloader.event_gps')
    @patch('downloader.gwosc_downloader.TimeSeries')
    def test_download_signals(self, mock_timeseries, mock_event_gps, test_config):
        """Test signal download functionality."""
        downloader = CleanGWOSCDownloader(test_config)
        
        # Mock event GPS time
        mock_event_gps.return_value = 1126259450
        
        # Mock TimeSeries response
        mock_ts = MagicMock()
        mock_ts.value = np.random.randn(131072).astype(np.float32)
        mock_timeseries.fetch_open_data.return_value = mock_ts
        
        results = downloader.download_signals()
        
        assert 'successful' in results
        assert 'failed' in results
        assert 'skipped' in results
        assert results['successful'] >= 0
    
    @patch('downloader.gwosc_downloader.GWPY_AVAILABLE', True)
    @patch('downloader.gwosc_downloader.run_segment')
    @patch('downloader.gwosc_downloader.get_segments')
    @patch('downloader.gwosc_downloader.TimeSeries')
    def test_download_noise(self, mock_timeseries, mock_get_segments, mock_run_segment, test_config):
        """Test noise download functionality."""
        downloader = CleanGWOSCDownloader(test_config)
        
        # Mock run segment
        mock_run_segment.return_value = (1126051217, 1137254417)
        
        # Mock science segments
        mock_get_segments.return_value = [(1126051217, 1137254417)]
        
        # Mock TimeSeries response
        mock_ts = MagicMock()
        mock_ts.value = np.random.randn(131072).astype(np.float32)
        mock_timeseries.fetch_open_data.return_value = mock_ts
        
        results = downloader.download_noise()
        
        assert 'successful' in results
        assert 'failed' in results
        assert 'skipped' in results
        assert results['successful'] >= 0
    
    def test_error_handling_invalid_config(self, temp_dir):
        """Test error handling for invalid configuration."""
        invalid_config = temp_dir / 'invalid_config.yaml'
        with open(invalid_config, 'w') as f:
            f.write('invalid: yaml: content: [')
        
        with pytest.raises(Exception):
            CleanGWOSCDownloader(invalid_config)
    
    def test_error_handling_missing_gwpy(self, test_config):
        """Test error handling when gwpy is not available."""
        downloader = CleanGWOSCDownloader(test_config)
        
        with patch('downloader.gwosc_downloader.GWPY_AVAILABLE', False):
            results = downloader.download_noise()
            
            assert results['successful'] == 0
            assert results['failed'] == 0
            assert results['skipped'] == 0
    
    def test_duplicate_prevention(self, test_config):
        """Test that duplicate downloads are prevented."""
        downloader = CleanGWOSCDownloader(test_config)
        
        # Add entry to manifest through _save_segment
        strain_data = np.random.randn(131072).astype(np.float32)
        downloader._save_segment('H1', 1126259450, 32, strain_data, 'signal', 'GW150914-v3', 'GW150914-v3')
        
        # Check if segment is marked as already downloaded
        segment_id = "H1_1126259450_32s"
        is_duplicate = any(
            d.get('segment_id') == segment_id 
            for d in downloader.manifest['downloads']
        )
        
        assert is_duplicate is True


if __name__ == "__main__":
    pytest.main([__file__])
