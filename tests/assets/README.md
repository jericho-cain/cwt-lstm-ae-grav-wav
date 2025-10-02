# Test Assets Directory

This directory contains test assets for the Gravitational Wave Hunter v2.0 project.

## Structure

```
tests/assets/
├── README.md                    # This file
├── test_config.yaml            # Test configuration file
├── test_data/                  # Test data directory
│   ├── raw/                   # Raw test data
│   └── processed/             # Processed test data
└── cwt_timing_validation_report.txt  # CWT timing validation results
```

## Files

### `test_config.yaml`
Test configuration file used by unit tests. Contains:
- Test detector configurations (H1, L1)
- Test GPS time ranges
- Test download parameters
- Safety settings disabled for testing

### `test_data/`
Directory for test data files:
- `raw/`: Raw gravitational wave data for testing
- `processed/`: Processed data from CWT preprocessing

### `cwt_timing_validation_report.txt`
Results from CWT timing validation testing with real GWOSC data.

## Usage

Tests automatically use these assets when running. The test configuration is designed to:
- Use temporary directories for test data
- Avoid conflicts with production data
- Provide consistent test environments
- Enable parallel test execution

## Maintenance

- Keep test data minimal and focused
- Update test config when adding new test scenarios
- Clean up temporary files after test runs
- Document any new test assets added
