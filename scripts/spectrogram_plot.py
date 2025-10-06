#!/usr/bin/env python3
"""
Spectrogram visualization tool for gravitational wave data.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from spectrogram.npz_cwt_plot import main

if __name__ == "__main__":
    main()
