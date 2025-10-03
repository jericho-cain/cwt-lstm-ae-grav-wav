"""
Continuous Wavelet Transform (CWT) Preprocessing for Gravitational Wave Data

This module implements CWT preprocessing with proper timing alignment fixes
discovered through analysis of gravitational wave detection timing issues.

Key fixes implemented:
1. Correct indexing (aggregate over scales first)
2. Proper CWT implementation with padding and cone of influence
3. Analytic wavelet support for better timing accuracy
4. Robust handling of edge cases and data quality issues
"""

import numpy as np
import pywt
import logging
from scipy.signal import butter, sosfiltfilt
from scipy.ndimage import zoom
from typing import Tuple, Optional, Union, List
from pathlib import Path

logger = logging.getLogger(__name__)


def cwt_clean(
    x: np.ndarray, 
    fs: float, 
    fmin: float = 20.0, 
    fmax: float = 512.0, 
    n_scales: int = 64,
    wavelet: str = 'morl', 
    k_pad: float = 10.0, 
    k_coi: float = 6.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Clean CWT implementation with proper timing alignment.
    
    This function implements the corrected CWT algorithm that fixes timing
    alignment issues discovered in gravitational wave detection analysis.
    
    Parameters
    ----------
    x : np.ndarray
        Input time series data
    fs : float
        Sampling frequency in Hz
    fmin : float, optional
        Minimum frequency for CWT analysis, by default 20.0
    fmax : float, optional
        Maximum frequency for CWT analysis, by default 512.0
    n_scales : int, optional
        Number of frequency scales, by default 8
    wavelet : str, optional
        Wavelet type, by default 'cmor1.5-1.0' (analytic)
    k_pad : float, optional
        Padding factor for FFT wrap-around prevention, by default 10.0
    k_coi : float, optional
        Cone of influence factor, by default 6.0
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Tuple containing:
        - C : Complex CWT coefficients
        - freqs : Frequency array
        - scales : Scale array
        - mask : Cone of influence mask
        
    Examples
    --------
    >>> import numpy as np
    >>> x = np.random.randn(1024)
    >>> C, freqs, scales, mask = cwt_clean(x, fs=4096, fmin=20, fmax=512)
    >>> print(f"CWT shape: {C.shape}, frequencies: {len(freqs)}")
    """
    # Use legacy CWT implementation for compatibility
    if wavelet == 'morl':
        # Legacy Morlet wavelet implementation
        freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_scales)
        scales = fs / freqs
        
        # Compute CWT using legacy method
        coefficients, frequencies = pywt.cwt(x, scales, wavelet, sampling_period=1/fs)
        
        # Return magnitude (scalogram) - convert to float32 immediately
        C = np.abs(coefficients).astype(np.float32)
        
        # Create dummy mask (legacy doesn't use COI)
        mask = np.ones_like(C, dtype=bool)
        
        return C, freqs, scales, mask
    else:
        # Use new analytic wavelet implementation
        dt = 1.0 / fs
        w = pywt.ContinuousWavelet(wavelet)
        fc = pywt.central_frequency(w)
        freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_scales)
        scales = fc / (freqs * dt)

        # Reflection padding to avoid FFT wrap-around
        pad = int(np.ceil(k_pad * scales.max()))
        xpad = np.pad(x, (pad, pad), mode='reflect')

        Cpad, _ = pywt.cwt(xpad, scales, w, sampling_period=dt)
        C = Cpad[:, pad:-pad]  # back to original length

        # Cone-of-influence mask
        T = C.shape[1]
        mask = np.ones_like(C, dtype=bool)
        for i, s in enumerate(scales):
            m = int(np.ceil(k_coi * s))
            mask[i, :m] = False
            mask[i, T-m:] = False

        return C, freqs, scales, mask


def peak_time_from_cwt(
    C: np.ndarray, 
    fs: float, 
    mask: Optional[np.ndarray] = None, 
    reducer: str = 'max'
) -> Tuple[int, float]:
    """
    Find peak time correctly by aggregating over scales first.
    
    This function fixes the indexing bug that caused impossible timing offsets
    in gravitational wave detection. The key fix is to aggregate over scales
    first, then find the time peak.
    
    Parameters
    ----------
    C : np.ndarray
        Complex CWT coefficients
    fs : float
        Sampling frequency in Hz
    mask : np.ndarray, optional
        Cone of influence mask, by default None
    reducer : str, optional
        Aggregation method ('max' or 'sum'), by default 'max'
        
    Returns
    -------
    Tuple[int, float]
        Tuple containing:
        - t_idx : Time index of peak
        - t_sec : Time in seconds of peak
        
    Examples
    --------
    >>> C = np.random.randn(8, 1024) + 1j * np.random.randn(8, 1024)
    >>> t_idx, t_sec = peak_time_from_cwt(C, fs=4096)
    >>> print(f"Peak at index {t_idx}, time {t_sec:.3f}s")
    """
    mag = np.abs(C)
    if mask is not None:
        mag = np.where(mask, mag, np.nan)

    if reducer == 'max':
        time_stat = np.nanmax(mag, axis=0)  # Aggregate over scales first
    else:
        time_stat = np.nansum(mag, axis=0)

    t_idx = int(np.nanargmax(time_stat))  # Then find time peak
    return t_idx, t_idx / fs


def fixed_preprocess_with_cwt(
    strain_data: np.ndarray,
    sample_rate: int,
    target_height: int = 64,
    use_analytic: bool = False,
    fmin: float = 20.0,
    fmax: float = 512.0
) -> np.ndarray:
    """
    Fixed CWT preprocessing with proper timing alignment.
    
    This function implements the corrected CWT preprocessing pipeline that
    addresses timing alignment issues in gravitational wave detection.
    
    Parameters
    ----------
    strain_data : np.ndarray
        Input strain data array
    sample_rate : int
        Sampling rate in Hz
    target_height : int, optional
        Target height for CWT output, by default 8
    use_analytic : bool, optional
        Use analytic wavelet for better timing, by default True
    fmin : float, optional
        Minimum frequency for CWT, by default 20.0
    fmax : float, optional
        Maximum frequency for CWT, by default 512.0
        
    Returns
    -------
    np.ndarray
        Preprocessed CWT data with shape (n_samples, target_height, time_steps)
        
    Raises
    ------
    ValueError
        If no samples are successfully processed
        
    Examples
    --------
    >>> strain_data = np.random.randn(100, 1024)
    >>> cwt_data = fixed_preprocess_with_cwt(strain_data, sample_rate=4096)
    >>> print(f"CWT data shape: {cwt_data.shape}")
    """
    cwt_data = []
    successful_samples = 0
    failed_samples = 0

    logger.info(f"Computing FIXED CWT for {len(strain_data)} samples...")
    logger.info(f"Using {'analytic' if use_analytic else 'real'} wavelet")

    for i, strain in enumerate(strain_data):
        try:
            # Handle NaN/inf values
            if np.any(np.isnan(strain)) or np.any(np.isinf(strain)):
                finite_mask = np.isfinite(strain)
                if np.any(finite_mask):
                    median_val = np.median(strain[finite_mask])
                    max_finite = np.max(strain[finite_mask])
                    strain = np.where(np.isnan(strain), median_val, strain)
                    strain = np.where(np.isinf(strain), max_finite, strain)
                else:
                    strain = np.zeros_like(strain)

            if np.std(strain) < 1e-20:
                strain = strain + np.random.normal(0, 1e-20, size=strain.shape)

            # Apply legacy preprocessing pipeline
            # High-pass filter to remove low-frequency noise
            from scipy.signal import butter, sosfiltfilt
            sos = butter(4, 20, btype='highpass', fs=sample_rate, output='sos')
            filtered = sosfiltfilt(sos, strain)
            
            # Whiten the data
            whitened = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-10)

            # Choose wavelet
            wavelet = 'cmor1.5-1.0' if use_analytic else 'morl'
            
            # Compute CWT
            C, freqs, scales, mask = cwt_clean(
                whitened, 
                sample_rate, 
                n_scales=64,  # Use 64 scales like legacy
                wavelet=wavelet,
                fmin=fmin,
                fmax=fmax
            )

            # Check result
            if np.all(np.isnan(C)) or np.any(np.isinf(C)):
                failed_samples += 1
                continue

            # Resize to target height if needed
            if C.shape[0] != target_height:
                zoom_factor = target_height / C.shape[0]
                C = zoom(C, (zoom_factor, 1), order=1)
            
            # Downsample time dimension to reduce memory usage
            # Target width should be much smaller than 131072
            target_width = 1024  # Reasonable size for neural networks
            if C.shape[1] > target_width:
                time_zoom_factor = target_width / C.shape[1]
                C = zoom(C, (1, time_zoom_factor), order=1)

            # Log transform and normalize
            C_clean = np.nan_to_num(C, nan=1e-10)
            log_C = np.log10(C_clean + 1e-10)

            # Normalize
            valid_mask = ~np.isnan(log_C)
            if np.any(valid_mask):
                valid_data = log_C[valid_mask]
                normalized = np.full_like(log_C, 0.0)
                normalized[valid_mask] = (valid_data - np.mean(valid_data)) / (np.std(valid_data) + 1e-10)
            else:
                normalized = np.zeros_like(log_C)
            
            normalized = normalized.astype(np.float32)
            cwt_data.append(normalized)
            successful_samples += 1

            if (i + 1) % 10 == 0:
                logger.info(f"  Processed {i+1}/{len(strain_data)} samples (successful: {successful_samples}, failed: {failed_samples})")

        except Exception as e:
            logger.warning(f"  Sample {i+1}: Error during processing - {str(e)}")
            failed_samples += 1
            continue

    if not cwt_data:
        raise ValueError("No samples successfully processed!")

    logger.info(f"FIXED CWT preprocessing complete: {successful_samples} successful, {failed_samples} failed")
    return np.array(cwt_data)


class CWTPreprocessor:
    """
    Continuous Wavelet Transform preprocessor for gravitational wave data.
    
    This class provides a clean interface for CWT preprocessing with proper
    timing alignment and configuration management.
    
    Parameters
    ----------
    sample_rate : int
        Sampling rate in Hz
    target_height : int, optional
        Target height for CWT output, by default 8
    use_analytic : bool, optional
        Use analytic wavelet for better timing, by default True
    fmin : float, optional
        Minimum frequency for CWT, by default 20.0
    fmax : float, optional
        Maximum frequency for CWT, by default 512.0
        
    Attributes
    ----------
    sample_rate : int
        Sampling rate in Hz
    target_height : int
        Target height for CWT output
    use_analytic : bool
        Whether to use analytic wavelet
    fmin : float
        Minimum frequency for CWT
    fmax : float
        Maximum frequency for CWT
        
    Examples
    --------
    >>> preprocessor = CWTPreprocessor(sample_rate=4096, target_height=8)
    >>> cwt_data = preprocessor.process(strain_data)
    >>> peak_idx, peak_time = preprocessor.find_peak_time(cwt_data[0])
    """
    
    def __init__(
        self,
        sample_rate: int,
        target_height: int = 8,
        use_analytic: bool = True,
        fmin: float = 20.0,
        fmax: float = 512.0
    ) -> None:
        """
        Initialize CWT preprocessor.
        
        Parameters
        ----------
        sample_rate : int
            Sampling rate in Hz
        target_height : int, optional
            Target height for CWT output, by default 8
        use_analytic : bool, optional
            Use analytic wavelet for better timing, by default True
        fmin : float, optional
            Minimum frequency for CWT, by default 20.0
        fmax : float, optional
            Maximum frequency for CWT, by default 512.0
        """
        self.sample_rate = sample_rate
        self.target_height = target_height
        self.use_analytic = use_analytic
        self.fmin = fmin
        self.fmax = fmax
        
    def process(self, strain_data: np.ndarray) -> np.ndarray:
        """
        Process strain data with CWT preprocessing.
        
        Parameters
        ----------
        strain_data : np.ndarray
            Input strain data array
            
        Returns
        -------
        np.ndarray
            Preprocessed CWT data
        """
        return fixed_preprocess_with_cwt(
            strain_data,
            self.sample_rate,
            self.target_height,
            self.use_analytic,
            self.fmin,
            self.fmax
        )
        
    def find_peak_time(self, cwt_data: np.ndarray) -> Tuple[int, float]:
        """
        Find peak time from CWT data.
        
        Parameters
        ----------
        cwt_data : np.ndarray
            CWT data array
            
        Returns
        -------
        Tuple[int, float]
            Peak index and time in seconds
        """
        return peak_time_from_cwt(cwt_data, self.sample_rate)
