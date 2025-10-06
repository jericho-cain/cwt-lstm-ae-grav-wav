#!/usr/bin/env python3
"""
Plot CWT scalogram from raw LIGO .npz files with optional GraceDB lookup.

This script loads raw gravitational wave strain data from .npz files,
applies CWT preprocessing, and generates publication-quality spectrograms
with optional GraceDB event information lookup.

Usage:
    python npz_cwt_plot.py H1_1126259462_32s.npz --lookup --save gw150914.png
    python npz_cwt_plot.py H1_1167559936_32s.npz --t0 2.5 --scales 256

Functions
---------
load_npz_any : Load strain data from .npz files
grace_lookup_t0 : Lookup event timing from GraceDB
whiten_gwpy_if_possible : Apply GWPy whitening if available
whiten_psd : Apply PSD-based whitening
cwt_compute : Compute Continuous Wavelet Transform
auto_peak_time_from_cwt : Auto-detect peak time from CWT energy
pretty_logfreq_axis : Style log-frequency axis with clean labels
needs_zoom : Determine if zoom panel is needed based on signal contrast
plot_cwt : Main plotting function with smart layout selection
main : Command-line interface

Examples
--------
>>> # Load and plot GW150914 with GraceDB lookup
>>> strain, fs, meta = load_npz_any("H1_1126259462_32s.npz")
>>> scalogram, freqs, scales = cwt_compute(strain, fs)
>>> plot_cwt(scalogram, freqs, 32.0, mode="auto", save="gw150914.png")

Command Line Examples
---------------------
# Working example for GW150914 with dual-panel layout and GraceDB lookup:
python scripts/spectrogram_plot.py data/raw/H1_1126259462_32s.npz --lookup --mode dual --scales 256 --window 4.0 --zoom_ms 1000 --save results/visualizations/spectrograms/gw150914_dual.png

# Auto mode (smart layout selection based on signal strength):
python scripts/spectrogram_plot.py data/raw/H1_1126259462_32s.npz --lookup --mode auto --scales 256 --window 4.0 --save gw150914_auto.png

# Single panel layout:
python scripts/spectrogram_plot.py data/raw/H1_1126259462_32s.npz --lookup --mode single --scales 256 --window 4.0 --save gw150914_single.png
"""

import re
import json
import argparse
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Optional deps (used if available)
try:
    from gwpy.timeseries import TimeSeries  # for easy whitening if you have gwpy
    HAVE_GWPY = True
except Exception:
    HAVE_GWPY = False

import pywt
from scipy.signal import welch, butter, sosfiltfilt

# ---------- Utilities ----------

def parse_filename(meta_path: Path):
    """Extract detector and gps from filenames like H1_1376004896_32s.npz."""
    m = re.search(r'(?P<ifo>[HLVIK]\d)_(?P<gps>\d+)', meta_path.stem)
    det = m.group('ifo') if m else None
    gps = float(m.group('gps')) if m else None
    return det, gps

def load_npz_any(path: Path) -> Tuple[np.ndarray, float, dict]:
    """
    Load strain data from .npz files with flexible key detection.
    
    Parameters
    ----------
    path : Path
        Path to the .npz file containing strain data
        
    Returns
    -------
    strain : ndarray
        Gravitational wave strain time series data
    fs : float
        Sampling frequency in Hz
    meta : dict
        Metadata dictionary containing file information
        
    Notes
    -----
    This function automatically detects the correct keys in .npz files
    by looking for common patterns like 'strain', 'data', 'h1', etc.
    It handles both single-detector and multi-detector files.
    
    Examples
    --------
    >>> strain, fs, meta = load_npz_any("H1_1126259462_32s.npz")
    >>> print(f"Loaded {len(strain)} samples at {fs} Hz")
    """
    d = np.load(path, allow_pickle=True)
    keys = {k.lower(): k for k in d.files}

    # try common keys for strain
    for cand in ["strain", "data", "h", "timeseries", "y", "x"]:
        if cand in keys:
            strain = np.asarray(d[keys[cand]]).astype(float).ravel()
            break
    else:
        # if single array inside
        if len(d.files) == 1:
            strain = np.asarray(d[list(d.files)[0]]).astype(float).ravel()
        else:
            raise ValueError(f"Couldn't find strain in {path.name} (found keys={list(d.files)})")

    # sample rate
    fs = None
    for cand in ["fs", "sample_rate", "sampling_rate", "sr", "rate", "fs_hz"]:
        if cand in keys:
            fs = float(d[keys[cand]])
            break
    if fs is None:
        # fallback to common LIGO rate
        fs = 4096.0

    return strain, fs, dict(d)

def whiten_psd(strain: np.ndarray, fs: float, band=(20.0, 512.0)):
    """Simple PSD-whitening + bandpass with Welch."""
    nperseg = int(min(max(1, fs//2), 4*fs))  # ~2–4 s windows
    freqs, Pxx = welch(strain, fs=fs, nperseg=nperseg, detrend='linear', scaling='density')
    # Avoid zeros
    Pxx = np.maximum(Pxx, np.percentile(Pxx, 0.1))
    # FFT -> divide by sqrt(PSD) -> iFFT
    fft = np.fft.rfft(strain)
    # Interpolate PSD to FFT bins
    psd_interp = np.interp(np.fft.rfftfreq(len(strain), 1/fs), freqs, Pxx)
    white_fft = fft / np.sqrt(psd_interp)
    white = np.fft.irfft(white_fft, n=len(strain))

    # band-limit (for display clarity)
    sos = butter(4, band, btype='band', fs=fs, output='sos')
    white = sosfiltfilt(sos, white)
    return white

def whiten_gwpy_if_possible(strain: np.ndarray, fs: float):
    if not HAVE_GWPY:
        return None
    # Create a TimeSeries at the given fs
    ts = TimeSeries(strain, sample_rate=fs)
    try:
        w = ts.whiten()
        return np.asarray(w.value)
    except Exception:
        return None

def cwt_compute(x: np.ndarray, fs: float, fmin: float = 20.0, fmax: float = 512.0, 
                n_scales: int = 128, wavelet: str = "cmor1.5-1.0") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Continuous Wavelet Transform of strain data.
    
    Parameters
    ----------
    x : ndarray
        Input strain time series data
    fs : float
        Sampling frequency in Hz
    fmin : float, optional
        Minimum frequency in Hz (default: 20.0)
    fmax : float, optional
        Maximum frequency in Hz (default: 512.0)
    n_scales : int, optional
        Number of frequency scales (default: 128)
    wavelet : str, optional
        Wavelet type for CWT (default: "cmor1.5-1.0")
        
    Returns
    -------
    scalogram : ndarray
        CWT coefficients as complex array (n_scales, n_time)
    freqs : ndarray
        Frequency array corresponding to scales
    scales : ndarray
        Wavelet scales used for CWT
        
    Notes
    -----
    Uses PyWavelets for CWT computation with complex Morlet wavelet.
    The scalogram preserves both amplitude and phase information
    critical for gravitational wave detection.
    
    Examples
    --------
    >>> scalogram, freqs, scales = cwt_compute(strain, 4096, fmin=20, fmax=512)
    >>> print(f"CWT shape: {scalogram.shape}, frequencies: {freqs[0]:.1f}-{freqs[-1]:.1f} Hz")
    """
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_scales)
    scales = fs / freqs
    coeffs, _ = pywt.cwt(x, scales=scales, wavelet=wavelet, sampling_period=1/fs)
    # magnitude for plotting
    scalogram = np.abs(coeffs).astype(np.float32)
    return scalogram, freqs, scales

def auto_peak_time_from_cwt(scalogram: np.ndarray, fs_time: float):
    """Return t0 in seconds using max-energy column."""
    energy_t = scalogram.sum(axis=0)
    idx = int(np.argmax(energy_t))
    return idx / fs_time  # fs_time is 'pixels per second' along time axis

def grace_lookup_t0(det: str, gps: float, pad=10.0) -> Tuple[Optional[str], Optional[float]]:
    """
    Lookup gravitational wave event timing from GraceDB.
    
    Parameters
    ----------
    det : str
        Detector identifier (e.g., 'H1', 'L1', 'V1')
    gps : float
        GPS timestamp to search around
    pad : float, optional
        Time window in seconds to search around GPS time (default: 10.0)
        
    Returns
    -------
    name : str or None
        Event name from GraceDB (e.g., 'GW150914')
    t0 : float or None
        Event time relative to GPS timestamp
        
    Notes
    -----
    This function queries the GraceDB API to find confirmed gravitational
    wave events near the specified GPS time. Requires internet connection.
    
    Examples
    --------
    >>> name, t0 = grace_lookup_t0("H1", 1126259462.4)
    >>> print(f"Found event {name} at t0={t0:.3f}s")
    """
    import requests  # only if user passed --lookup
    base = "https://gracedb.ligo.org/apiweb/superevents/?format=json"
    # filter by time window around gps
    try:
        r = requests.get(base, timeout=20)
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception:
        return None, None

    best = None
    best_dt = 1e9
    for se in results:
        t0 = se.get("t_0")
        if t0 is None:
            continue
        dt = abs(t0 - gps)
        if dt < pad and dt < best_dt:
            best_dt = dt
            best = (se.get("superevent_id"), float(t0))
    return best if best else (None, None)

def default_ticks(fmin, fmax):
    cands = np.array([20,30,40,50,60,75,100,120,150,200,256,300,400,512,700,1000], float)
    return cands[(cands >= fmin) & (cands <= fmax)]

def pretty_logfreq_axis(ax, fmin=20, fmax=512, sparse=False):
    """Tidy log-frequency axis (no squishing, human labels)."""
    import matplotlib.ticker as mticker
    
    # Choose clean major ticks you actually want to see
    all_ticks = [20, 30, 40, 50, 60, 75, 100, 120, 150, 200, 256, 300, 400, 512]
    ticks = all_ticks[::2] if sparse else all_ticks  # every other tick when sparse
    ax.set_yscale("log")
    ax.set_ylim(fmin, fmax)

    # Major ticks: fixed positions, plain integers
    ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mticker.FixedFormatter([str(int(v)) for v in ticks]))

    # Minor ticks: keep tick marks, but **remove labels** to prevent "502" etc.
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=(2,3,4,5,6,7,8,9)))
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())

    # Spacing/legibility
    ax.tick_params(axis="y", which="major", length=4, pad=4)
    ax.tick_params(axis="y", which="minor", length=2)
    ax.grid(False)  # optional: grids often add clutter on log axes

def needs_zoom(L, t0, duration, win=0.3, threshold=0.6):
    """Determine if zoom panel is needed based on signal contrast."""
    nT = L.shape[1]
    dt = duration / nT
    j0 = int(t0 / dt)
    w = int(win / dt)
    
    # Extract window around t0
    sl = L[:, max(0, j0-w):min(nT, j0+w)]
    
    # Compute contrast metric: 99th - 90th percentile
    contrast = np.percentile(sl, 99) - np.percentile(sl, 90)
    
    return contrast < threshold

def plot_cwt(scalogram: np.ndarray, freqs: np.ndarray, duration: float, 
             t0: Optional[float] = None, zoom_ms: float = 250, title: str = "", 
             save: Optional[str] = None, clip_pct: float = 99.0, mode: str = "auto", 
             window_start: float = 0.0) -> float:
    """
    Plot CWT spectrogram with smart layout selection.
    
    Parameters
    ----------
    scalogram : ndarray
        CWT scalogram data (n_scales, n_time)
    freqs : ndarray
        Frequency array corresponding to scalogram rows
    duration : float
        Total duration of the time series in seconds
    t0 : float, optional
        Detection time in seconds from start. If None, auto-detected from CWT energy
    zoom_ms : float, optional
        Zoom window size in milliseconds (default: 250)
    title : str, optional
        Plot title (default: "")
    save : str, optional
        Output filename for saving plot (default: None)
    clip_pct : float, optional
        Percentile for contrast clipping (default: 99.0)
    mode : str, optional
        Layout mode: "auto" (smart selection), "single" (one panel), "dual" (two panels)
        
    Returns
    -------
    t0 : float
        Detection time used for plotting
        
    Notes
    -----
    The function automatically chooses between single and dual panel layouts
    based on signal contrast when mode="auto". Strong signals get clean single
    panels, while weak signals get dual panels with zoom for detailed analysis.
    
    Examples
    --------
    >>> plot_cwt(scalogram, freqs, 32.0, mode="auto", save="gw150914.png")
    >>> plot_cwt(scalogram, freqs, 32.0, t0=2.0, mode="dual", zoom_ms=500)
    """
    n_scales, n_time = scalogram.shape
    # time axis: assume uniform sampling across 'duration'
    dt = duration / n_time
    extent = [0.0, duration, freqs[0], freqs[-1]]

    # Display-friendly normalization: row-wise z-score
    L = np.log10(scalogram + 1e-6)
    L = (L - L.mean(axis=1, keepdims=True)) / (L.std(axis=1, keepdims=True) + 1e-6)

    # Modern LIGO style colormap and contrast
    cmap = "turbo"  # classic blue→green→yellow→red
    v_max = np.percentile(L, 99.3)  # aggressive but clean
    v_min = v_max - 2.2  # reduce glare; tweak 1.8–2.5 as needed

    # Auto-detect t0 if not provided
    if t0 is None:
        t0 = auto_peak_time_from_cwt(scalogram, fs_time=n_time/duration)

    # Determine layout mode
    if mode == "auto":
        use_zoom = needs_zoom(L, t0, duration)
    elif mode == "single":
        use_zoom = False
    elif mode == "dual":
        use_zoom = True
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'auto', 'single', or 'dual'")

    if use_zoom:
        # Dual panel layout
        fig = plt.figure(constrained_layout=False, figsize=(12,7.6))
        gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.0], hspace=0.18)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharey=ax1)  # share y to keep ticks identical, independent x
        
        # full view
        im1 = ax1.imshow(L, origin="lower", aspect="auto",
                         extent=extent, vmin=v_min, vmax=v_max, cmap=cmap)
        pretty_logfreq_axis(ax1, 20, 512, sparse=False)  # full panel: all ticks
        ax1.set_ylabel("Frequency [Hz]")
        ax1.set_title(f"{title} - CWT Spectrogram (full {duration:.3f} s)", pad=6)

        # zoom view
        half = zoom_ms/1000.0
        t_left, t_right = max(0.0, t0-half), min(duration, t0+half)
        
        im2 = ax2.imshow(L, origin="lower", aspect="auto",
                         extent=extent, vmin=v_min, vmax=v_max, cmap=cmap)
        ax2.set_xlim(t_left, t_right)
        pretty_logfreq_axis(ax2, 20, 512, sparse=True)  # zoom panel: every other tick
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Frequency [Hz]")
        
        # Convert tick labels to absolute time for both panels
        fmt = FuncFormatter(lambda val, pos: f"{window_start + val:.1f}")
        ax1.xaxis.set_major_formatter(fmt)
        ax2.xaxis.set_major_formatter(fmt)

        # give the zoom panel its own small title, lifted slightly to avoid clash
        ax2.set_title(f"Zoom ±{zoom_ms:.0f} ms around t0 = {t0:.3f} s", y=1.02, pad=2, fontsize=11)


        # colorbar for the whole figure, aligned with top axes
        cbar = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
        cbar.set_label("Enhanced Log |CWT|")

        # red tick markers
        ax1.plot([t0], [22], marker="v", color="red", markersize=7, clip_on=False)
        ax2.plot([t0], [22], marker="v", color="red", markersize=7, clip_on=False)

        # extra left margin if labels feel tight
        fig.subplots_adjust(left=0.11, right=0.92, top=0.92, bottom=0.09)
        
    else:
        # Single panel layout
        fig, ax = plt.subplots(figsize=(12, 7.6))
        im = ax.imshow(L, origin="lower", aspect="auto",
                       extent=extent, vmin=v_min, vmax=v_max, cmap=cmap)
        pretty_logfreq_axis(ax, 20, 512, sparse=False)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(f"{title} - CWT Spectrogram ({duration:.3f} s)")
        
        # Convert tick labels to absolute time
        fmt = FuncFormatter(lambda val, pos: f"{window_start + val:.1f}")
        ax.xaxis.set_major_formatter(fmt)
        
        # bottom tick marker at t0
        ax.plot([t0], [22], marker="v", color="red", markersize=7, clip_on=False)
        
        # colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Enhanced Log |CWT|")
        
        fig.tight_layout()
    
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=160, bbox_inches="tight")
        print(f"Saved: {save}")
    else:
        plt.show()
    return t0

# ---------- Main ----------

def main() -> None:
    """
    Command-line interface for CWT spectrogram visualization.
    
    This function provides a command-line interface for loading gravitational
    wave data from .npz files and generating publication-quality spectrograms
    with optional GraceDB event lookup.
    
    Command Line Arguments
    ---------------------
    npz : str
        Path to .npz file containing strain data
    --duration : float, optional
        Total duration in seconds (default: auto-detect)
    --t0 : float, optional
        Detection time in seconds (default: auto-detect or GraceDB lookup)
    --lookup : bool, optional
        Enable GraceDB event lookup (default: False)
    --fmin : float, optional
        Minimum frequency in Hz (default: 20.0)
    --fmax : float, optional
        Maximum frequency in Hz (default: 512.0)
    --scales : int, optional
        Number of CWT scales (default: 128)
    --wavelet : str, optional
        Wavelet type (default: "cmor1.5-1.0")
    --zoom_ms : float, optional
        Zoom window in milliseconds (default: 250)
    --clip : float, optional
        Contrast clipping percentile (default: 99.0)
    --save : str, optional
        Output filename (default: auto-generate)
    --mode : str, optional
        Layout mode: "auto", "single", or "dual" (default: "auto")
    --window : float, optional
        Analysis window in seconds around t0 (default: None)
        
    Examples
    --------
    >>> # Basic usage with auto-detection
    >>> python npz_cwt_plot.py H1_1126259462_32s.npz
    >>> 
    >>> # With GraceDB lookup and custom output
    >>> python npz_cwt_plot.py H1_1126259462_32s.npz --lookup --save gw150914.png
    >>> 
    >>> # High-resolution dual panel
    >>> python npz_cwt_plot.py H1_1167559936_32s.npz --scales 256 --mode dual
    """
    ap = argparse.ArgumentParser(description="Plot CWT spectrograms from gravitational wave data")
    ap.add_argument("npz", type=str, help="Path to .npz (e.g., H1_<gps>_32s.npz)")
    ap.add_argument("--duration", type=float, default=None,
                    help="Total seconds represented in file (default: parsed from filename or len/FS)")
    ap.add_argument("--t0", type=float, default=None,
                    help="Detection time in seconds from start (if omitted: GraceDB lookup or auto by max-energy).")
    ap.add_argument("--lookup", action="store_true",
                    help="Attempt GraceDB lookup near GPS from filename (internet required).")
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=512.0)
    ap.add_argument("--scales", type=int, default=128,
                    help="Number of CWT scales (64/128/256 recommended).")
    ap.add_argument("--wavelet", type=str, default="cmor1.5-1.0")
    ap.add_argument("--zoom_ms", type=float, default=250.0)
    ap.add_argument("--clip", type=float, default=99.0, help="Percentile for color clipping.")
    ap.add_argument("--save", type=str, default=None, help="Path to save PNG. Default: <npz>_cwt.png")
    ap.add_argument("--mode", type=str, default="auto", choices=["auto", "single", "dual"], 
                    help="Layout mode: auto (smart selection), single (one panel), dual (two panels)")
    ap.add_argument("--window", type=float, default=None, help="Analysis window in seconds around t0 (e.g., 4.0 for ±2s)")
    args = ap.parse_args()

    path = Path(args.npz)
    strain, fs, meta = load_npz_any(path)

    det, gps = parse_filename(path)

    # Duration
    if args.duration is not None:
        duration = float(args.duration)
    elif "duration" in {k.lower() for k in meta.keys()}:
        # honor stored duration if present
        for k in meta.keys():
            if k.lower() == "duration":
                duration = float(meta[k])
                break
    else:
        duration = len(strain) / fs  # infer from data

    # Optional: Extract window around signal for better contrast
    window_start = 0.0
    if args.window is not None:
        # First get rough t0 estimate from full data
        w_full = whiten_gwpy_if_possible(strain, fs)
        if w_full is None:
            w_full = whiten_psd(strain, fs, band=(args.fmin, args.fmax))
        
        scalogram_full, _, _ = cwt_compute(w_full, fs, fmin=args.fmin, fmax=args.fmax,
                                          n_scales=args.scales, wavelet=args.wavelet)
        t0_rough = auto_peak_time_from_cwt(scalogram_full, fs_time=scalogram_full.shape[1] / duration)
        
        # Extract window around t0
        window_samples = int(args.window * fs)
        start_idx = max(0, int((t0_rough - args.window/2) * fs))
        end_idx = min(len(strain), start_idx + window_samples)
        
        strain_window = strain[start_idx:end_idx]
        window_start = start_idx / fs  # Calculate the start time of the window
        duration = len(strain_window) / fs
        print(f"Extracted {args.window}s window around t0={t0_rough:.3f}s, window starts at {window_start:.3f}s")
        
        # Re-whiten the windowed data
        w = whiten_gwpy_if_possible(strain_window, fs)
        if w is None:
            w = whiten_psd(strain_window, fs, band=(args.fmin, args.fmax))
    else:
        # Whitening
        w = whiten_gwpy_if_possible(strain, fs)
        if w is None:
            w = whiten_psd(strain, fs, band=(args.fmin, args.fmax))

    # Compute CWT
    scalogram, freqs, scales = cwt_compute(w, fs, fmin=args.fmin, fmax=args.fmax,
                                           n_scales=args.scales, wavelet=args.wavelet)

    # Determine t0
    t0 = args.t0
    grace_event_name = None
    if t0 is None and args.lookup and gps is not None:
        name, t0_abs = grace_lookup_t0(det or "", gps)
        if t0_abs is not None:
            # Map absolute GPS to seconds from start of this segment
            # If filename encodes a 32 s segment starting at 'gps', put t0 relative to that.
            # Common convention: files like H1_<gps>_32s.npz start at that <gps>.
            t0 = float(t0_abs - gps)
            grace_event_name = name
            print(f"GraceDB: {name} at GPS {t0_abs} (t0={t0:.3f}s from segment start)")
    if t0 is None:
        # fallback: auto from CWT energy
        t0 = auto_peak_time_from_cwt(scalogram, fs_time=scalogram.shape[1] / duration)
        print(f"Auto-detected t0={t0:.3f}s (max CWT energy)")

    # Save path
    save_path = args.save or str(path.with_suffix("").as_posix() + f"_cwt.png")

    title_bits = []
    if grace_event_name: title_bits.append(grace_event_name)
    if det: title_bits.append(det)
    if gps: title_bits.append(str(int(gps)))
    title_bits.append(f"{args.scales} scales")
    title = " ".join(title_bits)

    plot_cwt(scalogram, freqs, duration, t0=t0, zoom_ms=args.zoom_ms,
             title=title, save=save_path, clip_pct=args.clip, mode=args.mode, 
             window_start=window_start)

if __name__ == "__main__":
    main()
