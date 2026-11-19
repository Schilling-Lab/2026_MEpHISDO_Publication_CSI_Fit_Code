# ============================================================
# Multi-peak automatic CSI fitter
# ============================================================
#
# Per voxel and per user-defined frequency interval, at most one peak
# is detected in the line-broadened magnitude spectrum. All detected
# peaks are then fitted together in the time domain with one
# exponential each (LMFit). Missing or rejected peaks are NaN/0 in
# the output.
# ============================================================

import numpy as np
import xarray as xr
from dataclasses import dataclass
from typing import Optional, List
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Optional: sliders (gracefully degrades if ipywidgets is missing)
try:
    from ipywidgets import Checkbox, FloatSlider, HBox, IntSlider
    from IPython.display import display
    _WIDGETS_OK = True
except Exception:
    _WIDGETS_OK = False


# ------------------------------------------------------------
# Dataclass: DetectedPeak
# ------------------------------------------------------------
@dataclass
class DetectedPeak:
    """Container for a single detected peak inside a search interval."""
    freq_hz: float       # initial frequency guess
    amp: complex         # initial amplitude guess
    t2_s: float          # initial T2* guess
    snr: float           # estimated SNR


def compute_peak_width(f_i, m_i, idx, valley_window=5, frac_height=0.5):
    """
    Compute spectral peak width at 'fraction' height (default: FWHM).
    Returns width in Hz.
    Allows asymmetric peaks (e.g., small peaks near the flank of a large one).

    - f_i: frequency axis (interval)
    - m_i: spectrum magnitude (interval)
    - idx: index of detected peak
    - valley_window: number of points to search for local minima on each side
    - frac_height: threshold fraction relative to peak-minimum difference

    Returns:
        width_left_hz, width_right_hz, width_total_hz
    """

    n = len(m_i)
    m_peak = m_i[idx]

    # --- 1) local valley left ---
    left_min = np.min(m_i[max(0, idx - valley_window):idx]) if idx > 0 else m_i[idx]
    # --- 2) local valley right ---
    right_min = np.min(m_i[idx:min(n, idx + valley_window)]) if idx < n-1 else m_i[idx]

    # dynamic half-height thresholds relative to valleys
    thr_left = left_min + frac_height * (m_peak - left_min)
    thr_right = right_min + frac_height * (m_peak - right_min)

    # --- 3) find left boundary ---
    left = idx
    while left > 0 and m_i[left] > thr_left:
        left -= 1

    # --- 4) find right boundary ---
    right = idx
    while right < n - 1 and m_i[right] > thr_right:
        right += 1

    width_left_hz = abs(f_i[idx] - f_i[left])
    width_right_hz = abs(f_i[right] - f_i[idx])
    width_total_hz = width_left_hz + width_right_hz

    return width_left_hz, width_right_hz, width_total_hz


def _apply_exponential_line_broadening(
    fid,
    dwell=None,
    lb_hz=0.0,
    t_axis=None,
    line_broadening_hz=None,
):
    """
    Apply exponential line broadening / apodization.

    Supports both call styles:

    1) Using dwell time:
        _apply_exponential_line_broadening(fid, dwell, lb_hz)

    2) Using an explicit time axis:
        _apply_exponential_line_broadening(
            fid=fid,
            t_axis=t_axis,
            line_broadening_hz=lb_hz,
        )

    The window is:

        exp(-pi * LB * t)

    where LB is in Hz and t is in seconds.
    """

    fid = np.asarray(fid)

    # Support old keyword name
    if line_broadening_hz is not None:
        lb_hz = line_broadening_hz

    # Robust scalar conversion
    if lb_hz is None:
        lb_hz = 0.0

    lb_hz = float(lb_hz)

    if lb_hz <= 0:
        return fid

    # Prefer explicit time axis if available
    if t_axis is not None:
        t = np.asarray(t_axis, dtype=float)
        t = t - float(t[0])

    # Otherwise construct time axis from dwell
    elif dwell is not None:
        t = np.arange(len(fid), dtype=float) * float(dwell)

    else:
        raise ValueError(
            "Line broadening in Hz requires either `dwell` or `t_axis`. "
            "If neither is available, the time scale is unknown."
        )

    window = np.exp(-np.pi * lb_hz * t)

    return fid * window


# ------------------------------------------------------------
# Peak Detection: detect_peak_in_interval
# ------------------------------------------------------------

def detect_peak_in_interval(
    fid,
    t_axis,
    fmin,
    fmax,
    noise_sigma,
    snr_threshold=3.0,
    kappa_curvature = 2.0,
    min_prominence_threshold = None,
    min_fwhm_hz: float = 16.0,
    detection_line_broadening_hz: float = 5.0,
):
    """
    Detect at most one peak inside the frequency interval [fmin, fmax].

    Detection runs on the 2x zero-filled magnitude spectrum, apodized with
    detection_line_broadening_hz. Candidates are
      - zero crossings of the first derivative, and
      - the point of most negative curvature, if the second derivative there is
        below -kappa_curvature x std of the second derivative in the interval.
    A candidate is valid if
      - its prominence after subtracting a local linear baseline (±10 bins) is
        >= min_prominence_threshold (default: 2 x spectral noise std),
      - its FWHM (left + right half width) is >= min_fwhm_hz, and
      - its SNR (height / spectral noise std) is >= snr_threshold.
    The valid candidate with the highest SNR is returned.

    noise_sigma is the time-domain noise std per channel (real/imag), as returned by
    hypermri.utils.physics.estimate_noise.

    Returns:
        DetectedPeak or None
    """

    # -----------------------------
    # FFT with zero-padding
    # Detection spectrum is apodized to reduce truncation/ringing artifacts.
    # The original, unbroadened spectrum is kept for amplitude seeding.
    # -----------------------------
    n = fid.size
    zp = 2
    nz = zp * n
    dwell = float(t_axis[1] - t_axis[0])

    # Unbroadened zero-filled spectrum
    fid_z_raw = np.zeros(nz, dtype=fid.dtype)
    fid_z_raw[:n] = fid
    spec_raw = np.fft.fftshift(np.fft.fft(fid_z_raw))
    mag_raw = np.abs(spec_raw)

    # Noise std of the apodized spectrum: `noise_sigma` is the time-domain noise std
    # per channel; the FFT scales it by the l2-norm of the apodization window
    # (zero-filling adds no noise).
    window = _apply_exponential_line_broadening(
        fid=np.ones(n),
        t_axis=t_axis,
        line_broadening_hz=detection_line_broadening_hz,
    )
    noise_det = noise_sigma * np.sqrt(np.sum(window**2))

    # Broadened/apodized zero-filled spectrum for detection
    fid_det = _apply_exponential_line_broadening(
        fid=fid,
        t_axis=t_axis,
        line_broadening_hz=detection_line_broadening_hz,
    )

    fid_z = np.zeros(nz, dtype=fid.dtype)
    fid_z[:n] = fid_det

    spec = np.fft.fftshift(np.fft.fft(fid_z))
    freq = np.fft.fftshift(np.fft.fftfreq(nz, d=dwell))
    mag = np.abs(spec)

    # -----------------------------
    # Restrict to interval
    # -----------------------------
    mask = (freq >= fmin) & (freq <= fmax)
    if not np.any(mask):
        return None

    first_idx = np.flatnonzero(mask)[0]  # index of the interval start in the full spectrum
    f_i = freq[mask]
    m_i = mag[mask]
    m_i_raw = mag_raw[mask]  # unbroadened spectrum for amplitude seed

    # -----------------------------
    # Candidate collection
    # -----------------------------
    candidates = set()

    d1 = np.gradient(m_i)
    d2 = np.gradient(d1)

    # Point of most negative curvature (interior points only), if below the threshold
    kappa_thr = np.std(d2) * kappa_curvature
    interior_indices = np.arange(1, len(d2) - 1)
    curv_candidates = interior_indices[d2[interior_indices] < -kappa_thr]
    if curv_candidates.size > 0:
        candidates.add(int(curv_candidates[np.argmin(d2[curv_candidates])]))

    # Zero crossings of the first derivative
    for i in np.flatnonzero(np.diff(np.sign(d1))):
        candidates.add(int(i))

    if len(candidates) == 0:
        return None

    # -----------------------------
    # Evaluate candidates
    # -----------------------------
    if min_prominence_threshold is not None:
        prominence_thr = min_prominence_threshold
    else:
        prominence_thr = noise_det * 2

    best_idx = None
    best_snr = -np.inf

    for idx in candidates:
        idx_of_peak = first_idx + idx

        # Prominence after subtracting a linear baseline fitted to ±10 bins
        win_start = max(0, idx_of_peak - 10)
        win_end = min(len(mag), idx_of_peak + 10)
        x = freq[win_start:win_end]
        y = mag[win_start:win_end]
        y_corr = y - np.polyval(np.polyfit(x, y, deg=1), x)

        center = idx_of_peak - win_start
        left_min = y_corr[:center].min()
        right_min = y_corr[center:].min()
        prominence_corrected = y_corr[center] - max(left_min, right_min)
        if prominence_corrected < prominence_thr:
            continue

        # Minimal accepted peak width (full width = left + right half width)
        _, _, width_hz = compute_peak_width(
            freq, mag, idx_of_peak,
            valley_window=10,
            frac_height=0.5,
        )
        if width_hz < min_fwhm_hz:
            continue

        snr = float(m_i[idx] / noise_det)
        if snr < snr_threshold:
            continue

        if snr > best_snr:
            best_snr = snr
            best_idx = idx

    if best_idx is None:
        return None

    # Use the unbroadened magnitude as the amplitude seed, because the
    # apodized spectrum is intentionally damped for detection.
    return DetectedPeak(
        freq_hz=float(f_i[best_idx]),
        amp=float(m_i_raw[best_idx]),
        t2_s=0.013,
        snr=best_snr,
    )


# ------------------------------------------------------------
# Time‑domain exponential model for LMFit
# ------------------------------------------------------------
def _exp_fid_model(t, amps, freqs, t2s):
    """
    Multi-exponential complex FID model.

    Args:
        t: time axis (1D)
        amps: array of complex amplitudes
        freqs: array of frequencies (Hz)
        t2s: array of T2* values (s)

    Returns:
        Complex FID (1D)
    """
    out = np.zeros_like(t, dtype=np.complex128)
    for A, f, T2 in zip(amps, freqs, t2s):
        out += A * np.exp(1j * 2 * np.pi * f * t) * np.exp(-t / T2)
        
    return out

# ------------------------------------------------------------
# Sub-bin parabolic interpolation around a spectral maximum
# ------------------------------------------------------------
def _parabolic_subbin(freq_win: np.ndarray, mag_win: np.ndarray, idx_peak: int):
    """
    Quadratic interpolation of a peak position using the point and its immediate neighbors.
    Returns (f_refined, mag_refined). If neighbors are not available, returns the bin center.
    """
    n = len(mag_win)
    if idx_peak <= 0 or idx_peak >= n - 1:
        return float(freq_win[idx_peak]), float(mag_win[idx_peak])

    # Fit a parabola y = a x^2 + b x + c to three points (x: -1, 0, +1 around the peak)
    y1, y2, y3 = mag_win[idx_peak - 1], mag_win[idx_peak], mag_win[idx_peak + 1]
    # Vertex offset from the center bin in "bin units"
    denom = (y1 - 2*y2 + y3)
    if np.abs(denom) < 1e-20:
        delta = 0.0
    else:
        delta = 0.5 * (y1 - y3) / denom  # in bins

    # Clamp delta to [-1, 1] for safety
    delta = float(np.clip(delta, -1.0, 1.0))

    # Convert bin offset to frequency
    # Local bin spacing near idx_peak:
    if idx_peak + 1 < n:
        df_right = freq_win[idx_peak + 1] - freq_win[idx_peak]
    else:
        df_right = freq_win[idx_peak] - freq_win[idx_peak - 1]
    f_refined = float(freq_win[idx_peak] + delta * df_right)

    # Approximate refined magnitude (vertex value)
    mag_refined = float(y2 - 0.25 * (y1 - y3) * delta)

    return f_refined, mag_refined

def _remove_close_peaks(detected_peaks, min_dist_hz=15.0):
    """
    Remove peaks that lie closer than min_dist_hz.
    Keep the one with the higher SNR.

    Input:
        detected_peaks : list of DetectedPeak or None
    Output:
        cleaned_peaks  : list of DetectedPeak or None (same length)
    """
    # Valid peaks with their indices, sorted by frequency
    valid = [(i, dp) for i, dp in enumerate(detected_peaks) if dp is not None]
    valid_sorted = sorted(valid, key=lambda x: x[1].freq_hz)

    keep = []
    for idx, dp in valid_sorted:
        placed = False

        # Compare dp to already-kept peaks
        for k_idx, k_dp in keep:
            if abs(dp.freq_hz - k_dp.freq_hz) < min_dist_hz:
                # conflict -> keep the higher-SNR peak, otherwise discard dp
                if dp.snr > k_dp.snr:
                    keep.remove((k_idx, k_dp))
                    keep.append((idx, dp))
                placed = True
                break

        if not placed:
            keep.append((idx, dp))

    # Build result (same size as input)
    cleaned = [None] * len(detected_peaks)
    for i, dp in keep:
        cleaned[i] = dp

    return cleaned

# ------------------------------------------------------------
# Precise anchor refinement: zero-padding + LMFit single-peak
# ------------------------------------------------------------
def refine_anchor_frequency(
    fid: np.ndarray,
    dwell_time_s: float,
    t2_bounds: tuple,
    fmin_anchor: float,
    fmax_anchor: float,
    noise : float,
    snr_threshold: float = 3.0,
    zp_factor: int = 8,
    max_nfev: int = 500,
    detection_line_broadening_hz: float = 5.0,
):
    """
    Refine the anchor peak frequency with a two-stage approach:
      1) Zero-filled FFT + parabolic sub-bin interpolation to seed frequency
      2) LMFit single-exponential time-domain fit to refine f, T2*, amplitude

    The SNRs are peak heights in the spectrum apodized with
    detection_line_broadening_hz over the noise std of that spectrum; `noise` is the
    time-domain noise std per channel.

    Returns:
        dict with keys:
          success (bool),
          freq (float or np.nan),
          amp (complex),
          t2 (float or np.nan),
          snr_prefit (float),
          snr_postfit (float),
          message (str)
    """
    # ---------- Stage 1: zero-filled spectrum & sub-bin seed ----------
    n_orig = fid.size
    n_zp = max(zp_factor, 1) * n_orig

    t = np.arange(n_orig, dtype=float) * dwell_time_s

    fid_seed = _apply_exponential_line_broadening(
        fid=fid,
        t_axis=t,
        line_broadening_hz=detection_line_broadening_hz,
    )

    fid_zp = np.zeros(n_zp, dtype=fid.dtype)
    fid_zp[:n_orig] = fid_seed

    spec = np.fft.fftshift(np.fft.fft(fid_zp))
    freq = np.fft.fftshift(np.fft.fftfreq(n_zp, d=dwell_time_s))

    mask = (freq >= fmin_anchor) & (freq <= fmax_anchor)
    if not np.any(mask):
        return dict(success=False, freq=np.nan, amp=0j, t2=np.nan,
                    snr_prefit=0.0, snr_postfit=0.0,
                    message="Anchor window empty.")

    freq_win = freq[mask]
    mag = np.abs(spec[mask])

    idx = int(np.argmax(mag))
    if len(mag) < 3 or idx <= 0 or idx >= len(mag) - 1:
        return dict(success=False, freq=np.nan, amp=0j, t2=np.nan,
                    snr_prefit=0.0, snr_postfit=0.0,
                    message="Insufficient bins for sub-bin interpolation.")

    f_seed, mag_seed = _parabolic_subbin(freq_win, mag, idx)

    # `noise` is the time-domain noise std; the apodized spectrum has noise std
    # noise * ||window||_2 (zero-filling adds no noise).
    window = _apply_exponential_line_broadening(
        fid=np.ones(n_orig),
        t_axis=t,
        line_broadening_hz=detection_line_broadening_hz,
    )
    noise_spec = noise * np.sqrt(np.sum(window**2))

    snr_prefit = float(mag_seed / noise_spec)
    if snr_prefit < snr_threshold:
        return dict(success=False, freq=np.nan, amp=0j, t2=np.nan,
                    snr_prefit=snr_prefit, snr_postfit=0.0,
                    message=f"Anchor SNR below threshold ({snr_prefit:.2f} < {snr_threshold}).")

    # ---------- Stage 2: LMFit single-exponential refinement ----------
    try:
        import lmfit
    except Exception:
        # If LMFit not installed, return the spectral seed as best effort
        return dict(success=True, freq=float(f_seed), amp=0j, t2=np.nan,
                    snr_prefit=snr_prefit, snr_postfit=snr_prefit,
                    message="LMFit not available; returning spectral seed.")

    params = lmfit.Parameters()
    # Seed amplitude: use time-domain first sample magnitude as a rough scale
    A0 = complex(fid[0]) if fid[0] != 0 else complex(np.mean(fid))
    if not np.isfinite(A0):
        A0 = 1.0 + 0j
    params.add("A_real", value=float(np.real(A0)), min=-1e12, max=1e12)
    params.add("A_imag", value=float(np.imag(A0)), min=-1e12, max=1e12)
    params.add("f",      value=float(f_seed),      min=float(fmin_anchor), max=float(fmax_anchor))
    params.add("T2",     value=float(np.clip(0.03, *t2_bounds)), min=float(t2_bounds[0]), max=float(t2_bounds[1]))

    def _resid(p):
        A = p["A_real"].value + 1j * p["A_imag"].value
        f = p["f"].value
        T2 = p["T2"].value
        fit = A * np.exp(1j * 2 * np.pi * f * t) * np.exp(-t / T2)
        r = fit - fid
        return np.concatenate([r.real, r.imag])

    res = lmfit.minimize(_resid, params, method="leastsq", max_nfev=int(max_nfev))
    A_fit = res.params["A_real"].value + 1j * res.params["A_imag"].value
    f_fit = float(res.params["f"].value)
    T2_fit = float(res.params["T2"].value)

    # Post-fit SNR: height of the fitted peak in the apodized spectrum over its noise std
    snr_post = float(np.abs(A_fit) * np.sum(window * np.exp(-t / T2_fit)) / noise_spec)

    # Basic validity
    if not (np.isfinite(f_fit) and np.isfinite(T2_fit)):
        return dict(success=False, freq=np.nan, amp=0j, t2=np.nan,
                    snr_prefit=snr_prefit, snr_postfit=snr_post,
                    message="Anchor fit did not converge to finite values.")

    return dict(success=True, freq=f_fit, amp=A_fit, t2=T2_fit,
                snr_prefit=snr_prefit, snr_postfit=snr_post,
                message="Anchor refined successfully.")


def fit_single_voxel_multipeak(
    fid: np.ndarray,
    t_axis: np.ndarray,
    detected_peaks: List[DetectedPeak],
    freq_bounds_list: List[tuple],
    t2_bounds: tuple,
    snr_threshold: float,
    noise: float,
    boundary_tolerance_hz: Optional[float] = 3,
    detection_line_broadening_hz: float = 5.0,
):
    """
    Fit a single voxel's FID with one exponential A*exp(i*2*pi*f*t - t/T2*) per
    detected peak.

    Each frequency is bounded to detected frequency ± 45 Hz, within its interval.
    After each fit, check whether any fitted frequency converged to the boundary
    of its frequency bounds. If yes, remove those peaks from the model and refit
    from scratch, until no fitted frequency lies at a boundary. Peaks with a
    post-fit SNR <= snr_threshold are left out of the output.

    Args:
        fid: complex FID (1D)
        t_axis: time axis
        detected_peaks: list of DetectedPeak or None (per metabolite)
        freq_bounds_list: [(fmin, fmax), ...] per metabolite
        t2_bounds: (t2_min, t2_max)
        snr_threshold: user SNR threshold
        noise: time-domain noise std per channel (see hypermri.utils.physics.estimate_noise)
        boundary_tolerance_hz: absolute tolerance to consider "at boundary".
                               If None, computed per interval as max(0.2 Hz, 1% of width).
        detection_line_broadening_hz: line broadening of the detection spectrum. The
                               post-fit SNR is measured on the same apodized spectrum, so
                               it is comparable to the detection SNR.

    Returns:
        dict with keys (each of length M = number of metabolites):
          amps, freqs, t2s, fwhm, r2, low_snr_mask (always False)
    """
    # Lazy import of LMFit
    try:
        import lmfit
    except Exception:
        raise RuntimeError(
            "LMFit must be installed to use fit_single_voxel_multipeak(). "
            "Install with `pip install lmfit`."
        )

    # Number of metabolites (fixed dimension)
    M = len(detected_peaks)

    # Work on a mutable local copy—we may remove peaks between refits.
    dp_list = list(detected_peaks)

    def _boundary_tol_hz(fmin, fmax):
        if boundary_tolerance_hz is not None:
            return float(boundary_tolerance_hz)
        width = float(abs(fmax - fmin))
        return float(max(0.2, 0.01 * width))

    # --------------- Refit loop ---------------
    # Each refit removes at least one peak, so at most M refits are needed
    for _ in range(M + 1):
        # ------------------------------------------------------------
        # Build LMFit parameters from the CURRENT active peak list
        # ------------------------------------------------------------
        params = lmfit.Parameters()
        active_indices = []
        for m, dp in enumerate(dp_list):
            if dp is None:
                continue
            active_indices.append(m)
            # Frequency bounds: detected frequency ± 45 Hz, within the interval
            fmin_from_list, fmax_from_list = freq_bounds_list[m]
            fmin = max(dp.freq_hz - 45.0, fmin_from_list)
            fmax = min(dp.freq_hz + 45.0, fmax_from_list)

            # amplitude (complex → real+imag components)
            params.add(f"A_real_{m}", value=np.real(dp.amp), min=-1e9, max=1e9)
            params.add(f"A_imag_{m}", value=np.imag(dp.amp), min=-1e9, max=1e9)
            # frequency
            params.add(f"f_{m}", value=dp.freq_hz, min=fmin, max=fmax)

            # T2*
            params.add(
                f"T2_{m}",
                value=dp.t2_s,
                min=t2_bounds[0],
                max=t2_bounds[1],
            )

        # If no peaks are active, return empty results
        if len(active_indices) == 0:
            return dict(
                amps=np.zeros(M, dtype=np.complex128),
                freqs=np.full(M, np.nan),
                t2s=np.full(M, np.nan),
                fwhm=np.full(M, np.nan),
                r2=np.full(M, np.nan),
                low_snr_mask=np.full(M, False),
            )

        # ------------------------------------------------------------
        # LMFit residual (built from *current* active peaks)
        # ------------------------------------------------------------
        def residual(params):
            amps = []
            freqs = []
            t2s_ = []
            for m in active_indices:
                A = params[f"A_real_{m}"].value + 1j * params[f"A_imag_{m}"].value
                amps.append(A)
                freqs.append(params[f"f_{m}"].value)
                t2s_.append(params[f"T2_{m}"].value)
            amps = np.array(amps, dtype=np.complex128)
            freqs = np.array(freqs, dtype=np.float64)
            t2s_ = np.array(t2s_, dtype=np.float64)
            fit = _exp_fid_model(t_axis, amps, freqs, t2s_)
            # Return concatenated real/imag for LMFit
            return np.concatenate([(fit - fid).real, (fit - fid).imag])

        # ------------------------------------------------------------
        # Run optimization for this round
        # ------------------------------------------------------------
        result = lmfit.minimize(residual, params, method='leastsq')

        # ------------------------------------------------------------
        # Check for boundary-converged frequencies
        # ------------------------------------------------------------
        boundary_hits = []
        for m in active_indices:
            f_fit = float(result.params[f"f_{m}"].value)
            fmin, fmax = params[f"f_{m}"].min, params[f"f_{m}"].max
            tol = _boundary_tol_hz(fmin, fmax)
            if (f_fit - fmin) <= tol or (fmax - f_fit) <= tol:
                boundary_hits.append(m)

        # If we hit boundaries, remove those peaks and retry
        if boundary_hits:
            for m in boundary_hits:
                dp_list[m] = None  # remove this expected frequency from the model
            # Continue loop → rebuild params from scratch and re-fit
            continue

        # No boundary hits → finalize using this result
        final_active_indices = list(active_indices)
        break

    # ------------------------------------------------------------
    # Extract fitted parameters
    # ------------------------------------------------------------
    amps_fit = np.zeros(M, dtype=np.complex128)
    freqs_fit = np.full(M, np.nan)
    t2s_fit = np.full(M, np.nan)
    fwhm_fit = np.full(M, np.nan)
    r2_fit = np.full(M, np.nan)
    low_snr_mask = np.full(M, False)

    # Compute R² as goodness-of-fit metric
    fit_val = result.residual[: len(fid)] + 1j * result.residual[len(fid) :]
    ss_res = np.sum(np.abs(fit_val) ** 2)
    ss_tot = np.sum(np.abs(fid - np.mean(fid)) ** 2)
    r2_global = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Post-fit SNR is defined like the detection SNR: peak height of the fitted
    # component in the apodized spectrum over the noise std of that spectrum.
    window = _apply_exponential_line_broadening(
        fid=np.ones(len(t_axis)),
        t_axis=t_axis,
        line_broadening_hz=detection_line_broadening_hz,
    )
    noise_spec = noise * np.sqrt(np.sum(window**2))

    for running_idx, m in enumerate(final_active_indices):
        A = result.params[f"A_real_{m}"].value + 1j * result.params[f"A_imag_{m}"].value
        f = result.params[f"f_{m}"].value
        T2 = result.params[f"T2_{m}"].value

        # Post-fit SNR
        snr_post = np.abs(A) * np.sum(window * np.exp(-t_axis / 0.013)) / noise_spec
        if snr_post <= snr_threshold:
            continue

        # If the fitted frequency lies inside the next interval and that interval
        # has no active peak of its own, assign the peak to the next metabolite.
        final_idx = m
        if m < M - 1 and f >= freq_bounds_list[m + 1][0]:
            is_last_active = running_idx == len(final_active_indices) - 1
            if is_last_active or final_active_indices[running_idx + 1] != m + 1:
                final_idx = m + 1

        amps_fit[final_idx] = A
        freqs_fit[final_idx] = f
        t2s_fit[final_idx] = T2
        # FWHM = 1/(pi*T2*)
        if T2 > 0:
            fwhm_fit[final_idx] = 1.0 / (np.pi * T2)
        # Global R² assigned to each metabolite
        r2_fit[final_idx] = r2_global

    return dict(
        amps=amps_fit,
        freqs=freqs_fit,
        t2s=t2s_fit,
        fwhm=fwhm_fit,
        r2=r2_fit,
        low_snr_mask=low_snr_mask,
    )

# ------------------------------------------------------------
# Xarray wrapper: fit_data_multipeak
# -----------------------------------------------------------
def fit_data_multipeak(
    input_data: xr.DataArray,
    fit_params: xr.Dataset,
    snr_threshold: float = 5.0,
    noise : float = 1.0,
    kappa_curvature : float = 2.0,
    min_prominence_threshold : float = None,
    boundary_tolerance_hz: Optional[float] = 3,
    detection_line_broadening_hz: float = 5.0,
    min_fwhm_hz: float = 16.0,
):
    """
    Automatic one-peak-per-interval fitting for CSI data,
    with optional voxel-wise anchor-based B0 alignment.

    noise is the time-domain noise std per channel (real/imag) of a voxel FID, e.g.
    from hypermri.utils.physics.estimate_noise. All SNRs (detection and post-fit) are
    peak heights in the spectrum apodized with detection_line_broadening_hz, divided
    by the noise std of that spectrum.

    Per voxel:
      1) Interval centers: initial_freqs_Hz, optionally shifted by the anchor (below).
         Intervals: center_m ± freq_ranges_Hz[m]/2.
      2) Detect at most one peak per interval (detect_peak_in_interval).
      3) Of detected peaks closer than 15 Hz, keep the one with higher SNR.
      4) Fit all remaining peaks together (fit_single_voxel_multipeak).

    Anchor logic (optional, controlled via fit_params):
      • Find the anchor in [anchor_freq_Hz ± anchor_range_Hz/2] with a single-peak
        fit (refine_anchor_frequency); if that fails, with the peak detector.
      • If anchor found:
          - If 'relative_offsets_Hz' provided: center_m = f_anchor_fit + relative_offsets_Hz[m]
          - Else: center_m = initial_freqs_Hz[m] + (f_anchor_fit - anchor_freq_Hz)
      • If anchor NOT found: fall back to the unshifted intervals.

    Input expectations and outputs:
      - input_data dims: (..., 'time') with 'time' last
      - output: (fit dataset, debug dataset), same leading dims + 'metabolite'
    """

    # ------------ Validate presence & position of "time" ------------
    if "time" not in input_data.dims:
        raise ValueError(
            "fit_data_multipeak(): input_data must have a 'time' dimension. "
            "Use `prepare_inputs_for_multipeak()` to rename/transpose."
        )
    if input_data.dims[-1] != "time":
        raise ValueError(
            "fit_data_multipeak(): 'time' must be the LAST dimension. "
            "Use `prepare_inputs_for_multipeak()` to move it."
        )

    # ------------ Time axis construction ------------
    if "dwell_time_s" not in fit_params:
        raise ValueError("fit_params must contain 'dwell_time_s' (scalar).")
    dwell_value = fit_params["dwell_time_s"]
    dwell = float(dwell_value.item() if hasattr(dwell_value, "item") else dwell_value)

    n_time = input_data.sizes["time"]
    t_axis = np.arange(n_time, dtype=float) * dwell

    # ------------ Metabolite configuration ------------
    if "initial_freqs_Hz" not in fit_params or "freq_ranges_Hz" not in fit_params:
        raise ValueError(
            "fit_params must contain 'initial_freqs_Hz' and 'freq_ranges_Hz', each 1D over 'metabolite'."
        )
    if "metabolite" not in fit_params["initial_freqs_Hz"].dims:
        raise ValueError("'initial_freqs_Hz' must have a 'metabolite' dimension.")
    if "metabolite" not in fit_params["freq_ranges_Hz"].dims:
        raise ValueError("'freq_ranges_Hz' must have a 'metabolite' dimension.")

    metab_names = list(fit_params["initial_freqs_Hz"].metabolite.values)
    M = len(metab_names)

    initial_freqs = np.asarray(fit_params["initial_freqs_Hz"].values, dtype=float).reshape(-1)
    ranges = np.asarray(fit_params["freq_ranges_Hz"].values, dtype=float).reshape(-1)

    # T2 bounds
    if "t2_bounds" not in fit_params:
        raise ValueError("fit_params must contain 't2_bounds' = [min_T2, max_T2].")
    t2b_raw = fit_params["t2_bounds"]
    t2b_arr = np.asarray(t2b_raw.values if hasattr(t2b_raw, "values") else t2b_raw, dtype=float).reshape(-1)
    if t2b_arr.size != 2:
        raise ValueError("'t2_bounds' must be a length-2 array-like: [min_T2, max_T2].")
    t2_bounds = (float(t2b_arr[0]), float(t2b_arr[1]))

    # ------------ Anchor configuration (optional) ------------
    use_anchor = False
    anchor_freq = None
    anchor_range = None
    rel_offsets = None

    if "use_anchor_alignment" in fit_params:
        use_anchor = bool(fit_params["use_anchor_alignment"].item()
                          if hasattr(fit_params["use_anchor_alignment"], "item")
                          else fit_params["use_anchor_alignment"])

    if use_anchor:
        if "anchor_freq_Hz" not in fit_params or "anchor_range_Hz" not in fit_params:
            raise ValueError("To use anchor alignment, provide 'anchor_freq_Hz' and 'anchor_range_Hz' in fit_params.")

        anchor_freq = float(fit_params["anchor_freq_Hz"].item()
                            if hasattr(fit_params["anchor_freq_Hz"], "item")
                            else fit_params["anchor_freq_Hz"])
        anchor_range = float(fit_params["anchor_range_Hz"].item()
                             if hasattr(fit_params["anchor_range_Hz"], "item")
                             else fit_params["anchor_range_Hz"])

        if "relative_offsets_Hz" in fit_params:
            rel_offsets = np.asarray(fit_params["relative_offsets_Hz"].values, dtype=float).reshape(-1)
            if rel_offsets.shape[0] != M:
                raise ValueError("'relative_offsets_Hz' must have the same length as 'metabolite'.")

    detection_kwargs = dict(
        t_axis=t_axis,
        noise_sigma=noise,
        snr_threshold=snr_threshold,
        kappa_curvature=kappa_curvature,
        min_prominence_threshold=min_prominence_threshold,
        min_fwhm_hz=min_fwhm_hz,
        detection_line_broadening_hz=detection_line_broadening_hz,
    )

    # ------------ Per-voxel worker ------------
    def _worker(fid_1d):
        fid = np.asarray(fid_1d, dtype=np.complex128)

        # Default: unshifted centers
        centers = initial_freqs.copy()

        # ----- Anchor detection (optional) -----
        if use_anchor:
            fmin_a = anchor_freq - anchor_range * 0.5
            fmax_a = anchor_freq + anchor_range * 0.5

            # Precise anchor refinement (spectral sub-bin + single-peak LMFit)
            anchor_res = refine_anchor_frequency(
                fid=fid,
                dwell_time_s=dwell,
                t2_bounds=t2_bounds,
                fmin_anchor=fmin_a,
                fmax_anchor=fmax_a,
                noise=noise,
                snr_threshold=float(snr_threshold),
                zp_factor=int(fit_params.get("anchor_zp_factor", 8)),
                max_nfev=int(fit_params.get("anchor_max_nfev", 600)),
                detection_line_broadening_hz=detection_line_broadening_hz,
            )

            f_anchor_fit = None
            if anchor_res["success"]:
                f_anchor_fit = anchor_res["freq"]
            else:
                # Fallback: use the peak detector
                dp_anchor = detect_peak_in_interval(fid=fid, fmin=fmin_a, fmax=fmax_a, **detection_kwargs)
                if dp_anchor is not None:
                    f_anchor_fit = dp_anchor.freq_hz

            # If the anchor was found, shift the centers; otherwise keep them unshifted
            if f_anchor_fit is not None and np.isfinite(f_anchor_fit):
                if rel_offsets is not None:
                    # Tie windows to the anchor by relative offsets
                    centers = f_anchor_fit + rel_offsets
                else:
                    # Global B0 shift
                    centers = initial_freqs + (f_anchor_fit - anchor_freq)

        # ----- Per-metabolite bounds (center ± range/2) -----
        freq_bounds_list = [(c - R * 0.5, c + R * 0.5) for c, R in zip(centers, ranges)]

        # ----- Detect at most one peak per interval -----
        detected_peaks = [
            detect_peak_in_interval(fid=fid, fmin=fmin, fmax=fmax, **detection_kwargs)
            for fmin, fmax in freq_bounds_list
        ]

        # Remove detected peaks that are closer than 15 Hz apart
        detected_peaks = _remove_close_peaks(detected_peaks, min_dist_hz=15)

        # ----- Fit voxel (one exponential per detected interval) -----
        res = fit_single_voxel_multipeak(
            fid=fid,
            t_axis=t_axis,
            detected_peaks=detected_peaks,
            freq_bounds_list=freq_bounds_list,
            t2_bounds=t2_bounds,
            snr_threshold=snr_threshold,
            noise=noise,
            boundary_tolerance_hz=boundary_tolerance_hz,
            detection_line_broadening_hz=detection_line_broadening_hz,
        )

        # ----- Debug info -----
        peak_freqs = np.full(M, np.nan, float)
        peak_amps = np.full(M, np.nan, complex)
        peak_snrs = np.full(M, np.nan, float)
        peak_valid = np.zeros(M, bool)
        for m, dp in enumerate(detected_peaks):
            if dp is not None:
                peak_freqs[m] = dp.freq_hz
                peak_amps[m] = dp.amp
                peak_snrs[m] = dp.snr
                peak_valid[m] = True

        return (
            res["amps"],
            res["freqs"],
            res["t2s"],
            res["fwhm"],
            res["r2"],
            res["low_snr_mask"].astype(np.int8),
            np.asarray(centers, dtype=float),           # (M,)
            np.asarray(freq_bounds_list, dtype=float),  # (M, 2)
            peak_freqs,
            peak_amps,
            peak_snrs,
            peak_valid.astype(np.int8),
        )

    # ------------ Apply across all leading dims (vectorized) ------------
    out = xr.apply_ufunc(
        _worker,
        input_data,
        input_core_dims=[["time"]],
        output_core_dims=[
            ["metabolite"],                   # amps
            ["metabolite"],                   # freqs
            ["metabolite"],                   # t2s
            ["metabolite"],                   # fwhm
            ["metabolite"],                   # r2
            ["metabolite"],                   # low_snr_mask
            ["metabolite"],                   # centers
            ["metabolite", "interval_edge"],  # bounds
            ["metabolite"],                   # peak_freqs
            ["metabolite"],                   # peak_amps
            ["metabolite"],                   # peak_snrs
            ["metabolite"],                   # peak_valid
        ],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[
            np.complex128,  # amps
            np.float64,     # freqs
            np.float64,     # t2s
            np.float64,     # fwhm
            np.float64,     # r2
            np.int8,        # low_snr_mask
            np.float64,     # centers
            np.float64,     # bounds
            np.float64,     # peak_freqs
            np.complex128,  # peak_amps
            np.float64,     # peak_snrs
            np.int8,        # peak_valid
        ],
    )

    # ------------ Collect outputs and build dataset (exclude 'time' coords) ------------
    (
        amps_arr, freqs_arr, t2s_arr, fwhm_arr, r2_arr, low_arr,
        centers_arr, bounds_arr,
        peak_freqs_arr, peak_amps_arr, peak_snrs_arr, peak_valid_arr,
    ) = out

    leading_dims = input_data.dims[:-1]  # all dims except "time"
    out_dims = leading_dims + ("metabolite",)

    out_coords = {}
    for d in leading_dims:
        out_coords[d] = input_data.coords[d] if d in input_data.coords else np.arange(input_data.sizes[d])
    out_coords["metabolite"] = fit_params["initial_freqs_Hz"].metabolite

    def _as_dataset(arrays):
        return xr.Dataset(
            {name: xr.DataArray(arr, dims=out_dims, coords=out_coords) for name, arr in arrays.items()}
        )

    ds = _as_dataset({
        "fit_amps": amps_arr,
        "fit_freqs": freqs_arr,
        "fit_t2s": t2s_arr,
        "fit_fwhm_Hz": fwhm_arr,
        "fit_r2": r2_arr,
        "fit_low_snr_refitted": low_arr,
    })

    debug_ds = _as_dataset({
        "centers_Hz": centers_arr,
        "interval_fmin_Hz": bounds_arr[..., 0],
        "interval_fmax_Hz": bounds_arr[..., 1],
        "detected_peak_freq_Hz": peak_freqs_arr,
        "detected_peak_amp": peak_amps_arr,
        "detected_peak_snr": peak_snrs_arr,
        "detected_peak_valid": peak_valid_arr.astype(bool),
    })

    return ds, debug_ds


# ------------------------------------------------------------
# Optional convenience wrappers and final helpers
# ------------------------------------------------------------
def compute_frequency_axis(n_time: int, dwell_time_s: float) -> np.ndarray:
    """
    Compute FFT frequency axis (Hz) with the same convention
    used throughout this module.

    Args:
        n_time: number of time points
        dwell_time_s: dwell time (s)

    Returns:
        freq_axis_hz: 1D frequency axis (Hz)
    """
    return np.fft.fftshift(np.fft.fftfreq(n_time, d=dwell_time_s))


def run_multipeak_fitting(
    input_data: xr.DataArray,
    fit_params: xr.Dataset,
    snr_threshold: float = 5.0,
    noise: float = 1.0,
    kappa_curvature : float = 2.0,
    min_prominence_threshold : float =None,
    boundary_tolerance_hz: Optional[float] = 3,
    detection_line_broadening_hz: float = 5.0,
    min_fwhm_hz: float = 16.0,
):
    """
    Convenience wrapper calling fit_data_multipeak().
    Provided for symmetry with your existing pseudo-inverse pipeline.

    Args:
        input_data: xarray DataArray (..., time)
        fit_params: xarray Dataset (initial frequencies, ranges, t2_bounds)
        snr_threshold: minimum SNR to accept a peak
        noise: time-domain noise std per channel of a voxel FID

    Returns:
        Output dataset from fit_data_multipeak()
    """
    return fit_data_multipeak(
        input_data=input_data,
        fit_params=fit_params,
        snr_threshold=snr_threshold,
        noise=noise,
        kappa_curvature=kappa_curvature,
        min_prominence_threshold=min_prominence_threshold,
        boundary_tolerance_hz = boundary_tolerance_hz,
        detection_line_broadening_hz=detection_line_broadening_hz,
        min_fwhm_hz=min_fwhm_hz,
    )


def plot_detected_peaks_and_fit(
    fid: np.ndarray,
    t_axis: np.ndarray,
    freq_axis_hz: np.ndarray,
    detected_peaks,
    fitted_result: dict,
    fmin_list,
    fmax_list,
    figsize=(15, 5),
    spectrum_mode: str = "magnitude",  # "magnitude" | "realimag"
    zero_pad_factor: int = 4,          # 1 = no zero padding; 4/8/16 gives smoother spectra
    show_residual: bool = True,
    interval_alpha: float = 0.10,
    peak_marker_color: str = "red",
    peak_marker_ms: int = 6,
):
    """
    Visualize:
      (Left)  Reconstructed spectrum (measured) + simulated spectrum (from fitted FID)
              with interval shading and detected peak markers.
      (Right) FID time-domain: measured + fitted + (optional) residual.

    Parameters
    ----------
    fid : np.ndarray (complex, shape [N])
        Measured FID.
    t_axis : np.ndarray (float, [N])
        Time axis in seconds for the FID.
    freq_axis_hz : np.ndarray (float, [N])
        Not used (the spectra are computed from t_axis); kept for compatibility.
    detected_peaks : list[DetectedPeak or None]
        Output of your peak detector per interval (one or zero per interval).
    fitted_result : dict
        Must have keys: "amps", "freqs", "t2s". (Optional keys are ignored.)
    fmin_list, fmax_list : list[float]
        Interval boundaries per metabolite (same order as detected_peaks).
    figsize : tuple
        Matplotlib figure size.
    spectrum_mode : str
        "magnitude": plot |S|; "realimag": plot Re/Im as separate curves.
    zero_pad_factor : int
        Zero-filling factor to improve spectral resolution in the overlay.
    show_residual : bool
        If True, show residual in FID panel.
    interval_alpha : float
        Alpha for interval shading.
    peak_marker_color : str
        Color for detected-peak vertical lines/markers.
    peak_marker_ms : int
        Marker size.
    """

    # --------------- Helpers ---------------

    def _reconstruct_fid_from_fit(t, amps, freqs, t2s):
        y = np.zeros_like(t, dtype=np.complex128)
        for A, f, T2 in zip(amps, freqs, t2s):
            if (A is None) or (f is None) or (T2 is None):
                continue
            if np.isnan(f) or np.isnan(T2):
                continue
            y += A * np.exp(1j * 2 * np.pi * f * t) * np.exp(-t / T2)
        return y

    def _fft_with_padding(y, dwell, pad_factor):
        n = len(y)
        if pad_factor and pad_factor > 1:
            n_zp = int(pad_factor) * n
            y_pad = np.zeros(n_zp, dtype=y.dtype)
            y_pad[:n] = y
        else:
            n_zp = n
            y_pad = y
        spec = np.fft.fftshift(np.fft.fft(y_pad))
        freq = np.fft.fftshift(np.fft.fftfreq(n_zp, d=dwell))
        return freq, spec

    def _nearest_mag(freq_axis, spec, f0):
        """helper to sample |S| at closest bin (for plotting peak markers)."""
        if f0 is None or not np.isfinite(f0):
            return None
        idx = int(np.argmin(np.abs(freq_axis - f0)))
        return np.abs(spec[idx])

    # --------------- Prepare data ---------------

    # Extract fitted params (handle missing keys gracefully)
    amps = np.asarray(fitted_result.get("amps", []))
    freqs = np.asarray(fitted_result.get("freqs", []))
    t2s  = np.asarray(fitted_result.get("t2s", []))

    if amps.ndim == 0: amps = np.atleast_1d(amps)
    if freqs.ndim == 0: freqs = np.atleast_1d(freqs)
    if t2s.ndim == 0: t2s = np.atleast_1d(t2s)

    # Compute fitted FID and residual
    fid_fit = _reconstruct_fid_from_fit(t_axis, amps, freqs, t2s)
    resid   = fid - fid_fit

    # Dwell from time axis
    if len(t_axis) < 2:
        raise ValueError("t_axis must have at least 2 points.")
    dwell = float(t_axis[1] - t_axis[0])

    # Build spectra (with optional zero padding)
    f_meas, S_meas = _fft_with_padding(fid, dwell, zero_pad_factor)
    f_fitp, S_fitp = _fft_with_padding(fid_fit, dwell, zero_pad_factor)

    # --------------- Figure / Axes ---------------
    fig, (ax_spec, ax_fid) = plt.subplots(1, 2, figsize=figsize)

    # ===== Left: Spectrum overlay =====
    ax = ax_spec
    if spectrum_mode.lower() == "realimag":
        ax.plot(f_meas, S_meas.real, color="black", lw=1.0, label="Re{Spectrum}")
        ax.plot(f_meas, S_meas.imag, color="gray",  lw=1.0, label="Im{Spectrum}")
        ax.plot(f_fitp, S_fitp.real, "--", color="tab:red",   lw=1.2, label="Re{Fit}")
        ax.plot(f_fitp, S_fitp.imag, "--", color="tab:orange",lw=1.2, label="Im{Fit}")
        ax.set_ylabel("Re/Im")
    else:
        ax.plot(f_meas, np.abs(S_meas), color="black", lw=1.0, label="|Spectrum|")
        ax.plot(f_fitp, np.abs(S_fitp), "--", color="tab:red", lw=1.2, label="|Fit Spectrum|")
        ax.set_ylabel("Magnitude")

    ax.set_title("Spectrum Overlay (measured vs. fitted)")
    ax.set_xlabel("Frequency (Hz)")

    # Shade search intervals
    for fmin, fmax in zip(fmin_list, fmax_list):
        ax.axvspan(fmin, fmax, color="orange", alpha=interval_alpha, lw=0)

    # Mark detected peaks
    for dp in detected_peaks:
        if dp is None or not getattr(dp, "valid", True):
            continue
        ax.axvline(dp.freq_hz, color=peak_marker_color, lw=1.2, alpha=0.9)
        # Plot marker at the measured magnitude for visibility
        mag_here = _nearest_mag(f_meas, S_meas, dp.freq_hz)
        if mag_here is not None:
            ax.plot(dp.freq_hz, mag_here, "o", color=peak_marker_color, ms=peak_marker_ms)

    ax.legend(loc="best")

    # ===== Right: FID (time-domain) =====
    ax2 = ax_fid
    ax2.plot(t_axis, fid.real, label="FID real", color="tab:blue")
    ax2.plot(t_axis, fid.imag, label="FID imag", color="tab:green")
    ax2.plot(t_axis, fid_fit.real, "--", label="Fit real", color="tab:red", lw=1.2, alpha=0.7)
    ax2.plot(t_axis, fid_fit.imag, "--", label="Fit imag", color="tab:purple", lw=1.2, alpha=0.7)
    if show_residual:
        ax2.plot(t_axis, resid.real, color="black", alpha=1, label="Residual real")

    ax2.set_title("FID + Fitted")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Signal")
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# Function necessary to plug in with old fitting routine
# -----------------------------------------------------------
def prepare_inputs_for_multipeak(
    input_data: xr.DataArray,
    fit_params: xr.Dataset,
    spectral_dim: str = "acquisition_idx",
    time_dim: str = "time",
    keep_attrs: bool = True,
    ensure_complex: bool = True,
    infer_dwell_from_coord: bool = True,
):
    """
    Prepare xarray inputs for fit_data_multipeak():
      • Rename the spectral dimension to 'time'
      • Move 'time' to the last axis
      • Ensure complex dtype (if requested)
      • Ensure fit_params contains dwell_time_s; infer if possible

    Parameters
    ----------
    input_data : xr.DataArray
        CSI data with an explicit spectral/time-like dimension (e.g., 'acquisition_idx').
        Can have arbitrary leading (loop) dimensions (e.g., x, y, z, repetition, channel).
    fit_params : xr.Dataset
        Must contain (at minimum):
          - 'initial_freqs_Hz' (per 'metabolite')
          - 'freq_ranges_Hz'   (per 'metabolite')
        and optionally:
          - 'dwell_time_s' (scalar). If missing and `infer_dwell_from_coord=True`,
            the wrapper attempts to infer it.
    spectral_dim : str, default 'acquisition_idx'
        Name of the spectral dimension in `input_data`.
    time_dim : str, default 'time'
        Target name for the spectral dimension required by `fit_data_multipeak()`.
    keep_attrs : bool, default True
        If True, keep original attributes and coordinates.
    ensure_complex : bool, default True
        If True, cast data to complex128 if it isn't complex already.
    infer_dwell_from_coord : bool, default True
        If True and 'dwell_time_s' is missing, infer from a time-like coordinate
        (attached to the spectral dimension), if available.

    Returns
    -------
    input_data_prep : xr.DataArray
        Prepared DataArray with dims (..., 'time').
    fit_params_prep : xr.Dataset
        Updated fit_params with guaranteed 'dwell_time_s' and same metabolite coords.

    Raises
    ------
    ValueError
        If spectral_dim is not found, or if dwell_time_s cannot be determined.
    """

    da = input_data

    # 1) Check the spectral dimension exists
    if spectral_dim not in da.dims and time_dim not in da.dims:
        raise ValueError(
            f"Neither '{spectral_dim}' nor '{time_dim}' found in input_data.dims={da.dims}."
        )

    # 2) If the spectral dim is not already named 'time', rename it.
    if time_dim not in da.dims:
        da = da.rename({spectral_dim: time_dim})

    # 3) Move 'time' to the end
    new_order = tuple(d for d in da.dims if d != time_dim) + (time_dim,)
    da = da.transpose(*new_order)

    # 4) Ensure complex dtype (recommended for FID processing)
    if ensure_complex and not np.issubdtype(da.dtype, np.complexfloating):
        da = da.astype(np.complex128)

    # 5) Prepare fit_params
    fp = fit_params.copy(deep=True)

    # Validate required metabolite arrays:
    if "initial_freqs_Hz" not in fp or "freq_ranges_Hz" not in fp:
        raise ValueError(
            "fit_params must contain 'initial_freqs_Hz' and 'freq_ranges_Hz' (both 1D over 'metabolite')."
        )
    if "metabolite" not in fp["initial_freqs_Hz"].dims or "metabolite" not in fp["freq_ranges_Hz"].dims:
        raise ValueError(
            "'initial_freqs_Hz' and 'freq_ranges_Hz' must have a 'metabolite' dimension."
        )

    # 6) Ensure dwell_time_s
    have_dwell = "dwell_time_s" in fp
    if not have_dwell and infer_dwell_from_coord:
        # Try to infer from a coordinate attached to 'time'
        # Expecting a numeric coord with units in seconds, strictly increasing
        coord = None
        if time_dim in da.coords:
            coord = da.coords[time_dim].values
        elif spectral_dim in input_data.coords and time_dim not in input_data.coords:
            # if original had coord on spectral_dim name, try it
            coord = input_data.coords.get(spectral_dim, None)
            if coord is not None:
                coord = coord.values

        if coord is not None and np.ndim(coord) == 1 and len(coord) >= 2:
            # Dwell = mean delta between consecutive time samples (absolute)
            diffs = np.diff(np.asarray(coord, dtype=float))
            # Be robust to sign; we only need magnitude
            dwell_est = float(np.mean(np.abs(diffs)))
            if dwell_est > 0:
                fp["dwell_time_s"] = dwell_est

    # Final check
    if "dwell_time_s" not in fp:
        raise ValueError(
            "fit_params must contain 'dwell_time_s' (scalar) or provide a time coordinate "
            f"on '{time_dim}' to infer it (set infer_dwell_from_coord=True)."
        )

    # 7) Optionally keep original attributes
    if keep_attrs:
        da.attrs.update(getattr(input_data, "attrs", {}))
        fp.attrs.update(getattr(fit_params, "attrs", {}))

    return da, fp


def view_csi_fit_with_map(
    signal_data: xr.DataArray,
    fit_results: xr.Dataset,
    fit_params: xr.Dataset,
    debug_info: xr.Dataset,
    spatial_dims=("read_idx", "phase_idx"),
    metabolite_index: int = 0,
    figsize=(18, 6),
    spectrum_mode="magnitude",     # "magnitude" | "realimag"
    zero_pad_factor=8,
    show_residual=True,
    fixed_coords=None,
    show_sliders=True,
    legend_max_rows=8,
    noise=None,
    show_line_broadening=False,
    line_broadening_hz=5.0,
):
    """
    3-panel CSI viewer using *debug_info* from fit_data_multipeak:
      - Left  : clickable amplitude map
      - Middle: FFT spectrum (measured + simulated) + anchor-shifted intervals + detected peaks
      - Right : FID (real/imag + fit + residual)

    Intervals and detected peaks are taken from debug_info (interval_fmin_Hz,
    interval_fmax_Hz, detected_peak_freq_Hz, detected_peak_valid).
    show_line_broadening and line_broadening_hz set the initial state of the
    line-broadening widgets. noise is not used; kept for compatibility.
    """

    # ---------------------------- Validation ----------------------------
    if signal_data.dims[-1] != "time":
        raise ValueError("signal_data must have 'time' as last dimension.")

    if "metabolite" not in fit_results:
        raise ValueError("fit_results must contain dimension 'metabolite'.")

    if spatial_dims[0] not in signal_data.dims or spatial_dims[1] not in signal_data.dims:
        raise ValueError("spatial_dims not in signal_data dims.")

    xdim, ydim = spatial_dims
    nx = signal_data.sizes[xdim]
    ny = signal_data.sizes[ydim]

    metab_names = list(fit_results.metabolite.values)
    if metabolite_index < 0 or metabolite_index >= len(metab_names):
        raise IndexError("Invalid metabolite_index.")

    fixed_coords = dict(fixed_coords or {})
    for d in signal_data.dims[:-1]:
        if d not in spatial_dims:
            fixed_coords.setdefault(d, 0)

    dwell = float(fit_params["dwell_time_s"])
    n_time = signal_data.sizes["time"]
    t_axis = np.arange(n_time) * dwell

    # ---------------------------- Map data ----------------------------
    amp_da = fit_results["fit_amps"].sel(metabolite=metab_names[metabolite_index])
    sel_map = {k: v for k, v in fixed_coords.items() if k in amp_da.dims}
    amp_map = np.abs(amp_da.isel(**sel_map))

    # ---------------------------- Figure ----------------------------
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 3, width_ratios=[1.3, 1.4, 1.4], figure=fig)

    ax_map = fig.add_subplot(gs[0, 0])
    ax_spec = fig.add_subplot(gs[0, 1])
    ax_fid = fig.add_subplot(gs[0, 2])

    ax_map.imshow(amp_map.T, origin="lower", cmap="viridis")
    ax_map.set_title(f"Amplitude map ({metab_names[metabolite_index]})")
    ax_map.set_xlabel(xdim)
    ax_map.set_ylabel(ydim)
    sel_marker, = ax_map.plot([], [], "ro", ms=6)

    legend_text_artist = None

    # ---------------------------- Helpers ----------------------------
    def _reconstruct_fid(amps, freqs, t2s):
        out = np.zeros_like(t_axis, dtype=np.complex128)
        for A, f, T2 in zip(amps, freqs, t2s):
            if np.isfinite(f) and np.isfinite(T2):
                out += A * np.exp(1j * 2 * np.pi * f * t_axis) * np.exp(-t_axis / T2)
        return out

    def _fft(y):
        n = len(y)
        if zero_pad_factor > 1:
            n2 = n * zero_pad_factor
            y2 = np.zeros(n2, dtype=y.dtype)
            y2[:n] = y
        else:
            y2 = y
            n2 = n
        spec = np.fft.fftshift(np.fft.fft(y2))
        freq = np.fft.fftshift(np.fft.fftfreq(n2, d=dwell))
        return freq, spec

    # ---------------------------- Update function ----------------------------
    def update_voxel(ix, iy):
        nonlocal legend_text_artist
        ix = int(np.clip(ix, 0, nx - 1))
        iy = int(np.clip(iy, 0, ny - 1))
        sel_marker.set_data([ix], [iy])

        # voxel selectors
        sel = {xdim: ix, ydim: iy, **fixed_coords}

        # Extract voxel FID
        fid_v = signal_data.isel(**{k: v for k, v in sel.items() if k in signal_data.dims}).values

        # Fitted parameters
        amps_v = fit_results["fit_amps"].isel(**sel).values
        freqs_v = fit_results["fit_freqs"].isel(**sel).values
        t2s_v = fit_results["fit_t2s"].isel(**sel).values

        # Debug info
        fmin_v = debug_info["interval_fmin_Hz"].isel(**sel).values
        fmax_v = debug_info["interval_fmax_Hz"].isel(**sel).values
        peak_freq_v = debug_info["detected_peak_freq_Hz"].isel(**sel).values
        peak_valid_v = debug_info["detected_peak_valid"].isel(**sel).values.astype(bool)

        # Reconstruct fitted FID + residual
        fid_fit = _reconstruct_fid(amps_v, freqs_v, t2s_v)
        resid = fid_v - fid_fit

        # ---------------- FID Panel ----------------
        ax_fid.clear()
        ax_fid.plot(t_axis, fid_v.real, label="FID real", color="tab:blue")
        ax_fid.plot(t_axis, fid_v.imag, label="FID imag", color="tab:green")
        ax_fid.plot(t_axis, fid_fit.real, "--", label="Fit real", color="tab:red", alpha=0.7)
        ax_fid.plot(t_axis, fid_fit.imag, "--", label="Fit imag", color="tab:purple", alpha=0.7)
        if show_residual:
            ax_fid.plot(t_axis, resid.real, color="black", alpha=1.0, label="Residual real")
        ax_fid.legend()
        ax_fid.set_title(f"FID @ ({xdim}={ix}, {ydim}={iy})")

        # ---------------- Spectrum Panel ----------------
        ax_spec.clear()

        # Line broadening of the displayed spectrum (widgets, or the initial setting)
        if show_sliders and _WIDGETS_OK:
            lb_hz = float(lb_slider.value) if cb_lb.value else 0.0
        else:
            lb_hz = float(line_broadening_hz) if show_line_broadening else 0.0

        fid_plot = _apply_exponential_line_broadening(fid_v, dwell, lb_hz)

        f_meas, S_meas = _fft(fid_plot)
        f_sim, S_sim = _fft(fid_fit)

        if spectrum_mode == "realimag":
            ax_spec.plot(f_meas, S_meas.real, "k-", label="Re Spectrum")
            ax_spec.plot(f_meas, S_meas.imag, "gray", label="Im Spectrum")
            ax_spec.plot(f_sim, S_sim.real, "r--", label="Re Fit")
            ax_spec.plot(f_sim, S_sim.imag, "m--", label="Im Fit")
        else:
            ax_spec.plot(f_meas, np.abs(S_meas), "k-", label="|Spectrum|")
            ax_spec.plot(f_sim, np.abs(S_sim), "r--", label="|Fit Spectrum|")

        for f in freqs_v:
            ax_spec.axvline(f, color="lightgray")

        # Shade *actual* intervals (from debug_info)
        for fmin, fmax in zip(fmin_v, fmax_v):
            if np.isfinite(fmin) and np.isfinite(fmax):
                ax_spec.axvspan(fmin, fmax, color="orange", alpha=0.15)

        # Mark *actual detected peaks*
        for pf, valid in zip(peak_freq_v, peak_valid_v):
            if valid and np.isfinite(pf):
                idx = np.argmin(np.abs(f_meas - pf))
                mag_here = np.abs(S_meas[idx])
                ax_spec.plot(pf, mag_here, "ro", ms=6)

        ax_spec.set_title("Spectrum + intervals + detected peaks")
        ax_spec.set_xlabel("Frequency (Hz)")
        ax_spec.legend(loc="upper left")

        # Legend with fitted parameters
        if legend_text_artist:
            legend_text_artist.remove()

        legend_lines = []
        for name, f, A in zip(metab_names, freqs_v, amps_v):
            if np.isfinite(f):
                legend_lines.append(f"{name:>8s}  f={f:8.2f}  |A|={abs(A):8.2f}")

        legend_text = "\n".join(legend_lines[:legend_max_rows])
        legend_text_artist = ax_spec.text(
            0.99, 0.99, legend_text,
            ha="right", va="top", transform=ax_spec.transAxes,
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85),
        )

        fig.canvas.draw_idle()

    # ---------------------------- Interaction ----------------------------
    init_ix = nx // 2
    init_iy = ny // 2

    def onclick(event):
        if event.inaxes is ax_map and event.xdata is not None:
            ix = int(round(event.xdata))
            iy = int(round(event.ydata))
            if 0 <= ix < nx and 0 <= iy < ny:
                if show_sliders and _WIDGETS_OK:
                    sx.value = ix
                    sy.value = iy
                else:
                    update_voxel(ix, iy)

    fig.canvas.mpl_connect("button_press_event", onclick)

    if show_sliders and _WIDGETS_OK:
        sx = IntSlider(value=init_ix, min=0, max=nx - 1, description=xdim)
        sy = IntSlider(value=init_iy, min=0, max=ny - 1, description=ydim)
        cb_lb = Checkbox(value=bool(show_line_broadening), description="Line Broadening")
        lb_slider = FloatSlider(
            value=float(line_broadening_hz),
            min=0,
            max=20,
            step=0.5,
            description="LB (Hz)",
            continuous_update=False,
        )

        def widget_update(change=None):
            update_voxel(sx.value, sy.value)

        for widget in (sx, sy, cb_lb, lb_slider):
            widget.observe(widget_update, "value")

        display(HBox([sx, sy, cb_lb, lb_slider]))

    update_voxel(init_ix, init_iy)

    plt.tight_layout()
    plt.show()
    return fig