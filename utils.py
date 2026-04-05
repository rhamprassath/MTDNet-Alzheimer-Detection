import numpy as np
import torch
from scipy.signal import butter, filtfilt, welch

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply a Butterworth bandpass filter to EEG data using zero-phase filtering.
    data: (n_channels, n_samples)
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # Use filtfilt instead of lfilter to prevent biological phase shifting
    y = filtfilt(b, a, data, axis=-1)
    return y

def segment_signal(data, fs, segment_len_sec=1, stride_sec=None):
    """
    Split EEG signal into highly overlapping segments (Maximum Inference Strategy).
    data: (n_channels, total_samples)
    returns list of segments: (n_channels, fs * segment_len_sec)
    """
    segment_samples = int(fs * segment_len_sec)
    
    # Master-Level Upgrade: Overlapping Windows
    # If stride not provided, default to 50% overlap for massive resolution boost.
    if stride_sec is None:
        stride_sec = segment_len_sec / 2.0
        
    stride_samples = int(fs * stride_sec)
    total_samples = data.shape[1]
    
    segments = []
    start = 0
    while start + segment_samples <= total_samples:
        end = start + segment_samples
        segments.append(data[:, start:end])
        start += stride_samples
    
    return segments

def z_score_normalize(segment):
    """
    Apply Robust Per-Channel Z-score normalization with Clinical Artifact Suppression.
    segment: (n_channels, n_samples)
    """
    mean = segment.mean(axis=1, keepdims=True)
    std = segment.std(axis=1, keepdims=True)
    
    # Master-Level Upgrade: Winsorizing
    # High-amplitude artifacts (e.g. 500uV blinks) distort the Z-score.
    # We clip the pure signal at +/- 4 SDs so valid rhythms are preserved.
    # Avoid div by zero safely
    safe_std = std + 1e-8
    z_scored = (segment - mean) / safe_std
    np.clip(z_scored, -4.0, 4.0, out=z_scored) # In-place memory efficiency
    
    return z_scored

def preprocess_eeg(data, fs, lowcut=0.5, highcut=45.0, segment_len_sec=1, stride_sec=None):
    """
    Full preprocessing pipeline: Filter -> Segment (Overlapping) -> Normalize (Robust)
    """
    filtered_data = bandpass_filter(data, lowcut, highcut, fs)
    segments = segment_signal(filtered_data, fs, segment_len_sec=segment_len_sec, stride_sec=stride_sec)
    normalized_segments = [z_score_normalize(s) for s in segments]
    return normalized_segments

def get_alpha_theta_ratio(data, fs):
    """
    Physically measure the 'Slowing Factor' of the brain.
    Healthy: Lower Theta (4-8Hz), Higher Alpha (8-13Hz).
    Alzheimer's: Higher Theta, Lower Alpha.
    Returns: Average Theta/Alpha ratio across all channels.
    """
    from scipy.signal import welch
    
    # Define bands
    theta_band = (4, 8)
    alpha_band = (8, 13)
    
    # Calculate PSD using Welch's method
    # Adaptive Windowing: ensure nperseg never exceeds signal length
    n_samples = data.shape[-1]
    nperseg = min(n_samples, int(fs*2))
    if nperseg < 8: nperseg = 8 # minimum for Welch
    
    freqs, psd = welch(data, fs, nperseg=nperseg, axis=-1)
    
    # Find indices for bands
    theta_idx = np.logical_and(freqs >= theta_band[0], freqs <= theta_band[1])
    alpha_idx = np.logical_and(freqs >= alpha_band[0], freqs <= alpha_band[1])
    
    # Compute power in each band
    theta_power = np.mean(psd[:, theta_idx], axis=-1)
    alpha_power = np.mean(psd[:, alpha_idx], axis=-1)
    
    # Ratio Calculation (Avoid division by zero)
    alpha_power[alpha_power == 0] = 1e-9
    ratios = theta_power / alpha_power
    
    return np.mean(ratios)

def get_dominant_freq(data, fs):
    """
    Find the Peak Frequency in the Alpha-Theta range (4-15Hz).
    Healthy should have a peak strictly between 8-13Hz.
    Slowing (AD) pushes it to 4-7Hz.
    """
    from scipy.signal import welch
    n_samples = data.shape[-1]
    # Ensure window is not larger than data
    nperseg = min(n_samples, int(fs*2))
    if nperseg < 8: nperseg = 8
    
    freqs, psd = welch(data, fs, nperseg=nperseg, axis=-1)
    
    # Focus on Alpha-Theta area (4-15Hz)
    mask = np.logical_and(freqs >= 4, freqs <= 15)
    if not any(mask): return 10.0 # Default Alpha
    
    f_band = freqs[mask]
    p_band = psd[:, mask]
    
    # Peak per channel
    peaks = f_band[np.argmax(p_band, axis=-1)]
    return np.mean(peaks)

def pad_eeg_data(data, fs, target_sec=5):
    """
    Zero-pad the signal if it is shorter than target_sec.
    data: (n_channels, n_samples)
    """
    target_samples = int(fs * target_sec)
    current_samples = data.shape[1]
    
    if current_samples >= target_samples:
        return data
        
    padding_needed = target_samples - current_samples
    padding = np.zeros((data.shape[0], padding_needed))
    return np.hstack([data, padding])
