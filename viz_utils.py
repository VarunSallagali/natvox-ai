"""
Visualization utilities for NATVOX-AI
"""

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import Tuple
import streamlit as st


def plot_waveform_comparison(audio_before: np.ndarray, audio_after: np.ndarray, sr: int) -> plt.Figure:
    """
    Create side-by-side waveform comparison
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # Before
    time_before = np.arange(len(audio_before)) / sr
    axes[0].plot(time_before, audio_before, linewidth=0.5, color='#FF6B6B')
    axes[0].fill_between(time_before, audio_before, alpha=0.3, color='#FF6B6B')
    axes[0].set_ylabel('Amplitude', fontsize=10, fontweight='bold')
    axes[0].set_title('Original Audio Waveform', fontsize=12, fontweight='bold', color='#FF6B6B')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, max(time_before[-1], len(audio_after)/sr))
    
    # After
    time_after = np.arange(len(audio_after)) / sr
    axes[1].plot(time_after, audio_after, linewidth=0.5, color='#4ECDC4')
    axes[1].fill_between(time_after, audio_after, alpha=0.3, color='#4ECDC4')
    axes[1].set_xlabel('Time (seconds)', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Amplitude', fontsize=10, fontweight='bold')
    axes[1].set_title('Converted Audio Waveform', fontsize=12, fontweight='bold', color='#4ECDC4')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, max(time_before[-1], time_after[-1]))
    
    plt.tight_layout()
    return fig


def plot_spectrogram_comparison(audio_before: np.ndarray, audio_after: np.ndarray, sr: int) -> plt.Figure:
    """
    Create side-by-side spectrogram comparison
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Compute spectrograms
    S_before = librosa.feature.melspectrogram(y=audio_before, sr=sr, n_mels=128)
    S_before_db = librosa.power_to_db(S_before, ref=np.max)
    
    S_after = librosa.feature.melspectrogram(y=audio_after, sr=sr, n_mels=128)
    S_after_db = librosa.power_to_db(S_after, ref=np.max)
    
    # Before spectrogram
    img1 = librosa.display.specshow(
        S_before_db, 
        sr=sr, 
        x_axis='time', 
        y_axis='mel',
        ax=axes[0],
        cmap='viridis'
    )
    axes[0].set_title('Original Audio Spectrogram', fontsize=12, fontweight='bold', color='#FF6B6B')
    cbar1 = plt.colorbar(img1, ax=axes[0], format='%+2.0f dB')
    cbar1.set_label('Magnitude (dB)', fontsize=9)
    
    # After spectrogram
    img2 = librosa.display.specshow(
        S_after_db, 
        sr=sr, 
        x_axis='time', 
        y_axis='mel',
        ax=axes[1],
        cmap='viridis'
    )
    axes[1].set_title('Converted Audio Spectrogram', fontsize=12, fontweight='bold', color='#4ECDC4')
    cbar2 = plt.colorbar(img2, ax=axes[1], format='%+2.0f dB')
    cbar2.set_label('Magnitude (dB)', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_frequency_response(audio_before: np.ndarray, audio_after: np.ndarray, sr: int) -> plt.Figure:
    """
    Plot frequency response comparison
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Compute FFT
    n_fft = 2048
    fft_before = np.abs(np.fft.fft(audio_before, n=n_fft))[:n_fft//2]
    fft_after = np.abs(np.fft.fft(audio_after, n=n_fft))[:n_fft//2]
    
    freqs = np.fft.fftfreq(n_fft, 1/sr)[:n_fft//2]
    
    # Smooth with log scale
    fft_before_db = 20 * np.log10(fft_before + 1e-10)
    fft_after_db = 20 * np.log10(fft_after + 1e-10)
    
    # Apply smoothing
    from scipy.ndimage import uniform_filter1d
    fft_before_smooth = uniform_filter1d(fft_before_db, size=20)
    fft_after_smooth = uniform_filter1d(fft_after_db, size=20)
    
    ax.semilogx(freqs, fft_before_smooth, linewidth=2, label='Original', color='#FF6B6B', alpha=0.7)
    ax.semilogx(freqs, fft_after_smooth, linewidth=2, label='Converted', color='#4ECDC4', alpha=0.7)
    
    ax.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Magnitude (dB)', fontsize=11, fontweight='bold')
    ax.set_title('Frequency Response Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_xlim(20, sr/2)
    
    plt.tight_layout()
    return fig


def plot_loudness_envelope(audio_before: np.ndarray, audio_after: np.ndarray, sr: int, hop_length: int = 512) -> plt.Figure:
    """
    Plot loudness envelope over time
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Compute STFT
    S_before = librosa.stft(audio_before, hop_length=hop_length)
    S_before_mag = np.abs(S_before)
    loudness_before = np.sqrt(np.mean(S_before_mag**2, axis=0))
    
    S_after = librosa.stft(audio_after, hop_length=hop_length)
    S_after_mag = np.abs(S_after)
    loudness_after = np.sqrt(np.mean(S_after_mag**2, axis=0))
    
    # Convert to time
    times = librosa.frames_to_time(np.arange(len(loudness_before)), sr=sr, hop_length=hop_length)
    times_after = librosa.frames_to_time(np.arange(len(loudness_after)), sr=sr, hop_length=hop_length)
    
    ax.plot(times, loudness_before, linewidth=2, label='Original', color='#FF6B6B', alpha=0.7)
    ax.plot(times_after, loudness_after, linewidth=2, label='Converted', color='#4ECDC4', alpha=0.7)
    
    ax.set_xlabel('Time (seconds)', fontsize=11, fontweight='bold')
    ax.set_ylabel('RMS Loudness', fontsize=11, fontweight='bold')
    ax.set_title('Loudness Envelope', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    return fig


def create_metric_cards(before_metrics: dict, after_metrics: dict, conversion_metrics: dict) -> None:
    """
    Display metrics as Streamlit cards
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Original RMS",
            f"{before_metrics.get('rms', 0):.4f}",
            delta=f"{(after_metrics.get('rms', 0) - before_metrics.get('rms', 0)):.4f}"
        )
    
    with col2:
        st.metric(
            "Duration",
            f"{before_metrics.get('duration', 0):.2f}s",
            help="Audio length in seconds"
        )
    
    with col3:
        improvement = conversion_metrics.get('similarity_improvement', 0)
        st.metric(
            "Quality Change",
            f"{improvement:+.1%}",
            delta_color="off" if improvement >= 0 else "inverse"
        )
    
    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Pitch Shift",
            f"{conversion_metrics.get('pitch_shift_semitones', 0):+d}st",
            help="Semitones shifted"
        )
    
    with col2:
        speed = conversion_metrics.get('speed_factor', 1.0)
        speed_pct = (speed - 1.0) * 100
        st.metric(
            "Speed Change",
            f"{speed_pct:+.0f}%",
            help="Playback speed change"
        )
    
    with col3:
        st.metric(
            "Preset",
            conversion_metrics.get('preset', 'N/A')[:15],
            help="Applied voice preset"
        )
    
    with col4:
        st.metric(
            "Format",
            f"{before_metrics.get('sample_rate', 16000)} Hz",
            help="Sample rate in Hz"
        )


def create_comparison_section(audio_before: np.ndarray, audio_after: np.ndarray, sr: int, 
                             before_metrics: dict, after_metrics: dict, 
                             conversion_metrics: dict) -> None:
    """
    Create complete before/after comparison section
    """
    st.subheader("🎵 Before & After Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Original Audio**")
        st.audio(audio_before, sample_rate=sr, format='audio/wav')
        st.caption(f"Duration: {len(audio_before)/sr:.2f}s | Sample Rate: {sr}Hz")
    
    with col2:
        st.write("**Converted Audio**")
        st.audio(audio_after, sample_rate=sr, format='audio/wav')
        st.caption(f"Duration: {len(audio_after)/sr:.2f}s | Sample Rate: {sr}Hz")
    
    st.divider()
    
    # Download buttons
    col1, col2 = st.columns(2)
    
    with col1:
        # Convert to bytes for download
        import io
        buffer_before = io.BytesIO()
        sf.write(buffer_before, audio_before, sr, format='WAV')
        buffer_before.seek(0)
        
        st.download_button(
            label="📥 Download Original",
            data=buffer_before.getvalue(),
            file_name="original_audio.wav",
            mime="audio/wav"
        )
    
    with col2:
        buffer_after = io.BytesIO()
        sf.write(buffer_after, audio_after, sr, format='WAV')
        buffer_after.seek(0)
        
        st.download_button(
            label="📥 Download Converted",
            data=buffer_after.getvalue(),
            file_name="converted_audio.wav",
            mime="audio/wav"
        )
