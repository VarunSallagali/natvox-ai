"""
Voice Conversion Module
Handles DSP-based voice transformation with presets
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from typing import Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

# Voice Preset Definitions
VOICE_PRESETS = {
    "Natural Clean": {
        "pitch_shift": 0,
        "speed_factor": 0.98,
        "formant_shift": 0,
        "eq_low": 1.05,
        "eq_mid": 1.05,
        "eq_high": 1.3,
        "clarity": 0.3,
        "reverb": 0.02,
        "compression": 0.2,
        "description": "Clean, natural presentation with enhanced clarity"
    },
    "Deep Narrator": {
        "pitch_shift": -6,
        "speed_factor": 0.92,
        "formant_shift": -200,
        "eq_low": 1.4,
        "eq_mid": 0.9,
        "eq_high": 0.75,
        "clarity": 0.25,
        "reverb": 0.15,
        "compression": 0.6,
        "description": "Deep, resonant voice for narration"
    },
    "Female Soft": {
        "pitch_shift": 5,
        "speed_factor": 1.08,
        "formant_shift": 190,
        "eq_low": 0.8,
        "eq_mid": 1.25,
        "eq_high": 1.45,
        "clarity": 0.45,
        "reverb": 0.14,
        "compression": 0.45,
        "description": "Warm, feminine tone"
    },
    "Robot AI": {
        "pitch_shift": 6,
        "speed_factor": 0.96,
        "formant_shift": 120,
        "eq_low": 1.05,
        "eq_mid": 1.0,
        "eq_high": 1.5,
        "clarity": 0.7,
        "reverb": 0.12,
        "compression": 0.9,
        "description": "Synthetic, crisp robotic quality"
    },
    "Energetic Speaker": {
        "pitch_shift": 3,
        "speed_factor": 1.12,
        "formant_shift": 80,
        "eq_low": 0.95,
        "eq_mid": 1.35,
        "eq_high": 1.45,
        "clarity": 0.55,
        "reverb": 0.12,
        "compression": 0.7,
        "description": "Bright, energetic delivery"
    },
    "Child Voice": {
        "pitch_shift": 8,
        "speed_factor": 1.12,
        "formant_shift": 250,
        "eq_low": 0.7,
        "eq_mid": 1.35,
        "eq_high": 1.6,
        "clarity": 0.6,
        "reverb": 0.12,
        "compression": 0.5,
        "description": "Higher pitch, playful tone"
    }
}


def apply_pitch_shift(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Apply pitch shifting using librosa"""
    if n_steps == 0:
        return audio
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def apply_time_stretch(audio: np.ndarray, factor: float) -> np.ndarray:
    """Apply time stretching (speed change without pitch change)"""
    if factor == 1.0:
        return audio
    return librosa.effects.time_stretch(audio, rate=factor)


def apply_formant_shift(audio: np.ndarray, sr: int, shift_amount: float) -> np.ndarray:
    """
    Simulate formant shifting by resampling and pitch-correcting
    This creates a vocal character shift without changing pitch
    """
    if shift_amount == 0:
        return audio
    
    # Convert shift to resampling ratio
    ratio = 2 ** (shift_amount / 1200)  # Convert cents to ratio
    
    # Resample to shift formants
    audio_shifted = librosa.resample(audio, orig_sr=sr, target_sr=int(sr / ratio))
    
    # Resample back to original sr (pitch goes back but formants stay shifted)
    audio_shifted = librosa.resample(audio_shifted, orig_sr=int(sr / ratio), target_sr=sr)
    
    # Match lengths
    if len(audio_shifted) != len(audio):
        if len(audio_shifted) > len(audio):
            audio_shifted = audio_shifted[:len(audio)]
        else:
            audio_shifted = np.pad(audio_shifted, (0, len(audio) - len(audio_shifted)))
    
    return audio_shifted


def apply_eq(audio: np.ndarray, sr: int, eq_low: float, eq_mid: float, eq_high: float) -> np.ndarray:
    """
    Apply 3-band parametric EQ
    """
    # Design 3 filters: low, mid, high
    sos_low = signal.butter(2, 200, 'low', fs=sr, output='sos')
    sos_mid = signal.butter(2, [200, 2000], 'band', fs=sr, output='sos')
    sos_high = signal.butter(2, 2000, 'high', fs=sr, output='sos')
    
    # Apply filters
    low = signal.sosfilt(sos_low, audio) * eq_low
    mid = signal.sosfilt(sos_mid, audio) * eq_mid
    high = signal.sosfilt(sos_high, audio) * eq_high
    
    # Mix (with slight gain reduction to prevent clipping)
    output = (low + mid + high) / 3.0
    
    return output


def apply_clarity(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """
    Apply high-pass emphasis for clarity
    """
    if amount <= 0:
        return audio
    
    # High-pass filter for clarity
    sos = signal.butter(1, 1500, 'high', fs=sr, output='sos')
    high_passed = signal.sosfilt(sos, audio)
    
    # Blend with original
    output = audio + high_passed * amount
    
    return output


def apply_simple_reverb(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    """
    Apply simple reverb using multiple delayed copies
    """
    if amount <= 0:
        return audio
    
    output = audio.copy()
    
    # Add delayed copies at different times
    delays = [int(sr * 0.02), int(sr * 0.04), int(sr * 0.06)]
    decay_factors = [0.4, 0.25, 0.15]
    
    for delay, decay in zip(delays, decay_factors):
        if delay < len(audio):
            delayed = np.pad(audio, (delay, 0))[:len(audio)]
            output += delayed * decay * amount
    
    return output


def apply_compression(audio: np.ndarray, threshold: float = -20, ratio: float = 4) -> np.ndarray:
    """
    Apply dynamic range compression
    """
    # Convert to dB
    eps = 1e-10
    audio_db = 20 * np.log10(np.abs(audio) + eps)
    
    # Apply compression to peaks above threshold
    mask = audio_db > threshold
    audio_db[mask] = threshold + (audio_db[mask] - threshold) / ratio
    
    # Convert back
    output = np.sign(audio) * 10 ** (audio_db / 20)
    
    return output


def normalize_audio(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    """
    Normalize audio to target loudness (LUFS approximation)
    """
    # Simple LUFS-like normalization
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 1e-10:
        target_linear = 10 ** (target_db / 20)
        audio = audio * (target_linear / rms)
    
    # Soft clipping to prevent distortion
    audio = np.tanh(audio * 0.9) / np.tanh(0.9)
    
    return audio


def apply_preset(audio: np.ndarray, sr: int, preset_name: str, intensity: float = 1.0) -> Tuple[np.ndarray, Dict]:
    """
    Apply voice conversion preset to audio
    Returns: (processed_audio, metrics)
    """
    if preset_name not in VOICE_PRESETS:
        preset_name = "Natural Clean"
    
    preset = VOICE_PRESETS[preset_name]
    output = audio.copy()

    # Scale effect intensity
    pitch_shift = int(round(preset["pitch_shift"] * intensity))
    speed_factor = 1.0 + np.clip((preset["speed_factor"] - 1.0) * intensity, -0.2, 0.2)
    formant_shift = int(round(preset["formant_shift"] * intensity))
    clarity = float(np.clip(preset["clarity"] * intensity, 0.0, 1.2))
    reverb = float(np.clip(preset["reverb"] * intensity, 0.0, 0.3))
    compression = float(np.clip(preset["compression"] * intensity, 0.0, 1.0))

    # Apply transformations in order
    # 1. Pitch shift
    if pitch_shift != 0:
        output = apply_pitch_shift(output, sr, pitch_shift)

    # 2. Time stretch
    if abs(speed_factor - 1.0) > 1e-6:
        output = apply_time_stretch(output, speed_factor)

    # 3. Formant shift
    if formant_shift != 0:
        output = apply_formant_shift(output, sr, formant_shift)

    # 4. EQ
    output = apply_eq(output, sr, preset["eq_low"], preset["eq_mid"], preset["eq_high"])

    # 5. Clarity
    if clarity > 0:
        output = apply_clarity(output, sr, clarity)

    # 6. Compression
    if compression > 0:
        output = apply_compression(output, threshold=-20, ratio=2 + compression * 3)

    # 7. Reverb
    if reverb > 0:
        output = apply_simple_reverb(output, sr, reverb)

    # 8. Normalize
    output = normalize_audio(output, target_db=-3.0)
    
    # Ensure no clipping
    max_val = np.max(np.abs(output))
    if max_val > 1.0:
        output = output / (max_val * 1.05)
    
    # Calculate metrics
    original_rms = np.sqrt(np.mean(audio ** 2))
    output_rms = np.sqrt(np.mean(output ** 2))
    
    metrics = {
        "preset": preset_name,
        "pitch_shift_semitones": preset["pitch_shift"],
        "speed_factor": preset["speed_factor"],
        "original_rms": float(original_rms),
        "output_rms": float(output_rms),
        "description": preset["description"]
    }
    
    return output, metrics


def get_preset_list():
    """Get list of available presets"""
    return list(VOICE_PRESETS.keys())


def get_preset_description(preset_name: str) -> str:
    """Get description of a preset"""
    if preset_name in VOICE_PRESETS:
        return VOICE_PRESETS[preset_name]["description"]
    return "Unknown preset"


def convert_audio(input_path: str, output_path: str, preset: str = "Natural Clean", intensity: float = 1.0):
    """
    Convert audio file using a voice preset
    """
    # Load audio
    audio, sr = librosa.load(input_path, sr=None)
    
    # Apply preset
    converted, metrics = apply_preset(audio, sr, preset, intensity=intensity)
    metrics["intensity"] = float(intensity)
    
    # Save output
    sf.write(output_path, converted, sr)
    
    return output_path, metrics
