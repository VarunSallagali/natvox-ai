from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import librosa
import soundfile as sf
from scipy import signal

encoder = None

# -----------------------------
# LOAD ENCODER (Singleton)
# -----------------------------
def get_encoder():
    global encoder
    if encoder is None:
        encoder = VoiceEncoder()
    return encoder


# -----------------------------
# GET EMBEDDING
# -----------------------------
def get_embedding(audio_path):
    try:
        wav = preprocess_wav(audio_path)
        encoder = get_encoder()
        embedding = encoder.embed_utterance(wav)
        return embedding
    except Exception as e:
        raise Exception(f"Error processing audio file {audio_path}: {e}")


# -----------------------------
# AUDIO ENHANCEMENT (IMPROVED)
# -----------------------------
def enhance_audio(input_path, output_path="output.wav"):
    try:
        y, sr = librosa.load(input_path, sr=None)

        # ✅ Normalize
        max_val = np.max(np.abs(y))
        if max_val > 0:
            y = y / max_val

        # ✅ Pre-emphasis for clarity
        y = librosa.effects.preemphasis(y, coef=0.97)

        # ✅ Light noise reduction with median filter
        from scipy.ndimage import median_filter
        y = median_filter(y, size=3)

        # ✅ Light low-pass filter for smoothness
        from scipy.signal import butter, filtfilt
        b, a = butter(4, 8000, 'low', fs=sr)
        y = filtfilt(b, a, y)

        # ✅ Add dynamic range compression for smoother dynamics
        from scipy.signal import lfilter
        # Simple compressor: reduce peaks
        threshold = 0.6
        ratio = 4.0
        attack = 0.01
        release = 0.1

        # Envelope follower
        env = np.abs(signal.lfilter([1], [1, -0.99], y))
        # Compress
        gain = np.where(env > threshold, (env / threshold) ** (1/ratio - 1), 1)
        y = y * gain

        # ✅ De-essing to reduce harsh sibilance
        # High-pass filter for sibilance (around 5kHz)
        b_hp, a_hp = butter(2, 5000, 'high', fs=sr)
        sibilance = filtfilt(b_hp, a_hp, y)
        # Reduce sibilance by 3dB
        sibilance_reduced = sibilance * 0.7
        # Reconstruct
        y = y - sibilance + sibilance_reduced

        # ✅ Final smoothing with light reverb for naturalness
        # Simple reverb simulation
        delay_samples = int(0.02 * sr)  # 20ms delay
        decay = 0.3
        reverb = np.zeros_like(y)
        reverb[delay_samples:] = y[:-delay_samples] * decay
        y = y + reverb * 0.1  # Mix 10% reverb

        # ✅ Normalize again
        y = y / np.max(np.abs(y) + 1e-10)

        # ✅ Save output at higher quality
        sf.write(output_path, y, sr, subtype='PCM_16')

        return output_path

    except Exception as e:
        raise Exception(f"Error enhancing audio: {e}")