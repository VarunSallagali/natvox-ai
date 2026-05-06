from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import librosa
import soundfile as sf

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

        # ✅ Normalize again
        y = y / np.max(np.abs(y) + 1e-10)

        # ✅ Save output at higher quality
        sf.write(output_path, y, sr, subtype='PCM_16')

        return output_path

    except Exception as e:
        raise Exception(f"Error enhancing audio: {e}")