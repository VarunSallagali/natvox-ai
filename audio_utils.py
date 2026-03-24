from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np

encoder = None

def get_encoder():
    global encoder
    if encoder is None:
        encoder = VoiceEncoder()
    return encoder

def get_embedding(audio_path):
    try:
        wav = preprocess_wav(audio_path)
        encoder = get_encoder()
        embedding = encoder.embed_utterance(wav)
        return embedding
    except Exception as e:
        raise Exception(f"Error processing audio file {audio_path}: {e}")