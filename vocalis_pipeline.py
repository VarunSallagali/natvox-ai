import os
import numpy as np
import torch
from pydub import AudioSegment
from scipy.spatial.distance import cosine
from model import NATVOXAdapter
from audio_utils import get_embedding, enhance_audio
from voice_conversion import convert_audio


def text_to_wav(text, output_path, lang='en', slow=False):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150 if not slow else 120)
        voices = engine.getProperty('voices')
        if voices:
            engine.setProperty('voice', voices[0].id)  # Use first voice
        engine.save_to_file(text, output_path)
        engine.runAndWait()
    except ImportError:
        # Fallback to gTTS
        from gtts import gTTS
        tmp_mp3 = output_path.replace('.wav', '.mp3')
        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.save(tmp_mp3)

        audio = AudioSegment.from_mp3(tmp_mp3)
        audio.export(output_path, format='wav')

        try:
            os.remove(tmp_mp3)
        except OSError:
            pass

    return output_path


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


class AdaptedVoiceConversion:
    def __init__(self, adapter_model=None, encoder=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        if adapter_model is None:
            self.adapter = NATVOXAdapter().to(self.device)
            if os.path.exists('natvox_model.pth'):
                self.adapter.load_state_dict(torch.load('natvox_model.pth', map_location=self.device))
        else:
            self.adapter = adapter_model.to(self.device)

        self.encoder = encoder
        self.adapter.eval()

    def adapt_embedding(self, synthetic_embedding):
        self.adapter.eval()
        syn = torch.tensor(synthetic_embedding, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            adapted = self.adapter(syn)

        return adapted.cpu().numpy().squeeze(0)

    def convert_text(self, text, target_audio_path, tts_lang='en', slow=False, target_text=None):
        tmp_wav = 'temp_synthetic.wav'
        text_to_wav(text, tmp_wav, lang=tts_lang, slow=slow)

        # Generate target audio if not exists
        if not os.path.exists(target_audio_path):
            if target_text is None:
                target_text = "This is the target natural voice for comparison."
            text_to_wav(target_text, target_audio_path, lang=tts_lang, slow=slow)

        # Enhance the synthetic audio for clarity
        enhanced_wav = 'temp_enhanced.wav'
        enhance_audio(tmp_wav, enhanced_wav)

        synthetic_emb = get_embedding(enhanced_wav)
        target_emb = get_embedding(target_audio_path)

        adapted_emb = self.adapt_embedding(synthetic_emb)

        # Generate "converted" audio by applying voice conversion DSP
        converted_wav = 'temp_converted.wav'
        convert_audio(enhanced_wav, converted_wav, preset="Natural Clean")

        # Enhance the converted audio
        final_converted_wav = 'temp_final_converted.wav'
        enhance_audio(converted_wav, final_converted_wav)

        converted_emb = get_embedding(final_converted_wav)
        adapted_converted_emb = self.adapt_embedding(converted_emb)

        original_sim = compute_cosine_similarity(synthetic_emb, target_emb)
        adapted_sim = compute_cosine_similarity(adapted_emb, target_emb)
        converted_sim = compute_cosine_similarity(converted_emb, target_emb)
        adapted_converted_sim = compute_cosine_similarity(adapted_converted_emb, target_emb)

        result = {
            'SyntheticEmbedding': synthetic_emb,
            'AdaptedEmbedding': adapted_emb,
            'ConvertedEmbedding': converted_emb,
            'AdaptedConvertedEmbedding': adapted_converted_emb,
            'TargetEmbedding': target_emb,
            'OriginalSimilarity': original_sim,
            'AdaptedSimilarity': adapted_sim,
            'ConvertedSimilarity': converted_sim,
            'AdaptedConvertedSimilarity': adapted_converted_sim,
            'TempPath': tmp_wav,
            'EnhancedPath': enhanced_wav,
            'ConvertedPath': final_converted_wav,
            'TargetPath': target_audio_path
        }

        return result


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Vocalis voice conversion pipeline demo')
    parser.add_argument('--text', default='Hello from VOCALIS', help='Text prompt for synthetic voice')
    parser.add_argument('--target', default='target.wav', help='Target natural wav path (generated if not exists)')
    parser.add_argument('--target-text', default=None, help='Text for target voice if generating')
    parser.add_argument('--slow', action='store_true', help='Speak slowly')

    args = parser.parse_args()

    pipeline = AdaptedVoiceConversion()
    out = pipeline.convert_text(args.text, args.target, slow=args.slow, target_text=args.target_text)

    print('Result:')
    print(f"Original sim: {out['OriginalSimilarity']:.4f}")
    print(f"Adapted sim : {out['AdaptedSimilarity']:.4f}")
    print(f"Converted sim: {out['ConvertedSimilarity']:.4f}")
    print(f"Adapted Converted sim: {out['AdaptedConvertedSimilarity']:.4f}")
    print(f"Audio files:")
    print(f"  Enhanced: {out['EnhancedPath']}")
    print(f"  Converted: {out['ConvertedPath']}")
    print(f"  Target: {out['TargetPath']}")
