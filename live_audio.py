"""
Live audio input/output module for real-time NATVOX-AI processing
Captures audio streams, processes embeddings, and provides real-time metrics
"""

import sounddevice as sd
import numpy as np
import queue
import torch
import threading
from collections import deque
from audio_utils import get_encoder
import librosa


def safe_preprocess_wav(audio_data, sr=16000):
    """
    Preprocess audio for embedding without requiring webrtcvad.
    Simpler version that just normalizes and frames the audio.
    """
    try:
        # Try to use resemblyzer's preprocessing if available
        from resemblyzer import preprocess_wav
        return preprocess_wav(audio_data, sr)
    except ImportError:
        # Fallback: simple normalization
        audio = np.asarray(audio_data, dtype=np.float32)
        audio = librosa.util.normalize(audio)
        return audio


class LiveAudioProcessor:
    """
    Handles real-time audio capture, embedding extraction, and adaptation
    """
    
    def __init__(self, model, sample_rate=16000, chunk_size=2048, buffer_duration=2.0):
        """
        Args:
            model: NATVOXAdapter model (should be in eval mode)
            sample_rate: Audio sample rate (Hz)
            chunk_size: Size of audio chunks to process
            buffer_duration: Duration to buffer for embedding extraction (seconds)
        """
        self.model = model
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_duration = buffer_duration
        self.buffer_samples = int(sample_rate * buffer_duration)
        
        # Audio buffers
        self.audio_buffer = deque(maxlen=self.buffer_samples)
        self.output_buffer = np.array([])
        
        # Threading
        self.is_running = False
        self.audio_queue = queue.Queue()
        self.results_queue = queue.Queue()
        
        # Metrics
        self.metrics = {
            'processed_chunks': 0,
            'adapted_embeddings': [],
            'similarities': [],
            'input_rms': [],
        }
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for real-time audio input stream"""
        if status:
            print(f"Audio stream status: {status}")
        
        # Copy audio data to queue
        audio_chunk = indata[:, 0].copy()
        self.audio_queue.put(audio_chunk)
    
    def start_capture(self, device=None):
        """Start capturing audio from input device"""
        self.is_running = True
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            device=device,
            latency='low'
        )
        self.stream.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print(f"🎤 Live audio capture started (sample_rate={self.sample_rate}Hz)")
    
    def stop_capture(self):
        """Stop capturing audio"""
        self.is_running = False
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
        print("🛑 Live audio capture stopped")
    
    def _process_loop(self):
        """Main processing loop for audio chunks"""
        while self.is_running:
            try:
                # Get audio chunk from queue
                chunk = self.audio_queue.get(timeout=0.5)
                
                # Add to buffer
                for sample in chunk:
                    self.audio_buffer.append(sample)
                
                # Process when buffer is full
                if len(self.audio_buffer) >= self.buffer_samples:
                    self._process_buffer()
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")
    
    def _process_buffer(self):
        """Process buffered audio: extract embedding, adapt, compute metrics"""
        try:
            # Convert buffer to numpy array
            audio_data = np.array(list(self.audio_buffer), dtype=np.float32)
            
            # Normalize
            rms = np.sqrt(np.mean(audio_data ** 2))
            self.metrics['input_rms'].append(float(rms))
            
            if rms < 1e-4:  # Skip silent frames
                return
            
            # Preprocess audio for embedding (similar to resemblyzer)
            # Resample if needed to match encoder expectations
            audio_normalized = audio_data / (np.max(np.abs(audio_data)) + 1e-8)
            
            # Extract embedding using encoder
            encoder = get_encoder()
            try:
                # Use safe preprocessing
                wav_preprocessed = safe_preprocess_wav(audio_data, self.sample_rate)
                embedding = encoder.embed_utterance(wav_preprocessed)
            except Exception as e:
                print(f"Embedding extraction error: {e}")
                return
            
            # Convert to tensor and process with model
            embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                adapted_embedding = self.model(embedding_tensor)
            
            adapted_emb_np = adapted_embedding.squeeze(0).cpu().numpy()
            
            # Store results
            self.metrics['processed_chunks'] += 1
            self.metrics['adapted_embeddings'].append(adapted_emb_np)
            
            # Calculate cosine similarity (original vs adapted)
            original_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
            adapted_norm = adapted_emb_np / (np.linalg.norm(adapted_emb_np) + 1e-8)
            similarity = float(np.dot(original_norm, adapted_norm))
            self.metrics['similarities'].append(similarity)
            
            # Put results in queue for UI
            self.results_queue.put({
                'chunk': self.metrics['processed_chunks'],
                'original_embedding': embedding,
                'adapted_embedding': adapted_emb_np,
                'similarity': similarity,
                'rms': rms,
                'timestamp': len(self.metrics['input_rms'])
            })
            
        except Exception as e:
            print(f"Error processing buffer: {e}")
    
    def get_latest_results(self, timeout=0.1):
        """Get latest processing results from queue"""
        try:
            return self.results_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_metrics_summary(self):
        """Get summary of all processed metrics"""
        if not self.metrics['similarities']:
            return {
                'chunks_processed': 0,
                'avg_similarity': 0,
                'min_similarity': 0,
                'max_similarity': 0,
                'avg_input_rms': 0,
            }
        
        similarities = self.metrics['similarities']
        return {
            'chunks_processed': self.metrics['processed_chunks'],
            'avg_similarity': float(np.mean(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities)),
            'std_similarity': float(np.std(similarities)),
            'avg_input_rms': float(np.mean(self.metrics['input_rms'])) if self.metrics['input_rms'] else 0,
        }
    
    def reset_metrics(self):
        """Reset metrics for new session"""
        self.metrics = {
            'processed_chunks': 0,
            'adapted_embeddings': [],
            'similarities': [],
            'input_rms': [],
        }


class LiveAudioPlayback:
    """
    Handles real-time audio playback
    """
    
    def __init__(self, sample_rate=16000, channels=1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.stream = None
    
    def start_playback(self, device=None):
        """Start audio output stream"""
        self.stream = sd.OutputStream(
            channels=self.channels,
            samplerate=self.sample_rate,
            device=device,
            latency='low'
        )
        self.stream.start()
        print("🔊 Playback stream started")
    
    def play_audio(self, audio_data, blocking=False):
        """Play audio data"""
        if self.stream is None:
            self.start_playback()
        
        if isinstance(audio_data, np.ndarray):
            # Normalize to avoid clipping
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / (max_val + 1e-8)
            
            sd.play(audio_data, samplerate=self.sample_rate, blocking=blocking)
    
    def stop_playback(self):
        """Stop audio playback stream"""
        if self.stream is not None:
            sd.stop()
            self.stream.stop()
            self.stream.close()
            print("🛑 Playback stream stopped")


# Utility functions for Streamlit integration
def get_available_input_devices():
    """Get list of available audio input devices"""
    devices = sd.query_devices()
    input_devices = {}
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices[device['name']] = i
    
    return input_devices


def get_available_output_devices():
    """Get list of available audio output devices"""
    devices = sd.query_devices()
    output_devices = {}
    
    for i, device in enumerate(devices):
        if device['max_output_channels'] > 0:
            output_devices[device['name']] = i
    
    return output_devices
