import json
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import librosa
import soundfile as sf
from pydub import AudioSegment

from audio_utils import get_embedding
from config import ARTIFACTS_DIR, SUPPORTED_AUDIO_EXTENSIONS
from model import NATVOXAdapter
from voice_conversion import apply_preset, convert_audio, get_preset_list, get_preset_description
from viz_utils import (
    plot_waveform_comparison, 
    plot_spectrogram_comparison,
    plot_frequency_response,
    plot_loudness_envelope,
    create_metric_cards,
    create_comparison_section
)

# Conditional import for live audio (may not be available on Streamlit Cloud)
LIVE_AUDIO_AVAILABLE = False
try:
    from live_audio import LiveAudioProcessor, get_available_input_devices
    LIVE_AUDIO_AVAILABLE = True
except (ImportError, OSError):
    # PortAudio not available (common on Streamlit Cloud)
    pass


DEFAULT_ARTIFACTS = ARTIFACTS_DIR
SUPPORTED_UPLOADS = [ext.lstrip(".") for ext in SUPPORTED_AUDIO_EXTENSIONS]


@st.cache_resource
def load_model(model_path):
    model = NATVOXAdapter()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model


def convert_to_wav(file_path):
    suffix = Path(file_path).suffix.lower()
    if suffix != ".wav":
        audio = AudioSegment.from_file(file_path)
        new_path = f"{Path(file_path).stem}_converted.wav"
        audio.export(new_path, format="wav")
        return new_path
    return file_path


def save_audio_array(audio, sr, output_path):
    sf.write(output_path, audio, sr)
    return output_path


def save_uploaded_file(upload):
    suffix = Path(upload.name).suffix or ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.getvalue())
        return tmp.name


def cosine_similarity(a, b):
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))


def read_results_json(results_path):
    if not results_path.exists():
        return None
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_training_command(data_dir, output_dir, epochs, batch_size):
    cmd = [
        "python",
        "main.py",
        "--mode",
        "train",
        "--output-dir",
        str(output_dir),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
    ]
    if data_dir:
        cmd.extend(["--data-dir", data_dir])

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


st.set_page_config(page_title="NATVOX-AI Studio", layout="wide")
st.title("🎙️ NATVOX-AI Studio")
st.caption("Professional Voice Adaptation & Embedding Alignment Platform")

with st.sidebar:
    st.header("🧭 Navigation")
    nav_options = [
        "🏠 Overview", 
        "🎯 Voice Conversion",
        "📊 Training", 
        "📈 Results", 
        "🔬 Inference Lab",
        "⚙️ Dataset Setup"
    ]
    
    # Add live inference only if available
    if LIVE_AUDIO_AVAILABLE:
        nav_options.insert(5, "🎤 Live Inference")
    
    page = st.radio("Go to", nav_options)
    st.divider()
    artifacts_dir = Path(
        st.text_input("Artifacts directory", str(DEFAULT_ARTIFACTS))
    )


if page == "🏠 Overview":
    st.subheader("Welcome to NATVOX-AI Studio")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 🎯 Main Features
        - **Voice Conversion**: Transform any audio with professional voice presets
        - **Real-time Adaptation**: Live microphone streaming with metrics
        - **Embedding Analysis**: Deep dive into speaker embeddings
        - **Training Dashboard**: Build custom voice adapters
        - **Visual Analysis**: Waveforms, spectrograms, frequency analysis
        """)
    
    with col2:
        st.markdown("""
        ### ✨ Capabilities
        - 🎤 Professional voice transformation
        - 📊 Similarity metrics & improvements
        - 📈 Training graphs & analytics
        - 🔊 Real-time audio processing
        - 💾 Download converted audio
        """)
    
    st.divider()
    st.info("Start by uploading audio in **Voice Conversion** tab or train a model in **Training** tab.")

elif page == "🎯 Voice Conversion":
    st.subheader("🎤 Voice Conversion Studio")
    st.write("Transform your audio with professional voice presets. Upload any audio file and choose a preset to hear the difference.")
    
    # Upload section
    st.write("### 📁 Upload Audio")
    uploaded_file = st.file_uploader("Select audio file", type=SUPPORTED_UPLOADS)
    
    if uploaded_file:
        # Load audio
        audio_file_path = save_uploaded_file(uploaded_file)
        audio_file_path = convert_to_wav(audio_file_path)
        
        try:
            audio_data, sr = librosa.load(audio_file_path, sr=16000)
            
            # Preset selection
            st.write("### 🎨 Voice Preset")
            preset_list = get_preset_list()
            selected_preset = st.selectbox(
                "Choose a voice preset",
                preset_list,
                help="Select a preset to transform your voice"
            )
            
            preset_desc = get_preset_description(selected_preset)
            st.caption(f"📝 {preset_desc}")

            effect_strength = st.slider(
                "Conversion strength",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Increase to make the voice transformation more noticeable."
            )
            
            # Show preset parameters in an expander
            with st.expander("View Preset Details"):
                from voice_conversion import VOICE_PRESETS
                preset_params = VOICE_PRESETS[selected_preset]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pitch Shift", f"{preset_params['pitch_shift']:+d} st")
                    st.metric("Speed", f"{preset_params['speed_factor']:.2f}x")
                    st.metric("EQ Low", f"{preset_params['eq_low']:.1f}")
                with col2:
                    st.metric("Formant", f"{preset_params['formant_shift']:+d} cents")
                    st.metric("Clarity", f"{preset_params['clarity']:.2f}")
                    st.metric("EQ Mid", f"{preset_params['eq_mid']:.1f}")
                with col3:
                    st.metric("Reverb", f"{preset_params['reverb']:.2f}")
                    st.metric("Compression", f"{preset_params['compression']:.2f}")
                    st.metric("EQ High", f"{preset_params['eq_high']:.1f}")
            
            # Convert button
            if st.button("🚀 Apply Voice Conversion", type="primary"):
                with st.spinner("🔄 Converting your voice..."):
                    # Save current audio if needed for conversion and playback
                    converted_file_path = Path(audio_file_path).with_name(
                        f"{Path(audio_file_path).stem}_{selected_preset.replace(' ', '_')}.wav"
                    )

                    if Path(audio_file_path).suffix.lower() != ".wav":
                        audio_file_path = convert_to_wav(audio_file_path)

                    converted_audio_path, conversion_metrics = convert_audio(
                        audio_file_path,
                        str(converted_file_path),
                        selected_preset,
                        intensity=effect_strength,
                    )

                    converted_audio, _ = librosa.load(str(converted_audio_path), sr=sr)

                    # Calculate before/after metrics
                    before_metrics = {
                        'rms': np.sqrt(np.mean(audio_data ** 2)),
                        'duration': len(audio_data) / sr,
                        'sample_rate': sr,
                        'peak': np.max(np.abs(audio_data))
                    }

                    after_metrics = {
                        'rms': np.sqrt(np.mean(converted_audio ** 2)),
                        'duration': len(converted_audio) / sr,
                        'sample_rate': sr,
                        'peak': np.max(np.abs(converted_audio))
                    }

                    conversion_metrics['similarity_improvement'] = (
                        (after_metrics['rms'] - before_metrics['rms'])
                        / (before_metrics['rms'] + 1e-10)
                    )

                    st.success("✅ Conversion complete!")

                    # Output audio players
                    st.write("### 🔊 Before & After")
                    player_col1, player_col2 = st.columns(2)
                    with player_col1:
                        st.markdown("**Original Audio**")
                        st.audio(audio_file_path)
                    with player_col2:
                        st.markdown("**Converted Audio**")
                        st.audio(str(converted_audio_path))

                    download_col1, download_col2 = st.columns(2)
                    with download_col2:
                        with open(str(converted_audio_path), "rb") as f:
                            st.download_button(
                                label="Download converted audio",
                                data=f,
                                file_name=converted_audio_path.name,
                                mime="audio/wav"
                            )

                    # Display metrics
                    st.write("### 📊 Metrics & Analysis")
                    create_metric_cards(before_metrics, after_metrics, conversion_metrics)

                    # Comparison section
                    st.divider()
                    create_comparison_section(audio_data, converted_audio, sr, before_metrics, after_metrics, conversion_metrics)

                    # Visual Analysis
                    st.write("### 📈 Visual Analysis")
                    tabs = st.tabs(["Waveforms", "Spectrograms", "Frequency Response", "Loudness"])

                    with tabs[0]:
                        st.write("**Waveform Comparison**")
                        fig = plot_waveform_comparison(audio_data, converted_audio, sr)
                        st.pyplot(fig)

                    with tabs[1]:
                        st.write("**Spectrogram Comparison**")
                        with st.spinner("Generating spectrograms..."):
                            fig = plot_spectrogram_comparison(audio_data, converted_audio, sr)
                            st.pyplot(fig)

                    with tabs[2]:
                        st.write("**Frequency Response**")
                        with st.spinner("Analyzing frequency response..."):
                            fig = plot_frequency_response(audio_data, converted_audio, sr)
                            st.pyplot(fig)

                    with tabs[3]:
                        st.write("**Loudness Envelope**")
                        with st.spinner("Computing loudness envelope..."):
                            fig = plot_loudness_envelope(audio_data, converted_audio, sr)
                            st.pyplot(fig)

                    # Session summary
                    st.divider()
                    st.write("### 📋 Conversion Summary")
                    summary_col1, summary_col2 = st.columns(2)

                    with summary_col1:
                        st.write(f"""
                        **Original Audio**
                        - Duration: {before_metrics['duration']:.2f}s
                        - RMS Level: {before_metrics['rms']:.4f}
                        - Peak: {before_metrics['peak']:.4f}
                        - Sample Rate: {sr}Hz
                        """)
                    with summary_col2:
                        st.write(f"""
                        **Converted Audio**
                        - Duration: {after_metrics['duration']:.2f}s
                        - RMS Level: {after_metrics['rms']:.4f}
                        - Peak: {after_metrics['peak']:.4f}
                        - Preset: {selected_preset}
                        - Strength: {effect_strength:.1f}x
                        - Saved file: `{converted_audio_path.name}`
                        """)
        
        except Exception as e:
            st.error(f"❌ Error processing audio: {str(e)}")
            st.info("Please ensure the file is a valid audio format (.wav, .mp3, .m4a, .flac, .ogg)")
    else:
        st.info("👆 Upload an audio file to get started!")

elif page == "⚙️ Dataset Setup":
    st.subheader("How to add your voice dataset")
    st.markdown(
        """
        Create a paired dataset folder like this:

        ```
        your_dataset/
          synthetic/
            sample_001.wav
            sample_002.wav
          natural/
            sample_001.wav
            sample_002.wav
        ```

        Rules:
        - `synthetic/` and `natural/` must both exist.
        - File names must match by stem (`sample_001` in both folders).
        - Supported formats: `.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`.
        - Minimum recommended pairs: 30.
        """
    )
    st.success(
        "Place this dataset anywhere on disk, then provide that folder path on the Training page."
    )

elif page == "📊 Training":
    st.subheader("🏋️ Train NATVOX-AI Adapter")
    st.write("Train a custom embedding adapter on your voice data.")
    
    dataset_dir = st.text_input(
        "Dataset directory (contains synthetic/ and natural/)", value=""
    ).strip()
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input("Epochs", min_value=1, max_value=500, value=50)
    with col2:
        batch_size = st.number_input("Batch size", min_value=1, max_value=128, value=5)

    if st.button("▶️ Start Training", type="primary"):
        with st.spinner("Training in progress..."):
            result = run_training_command(
                dataset_dir or None,
                artifacts_dir,
                epochs,
                batch_size,
            )

        if result.returncode == 0:
            st.success("✅ Training finished successfully!")
            st.code(result.stdout[-4000:] or "Training completed.")
        else:
            st.error("❌ Training failed.")
            st.code(result.stderr or result.stdout or "No logs available.")

elif page == "📈 Results":
    st.subheader("📊 Training Results & Analysis")
    results_json = artifacts_dir / "results.json"
    payload = read_results_json(results_json)

    if not payload:
        st.warning(
            f"❌ No results found at {results_json}. Run training first from the Training page."
        )
    else:
        metrics = payload.get("metrics", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Before Cosine", f"{metrics.get('before_cosine', 0.0):.4f}")
        c2.metric("After Cosine", f"{metrics.get('after_cosine', 0.0):.4f}")
        c3.metric("📈 Improvement", f"{metrics.get('cosine_improvement', 0.0):+.4f}")

        c4, c5 = st.columns(2)
        c4.metric("Before MSE", f"{metrics.get('before_mse', 0.0):.6f}")
        c5.metric("After MSE", f"{metrics.get('after_mse', 0.0):.6f}")

        st.write("**Dataset Source:**", payload.get("data_dir"))

        loss_curve = artifacts_dir / "loss_curve.png"
        sim_bar = artifacts_dir / "similarity_bar.png"
        
        col1, col2 = st.columns(2)
        with col1:
            if loss_curve.exists():
                st.image(str(loss_curve), caption="Loss Curves", use_column_width=True)
        with col2:
            if sim_bar.exists():
                st.image(str(sim_bar), caption="Similarity Comparison", use_column_width=True)

        history = payload.get("history", {})
        if history.get("sim"):
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(history["sim"], linewidth=2, color='#4ECDC4', label="train cosine similarity")
            ax.fill_between(range(len(history["sim"])), history["sim"], alpha=0.2, color='#4ECDC4')
            ax.set_xlabel("Epoch", fontsize=11, fontweight='bold')
            ax.set_ylabel("Similarity", fontsize=11, fontweight='bold')
            ax.set_title("Cosine Similarity Trajectory", fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

elif page == "🔬 Inference Lab":
    st.subheader("🔬 Single-Sample Embedding Adaptation")
    st.write("Test the NATVOX adapter on a single pair of audio files.")
    
    model_path_input = st.text_input(
        "Model checkpoint path", value=str(artifacts_dir / "natvox_model.pth")
    )
    synthetic_audio = st.file_uploader(
        "Upload synthetic reference audio", type=SUPPORTED_UPLOADS
    )
    target_audio = st.file_uploader(
        "Upload natural target audio (for similarity comparison)",
        type=SUPPORTED_UPLOADS,
    )

    if synthetic_audio and target_audio and st.button("▶️ Run Adaptation", type="primary"):
        model_path = Path(model_path_input)
        if not model_path.exists():
            st.error(f"❌ Checkpoint not found: {model_path}")
            st.stop()

        model = load_model(str(model_path))
        synthetic_file = convert_to_wav(save_uploaded_file(synthetic_audio))
        target_file = convert_to_wav(save_uploaded_file(target_audio))

        try:
            syn_emb = get_embedding(synthetic_file)
            tgt_emb = get_embedding(target_file)
            syn_tensor = torch.tensor(syn_emb, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                adapted = model(syn_tensor).squeeze(0).cpu().numpy()

            before = cosine_similarity(syn_emb, tgt_emb)
            after = cosine_similarity(adapted, tgt_emb)
            improvement = after - before

            col1, col2, col3 = st.columns(3)
            col1.metric("Before", f"{before:.4f}")
            col2.metric("After", f"{after:.4f}")
            col3.metric("✨ Improvement", f"{improvement:+.4f}")

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#FF6B6B", "#4ECDC4"]
            bars = ax.bar(["Before", "After"], [before, after], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Cosine Similarity", fontsize=11, fontweight='bold')
            ax.set_title("Embedding Adaptation Result", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.2, axis='y')
            
            # Add value labels on bars
            for bar, val in zip(bars, [before, after]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.4f}', ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
        except Exception as exc:
            st.error(f"❌ Inference failed: {exc}")

elif page == "🎤 Live Inference":
    if not LIVE_AUDIO_AVAILABLE:
        st.error("❌ Live Audio Not Available")
        st.warning(
            "Live audio streaming requires PortAudio library which is not available "
            "in this environment. This feature works best when running locally:\n\n"
            "```bash\npython live_inference.py --checkpoint artifacts/natvox_model.pth\n```\n\n"
            "For cloud deployment, use the **Inference Lab** tab instead."
        )
    else:
        st.subheader("🎤 Live Audio Adaptation")
        st.write(
            """
            Stream audio in real-time from your microphone and see embedding adaptations 
            and similarity metrics update live. This uses the trained NATVOX adapter to 
            transform your audio embeddings from synthetic to natural manifold.
            """
        )
        
        # Model selection
        model_path_input = st.text_input(
            "Model checkpoint path", value=str(artifacts_dir / "natvox_model.pth")
        )
        
        model_path = Path(model_path_input)
        if not model_path.exists():
            st.error(f"❌ Checkpoint not found: {model_path}")
            st.info("Train a model first or provide a valid checkpoint path")
            st.stop()
        
        # Audio device selection
        st.write("**Audio Device Configuration**")
        col1, col2 = st.columns(2)
        
        input_devices = get_available_input_devices()
        if not input_devices:
            st.error("❌ No input devices found!")
            st.stop()
        
        device_names = list(input_devices.keys())
        default_device = device_names[0] if device_names else None
        selected_input = col1.selectbox("Input device", device_names, index=0)
        input_device_id = input_devices[selected_input]
        
        # Parameters
        col1, col2 = st.columns(2)
        buffer_duration = col1.slider(
            "Buffer duration (seconds)", 
            min_value=0.5, 
            max_value=5.0, 
            value=2.0, 
            step=0.5
        )
        session_duration = col2.slider(
            "Session duration (seconds, 0=infinite)", 
            min_value=0, 
            max_value=300, 
            value=0, 
            step=10
        )
        
        # Live inference section
        if st.button("🔴 Start Live Inference", type="primary"):
            # Load model
            @st.cache_resource
            def load_live_model(path):
                m = NATVOXAdapter()
                m.load_state_dict(torch.load(path, map_location="cpu"))
                m.eval()
                return m
            
            model = load_live_model(str(model_path))
            
            # Create placeholder for live metrics
            metrics_placeholder = st.empty()
            chart_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Initialize processor
            processor = LiveAudioProcessor(
                model, 
                sample_rate=16000, 
                buffer_duration=buffer_duration
            )
            
            try:
                processor.start_capture(device=input_device_id)
                status_placeholder.success("🟢 Live inference active - listening...")
                
                similarities = []
                timestamps = []
                start_time = time.time()
                chunk_count = 0
                
                # Live update loop
                while True:
                    # Check if duration exceeded
                    if session_duration > 0 and (time.time() - start_time) > session_duration:
                        status_placeholder.info("⏹️ Session duration reached, stopped.")
                        break
                    
                    # Get latest results
                    result = processor.get_latest_results(timeout=0.5)
                    
                    if result:
                        chunk_count += 1
                        similarities.append(result['similarity'])
                        timestamps.append(len(timestamps))
                        
                        # Update metrics
                        metrics = processor.get_metrics_summary()
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with metrics_placeholder.container():
                            col1.metric("Current Similarity", f"{result['similarity']:.4f}")
                            col2.metric("Average Similarity", f"{metrics['avg_similarity']:.4f}")
                            col3.metric("Chunks Processed", metrics['chunks_processed'])
                            col4.metric("Input RMS", f"{result['rms']:.4f}")
                    
                    # Update chart
                    if similarities:
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(timestamps, similarities, marker='o', linestyle='-', label='Similarity')
                        ax.axhline(y=np.mean(similarities), color='r', linestyle='--', label='Mean')
                        ax.set_xlabel("Chunk")
                        ax.set_ylabel("Cosine Similarity")
                        ax.set_title("Real-time Embedding Adaptation Similarity")
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        with chart_placeholder.container():
                            st.pyplot(fig)
                    
                    time.sleep(0.2)
            
            except KeyboardInterrupt:
                status_placeholder.warning("⏸️ Stopped by user")
            except Exception as e:
                status_placeholder.error(f"❌ Error: {e}")
            finally:
                processor.stop_capture()
                
                # Final summary
                if similarities:
                    final_metrics = processor.get_metrics_summary()
                    st.success("📊 Session Complete")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Chunks", final_metrics['chunks_processed'])
                    col2.metric("Avg Similarity", f"{final_metrics['avg_similarity']:.4f}")
                    col3.metric("Min/Max", f"{final_metrics['min_similarity']:.4f} / {final_metrics['max_similarity']:.4f}")
                    col4.metric("Std Dev", f"{final_metrics['std_similarity']:.4f}")
                    
                    # Distribution visualization
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Histogram
                    ax1.hist(similarities, bins=20, color='#2ECC71', alpha=0.7, edgecolor='black')
                    ax1.axvline(np.mean(similarities), color='red', linestyle='--', linewidth=2, label='Mean')
                    ax1.set_xlabel("Similarity")
                    ax1.set_ylabel("Frequency")
                    ax1.set_title("Similarity Distribution")
                    ax1.legend()
                    
                    # Time series with statistics
                    ax2.plot(timestamps, similarities, marker='o', linestyle='-', label='Similarity')
                    ax2.axhline(y=np.mean(similarities), color='r', linestyle='--', label='Mean')
                    ax2.fill_between(
                        timestamps, 
                        np.mean(similarities) - np.std(similarities),
                        np.mean(similarities) + np.std(similarities),
                        alpha=0.2,
                        color='red',
                        label='±1 Std Dev'
                    )
                    ax2.set_xlabel("Chunk")
                    ax2.set_ylabel("Similarity")
                    ax2.set_title("Similarity Over Time")
                    ax2.legend()
                    
                    st.pyplot(fig)