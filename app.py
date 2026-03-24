import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import NATVOXAdapter
from audio_utils import get_embedding
from pydub import AudioSegment
import os
import torchaudio
import numpy
import librosa
import scipy

# -----------------------------
# CACHED MODEL LOADING
# -----------------------------
@st.cache_resource
def load_model():
    try:
        model = NATVOXAdapter()
        model.load_state_dict(torch.load("natvox_model.pth"))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# -----------------------------
# UI FIX (CENTER ALIGN)
# -----------------------------
st.set_page_config(page_title="NATVOX-AI", layout="wide")

st.markdown("""
<style>
.main {
    max-width: 1100px;
    margin: auto;
}
.block-container {
    padding-top: 2rem;
}
section[data-testid="stSidebar"] {
    width: 250px !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# AUDIO CONVERSION FUNCTION
# -----------------------------
def convert_to_wav(file_path):
    if file_path.endswith(".m4a"):
        audio = AudioSegment.from_file(file_path, format="m4a")
        new_path = "converted.wav"
        audio.export(new_path, format="wav")
        return new_path
    return file_path

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio("Go to", [
    "Home",
    "Upload & Results",
    "Visualizations",
    "Model Info",
    "Future Work"
])

# -----------------------------
# HOME
# -----------------------------
if page == "Home":
    st.title("🎙️ NATVOX-AI")
    st.subheader("Voice Personalization System")

    st.write("""
    Convert synthetic speech embeddings into natural speech space 
    using a lightweight residual adapter model.
    """)

# -----------------------------
# UPLOAD PAGE
# -----------------------------
elif page == "Upload & Results":

    st.header("📂 Upload Audio & Get Results")

    uploaded = st.file_uploader("Upload Audio", type=["wav", "m4a"])

    if uploaded:

        file_path = "temp.wav"

        # Handle M4A
        if uploaded.name.endswith(".m4a"):
            file_path = "temp.m4a"

        # Save file correctly
        with open(file_path, "wb") as f:
            f.write(uploaded.getvalue())

        # Convert if needed
        file_path = convert_to_wav(file_path)

        st.success("Audio uploaded successfully!")

        col1, col2 = st.columns(2)

        # LEFT SIDE
        with col1:
            st.subheader("🔊 Input Audio")
            st.audio(uploaded)

        # RIGHT SIDE
        with col2:
            model = load_model()
            if model is None:
                st.error("Failed to load model. Please check if natvox_model.pth exists.")
                st.stop()

            with st.spinner("Processing..."):
                try:
                    emb = get_embedding(file_path)
                    emb = torch.tensor(emb).float().unsqueeze(0)

                    with torch.no_grad():
                        out = model(emb)
                except Exception as e:
                    st.error(f"Error processing audio: {e}")
                    st.stop()

            cos = nn.CosineSimilarity(dim=1)

            baseline = cos(emb, emb).item()
            adapted = cos(emb, out).item()
            improvement = adapted - baseline

            st.subheader("📊 Results")

            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline", f"{baseline:.3f}")
            c2.metric("After", f"{adapted:.3f}")
            c3.metric("Change", f"{improvement:.3f}")

            if adapted > 0.5:
                st.success("Good alignment ✅")
            else:
                st.warning("Needs real-data training ⚠️")

            # Store values
            st.session_state["baseline"] = baseline
            st.session_state["adapted"] = adapted

# -----------------------------
# VISUALIZATION
# -----------------------------
elif page == "Visualizations":

    st.header("📈 Visualizations")

    if "baseline" in st.session_state:

        baseline = st.session_state["baseline"]
        adapted = st.session_state["adapted"]

        st.subheader("Similarity Comparison")

        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(["Baseline", "After"], [baseline, adapted])
        ax.set_ylabel("Cosine Similarity")
        st.pyplot(fig)

        st.subheader("📉 Training Loss")
        st.image("loss_curve.png")

    else:
        st.info("Upload audio first")

# -----------------------------
# MODEL INFO
# -----------------------------
elif page == "Model Info":

    st.header("🧠 Model Details")

    st.write("""
    - Architecture: Residual Adapter Network  
    - Input/Output: 256 → 512 → 256  
    - Parameters: ~330K  
    - Loss: Cosine + L2  
    - Training Samples: 30  
    """)

    st.subheader("⚙️ Pipeline")

    st.write("""
    Audio → Encoder → Embedding → Adapter → Output Embedding
    """)

# -----------------------------
# FUTURE WORK
# -----------------------------
elif page == "Future Work":

    st.header("🚀 Future Improvements")

    st.write("""
    - Train on real datasets (VCTK, LibriTTS)
    - Improve speaker generalization
    - Add real-time processing
    - Integrate full voice conversion system
    """)

@st.cache_resource
def load_model():
    model = NATVOXAdapter()
    model.load_state_dict(torch.load("natvox_model.pth"))
    model.eval()
    return model