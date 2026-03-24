# NATVOX-AI: Generative Voice Personalization

## Overview
NATVOX-AI is a lightweight adapter model designed to bridge the gap between synthetic and natural speech embeddings.

## Features
- Residual adapter network (256 → 512 → 256)
- Cosine + L2 loss
- Training visualization
- Model saving & inference
- Domain gap simulation
- Web interface with Streamlit

## Results
- Similarity Before: ~0.001
- Similarity After: ~0.92

## System Requirements
- Python 3.8+
- PyTorch 1.9+
- CUDA-compatible GPU (recommended for training)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd natvox-ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python main.py
```
This will train the model and save it as `natvox_model.pth`.

### Inference
```bash
python inference.py
```
Make sure to place your audio file as `sample.wav` in the root directory.

### Web Interface
```bash
streamlit run app.py
```
Open the provided URL in your browser to use the interactive web interface.

## Project Structure
- `main.py`: Main training script
- `model.py`: NATVOXAdapter model architecture
- `train.py`: Training utilities
- `inference.py`: Inference script
- `app.py`: Streamlit web interface
- `audio_utils.py`: Audio processing utilities
- `dataset.py`: Data loading and generation
- `utils.py`: Loss functions and utilities