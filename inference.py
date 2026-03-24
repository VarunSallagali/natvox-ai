import torch
from model import NATVOXAdapter
from audio_utils import get_embedding

# Load model
model = NATVOXAdapter()
model.load_state_dict(torch.load("natvox_model.pth"))
model.eval()

# Load real audio
audio_path = "sample.wav"   # <-- put any .wav file
embedding = get_embedding(audio_path)

embedding = torch.tensor(embedding).float().unsqueeze(0)

with torch.no_grad():
    adapted = model(embedding)

print("Original embedding shape:", embedding.shape)
print("Adapted embedding shape:", adapted.shape)