import torch

def get_data():
    torch.manual_seed(42)

    synthetic = torch.randn(30, 256)

    # Create stronger domain gap
    transform = torch.randn(256, 256) * 0.5
    natural = synthetic @ transform  # matrix transform

    return synthetic, natural