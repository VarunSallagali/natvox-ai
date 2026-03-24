import torch
import torch.optim as optim
from utils import loss_fn

def train(model, synthetic, natural):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_history = []

    for epoch in range(50):
        model.train()
        
        optimizer.zero_grad()
        output = model(synthetic)
        
        loss = loss_fn(output, natural)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Save model
    try:
        torch.save(model.state_dict(), "natvox_model.pth")
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving model: {e}")
        raise

    return model, loss_history