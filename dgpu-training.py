# train.py
import torch
from lightning.fabric import Fabric

def main(fabric: Fabric):
    fabric.print(f"Running on: {fabric.device}")
    # Dummy model and data
    model = torch.nn.Linear(10, 1).to(fabric.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model, optimizer = fabric.setup(model, optimizer)

    for step in range(5):
        x = torch.randn(64, 10, device=fabric.device)
        y = torch.randn(64, 1, device=fabric.device)
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), y)
        fabric.backward(loss)
        optimizer.step()
        fabric.print(f"Step {step} Loss: {loss.item():.4f}")

if __name__ == "__main__":
    fabric = Fabric(accelerator="cuda", devices=1, strategy="ddp")
    fabric.launch(main)
