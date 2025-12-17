import torch
import torch.nn as nn
from model import IntentEncoder
from dataset import generate_batch

model = IntentEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for step in range(100):
    batch = generate_batch()
    past = batch[:, :-1, :]
    future = batch[:, -1:, :]

    intent = model(past)
    target = model(future)

    loss = loss_fn(intent, target.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 10 == 0:
        print(f"Step {step} | Loss {loss.item():.4f}")
