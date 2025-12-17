import torch
from model import IntentEncoder
from dataset import generate_sequence

model = IntentEncoder()
model.eval()

history = generate_sequence(9).unsqueeze(0)
new_action = generate_sequence(1).unsqueeze(0)

intent_vec = model(history)
new_vec = model(new_action)

similarity = torch.cosine_similarity(intent_vec, new_vec)

score = int(((similarity.item() + 1) / 2) * 100)

print("Intent Consistency Score:", score)
