import random
import torch

ACTIONS = ["swap", "stake", "vote", "mint"]

ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}

def generate_sequence(length=10):
    seq = []
    for _ in range(length):
        action = random.choice(ACTIONS)
        value = random.random() * 100
        time_delta = random.random() * 1000
        seq.append([
            ACTION_TO_ID[action],
            value,
            time_delta
        ])
    return torch.tensor(seq, dtype=torch.float)

def generate_batch(batch_size=32, seq_len=10):
    return torch.stack([
        generate_sequence(seq_len) for _ in range(batch_size)
    ])
