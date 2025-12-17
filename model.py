import torch
import torch.nn as nn

class IntentEncoder(nn.Module):
    def __init__(self, action_vocab=4, embed_dim=16, hidden_dim=32):
        super().__init__()

        self.action_embed = nn.Embedding(action_vocab, embed_dim)
        self.value_proj = nn.Linear(1, embed_dim)
        self.time_proj = nn.Linear(1, embed_dim)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        action_ids = x[:, :, 0].long()
        values = x[:, :, 1:2]
        times = x[:, :, 2:3]

        a = self.action_embed(action_ids)
        v = self.value_proj(values)
        t = self.time_proj(times)

        combined = a + v + t
        _, (h, _) = self.lstm(combined)

        return self.output(h.squeeze(0))
