from typing import List
import torch
from torch import nn
from modeling.models.modules.Transformer.PositionalEncoding import PositionalEncoding

class EventEncoder(nn.Module):
    def __init__(self, d_categories: List[int], d_model: int):
        super().__init__()
        self.pe = PositionalEncoding(d_model, max_len=100000)
        max_card = max(d_categories)
        self.embed = nn.Embedding(max_card, d_model)
        self.d_model = d_model
        self.cat_dims = d_categories

        total = d_model * len(d_categories)
        self.fuse = nn.Sequential(
            nn.Linear(total, total * 4),
            nn.Mish(),
            nn.Linear(total * 4, d_model)
        )

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        embeds = [self.embed(events[..., i]) for i in range(events.size(-1))]
        x = torch.cat(embeds, dim=-1)        # [batch, seq, d_model*num_cats]
        x = self.fuse(x)                     # [batch, seq, d_model]
        return self.pe(x)
