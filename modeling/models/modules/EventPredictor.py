from typing import List
import torch
from torch import nn

class EventPredictor(nn.Module):
    def __init__(
        self,
        d_categories: List[int],
        d_input: int,
        d_facts: int,
        d_model: int,
        num_heads: int,
        decoder_layers: int
    ):
        super().__init__()
        self.predictor = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                batch_first=True,
                norm_first=True,
                activation=nn.Mish()
            )
            for _ in range(decoder_layers)
        ])

        self.fact_proj = nn.Linear(d_model, d_facts)
        self.cat_heads = nn.ModuleList([
            nn.Linear(d_model, c) for c in d_categories
        ])
        self.out_proj = nn.Linear(d_model, d_input)

    def forward(self, tgt, memory, tgt_mask=None):
        x = tgt
        for layer in self.predictor:
            x = layer(tgt=x, memory=memory, tgt_mask=tgt_mask, tgt_is_causal=True)
        return [head(x) for head in self.cat_heads]
