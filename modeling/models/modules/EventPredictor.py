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
        super(EventPredictor, self).__init__()

        self.predictor = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=num_heads,
                batch_first=True,
                norm_first=True,
                activation=nn.Mish()
            ) for _ in range(decoder_layers)
        ])

        self.fact_projection = nn.Linear(d_model, d_facts)
        modules = []
        for d_category in d_categories:
            modules.append(nn.Linear(d_model, d_category))
        self.categorical_projections = nn.ModuleList(modules)
        self.output_projection = nn.Linear(d_model, d_input)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None
    ) -> List[torch.Tensor]:
        """
        Args:
            tgt:    [batch_size, seq_len, d_model]
            memory: [batch_size, mem_len, d_model]
            tgt_mask: [seq_len, seq_len]
        """
        y_e = tgt
        for layer in self.predictor:
            y_e = layer(
                tgt=y_e,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_is_causal=True
            )

        category_predictions = []
        for output in self.categorical_projections:
            category_predictions.append(output(y_e))

        return category_predictions
