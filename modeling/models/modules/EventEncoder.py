from typing import List

import torch
from torch import nn

from modeling.modules.Transformer.PositionalEncoding import PositionalEncoding


class EventEncoder(nn.Module):

    def __init__(
        self,
        d_categories: List[int],
        d_model: int
    ):
        super(EventEncoder, self).__init__()

        self.pe = PositionalEncoding(d_model, 100000)
        max_size = max(d_categories)
        embedding_size = d_model
        self.category_embedding = nn.Embedding(max_size, embedding_size)
        self.categories = d_categories
        self.down_projection = nn.Sequential(
            nn.Linear(embedding_size * len(d_categories), embedding_size * len(d_categories) * 10),
            nn.Mish(),
            nn.Linear(embedding_size * len(d_categories) * 10, d_model)
        )

    def forward(
        self,
        events: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            events: [batch_size, seq_len, d_categories]
        """
        output_tensors = []
        for i, _ in enumerate(self.categories):
            output_tensors.append(self.category_embedding(events[:, :, i]))
        embedded_categories = torch.cat(output_tensors, dim=2)
        embedded_events = self.down_projection(embedded_categories)
        return self.pe(embedded_events)

