from typing import List, Dict
import torch
from torch import nn

from modeling.models.modules.EventEncoder import EventEncoder
from modeling.models.modules.EventPredictor import EventPredictor

def causal_mask(sz: int) -> torch.Tensor:
    return torch.triu(torch.ones(sz, sz), diagonal=1).bool()

class Model(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_model: int,
        n_heads: int,
        encoder_layers: int,
        decoder_layers: int,
        d_categories: List[int],
        encoders: List[str],
        d_output: int
    ):
        super().__init__()
        self.embedder = EventEncoder(d_categories, d_model)

        self.encoders = nn.ModuleDict({
            name: nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    batch_first=True,
                    activation=nn.Mish()
                ),
                num_layers=encoder_layers
            )
            for name in encoders
        })

        self.predictor = EventPredictor(
            d_categories=d_categories,
            d_input=d_input,
            d_facts=d_output,
            d_model=d_model,
            num_heads=n_heads,
            decoder_layers=decoder_layers
        )

    def forward(self, src_events: Dict[str, List[torch.Tensor]], tgt_events: torch.Tensor, carry: int):
        enc_outs = []
        for name, seq_list in src_events.items():
            for seq in seq_list:
                emb = self.embedder(seq)              # [1, src_seq, d_model]
                out = self.encoders[name](emb)        # [1, src_seq, d_model]
                enc_outs.append(out[:, -carry:, :])    # take last `carry` tokens
        memory = torch.cat(enc_outs, dim=1)         # [1, total_carry, d_model]

        tgt_emb = self.embedder(tgt_events)        # [1, tgt_seq, d_model]
        m = causal_mask(tgt_emb.size(1)).to(tgt_emb.device)

        return self.predictor(tgt_emb, memory, tgt_mask=m)

class ModelTrainer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = Model(**kwargs)

    def forward(self, batch_src, batch_tgt, batch_mask, carry=1, run_backward=False):
        total_loss = 0.0
        cat_losses = {}

        for src, tgt, mask in zip(batch_src, batch_tgt, batch_mask):

            preds = self.model(src, tgt[:, :-1, :], carry)
            losses = []
            for i, logits in enumerate(preds):
                # apply per‐category mask
                logits = logits + mask[i].to(logits.device)
                tgt_i = tgt[:, 1:, i].reshape(-1)
                logits_flat = logits.reshape(-1, logits.size(-1))
                loss = nn.functional.cross_entropy(logits_flat, tgt_i)
                losses.append(loss)
                cat_losses[i] = cat_losses.get(i, 0.0) + loss.item()

            batch_loss = torch.stack(losses).mean()
            if run_backward:
                batch_loss.backward()
            total_loss += batch_loss.item()

        avg_loss = total_loss / len(batch_tgt)
        avg_per_cat = [v / len(batch_tgt) for i, v in sorted(cat_losses.items())]
        return avg_loss, avg_per_cat
