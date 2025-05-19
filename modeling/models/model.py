from typing import List, Dict

import torch
from torch import nn

from modeling.models.modules.EventEncoder import EventEncoder
from modeling.models.modules.EventPredictor import EventPredictor


def causal_attention_mask(seq_len: int) -> torch.Tensor:
    mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool()
    return mask


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
        super(Model, self).__init__()
        self.event_embedder = EventEncoder(d_categories, d_model)

        # build one TransformerEncoder per stream name
        encoder_modules = {}
        for encoder in encoders:
            encoder_modules[encoder] = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    batch_first=True,
                    activation=nn.Mish()
                ),
                num_layers=encoder_layers
            )
        self.encoders_dict = nn.ModuleDict(encoder_modules)

        self.event_predictor = EventPredictor(
            d_categories=d_categories,
            d_facts=d_output,
            d_input=d_input,
            d_model=d_model,
            num_heads=n_heads,
            decoder_layers=decoder_layers
        )

    def forward(
        self,
        src_events: Dict[str, List[torch.Tensor]],
        tgt_events: torch.Tensor,
        encodings_to_carry: int
    ) -> List[torch.Tensor]:
        """
        src_events: {
            stream_name: [ Tensor(batch, seq, cats), ... ],
            ...
        }
        tgt_events: Tensor(batch, tgt_seq, cats)
        """
        # 1) Encode each stream, collect the *last* encodings_to_carry tokens
        encoder_outputs = []
        for stream_name, stream_list in src_events.items():
            # stream_list is e.g. [ tensor(batch, seq, cats), tensor(batch, seq, cats) ]
            for seq in stream_list:
                # embed + positional
                emb = self.event_embedder(seq)             # [b, seq, d_model]
                encoded = self.encoders_dict[stream_name](emb)  # [b, seq, d_model]
                # take last encodings_to_carry tokens
                encoder_outputs.append(encoded[:, -encodings_to_carry:, :])

        # concat along time dimension → memory
        memory = torch.cat(encoder_outputs, dim=1)  # [b, total_streams * encodings_to_carry, d_model]

        # 2) Prepare target embeddings + causal mask
        embedded_target = self.event_embedder(tgt_events)   # [b, tgt_seq, d_model]
        seq_len = embedded_target.size(1)
        causal_mask = causal_attention_mask(seq_len).to(embedded_target.device)

        # 3) Predict next events
        return self.event_predictor(
            tgt=embedded_target,
            memory=memory,
            tgt_mask=causal_mask
        )


class ModelTrainer(nn.Module):
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
        super(ModelTrainer, self).__init__()
        self.model = Model(
            d_input=d_input,
            d_model=d_model,
            n_heads=n_heads,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            d_categories=d_categories,
            encoders=encoders,
            d_output=d_output
        )

    def forward(
        self,
        batch_src_events: List[Dict[str, List[torch.Tensor]]],
        batch_tgt_events: List[torch.Tensor],
        batch_masks: List[List[torch.Tensor]],
        cross_encoder_token_length: int = 1,
        run_backward: bool = False,
    ):
        loss_sum = 0.0
        per_category_loss = {}

        for i in range(len(batch_tgt_events)):
            # prepare one minibatch
            src_events = batch_src_events[i]
            tgt_events = batch_tgt_events[i].to(dtype=torch.int64).unsqueeze(0)  # [1, tgt_seq, cats]
            masks = batch_masks[i]  # list of [ Tensor(cat_size), ... ]

            # move src to correct dtype/device
            for stream_name in src_events:
                for j, seq in enumerate(src_events[stream_name]):
                    src_events[stream_name][j] = seq.to(dtype=torch.int64)

            # run forward
            preds = self.model(
                src_events,
                tgt_events[:, :-1, :],
                cross_encoder_token_length
            )

            # compute category losses
            cat_losses = []
            for cat_idx, logits in enumerate(preds):
                # logits: [1, seq, vocab_size]
                # mask out invalid positions
                mask_tensor = masks[cat_idx].to(logits.dtype).to(logits.device)  # [vocab_size]
                logits = logits + mask_tensor

                target = tgt_events[:, 1:, cat_idx].reshape(-1)
                logits_flat = logits.reshape(-1, logits.size(-1))
                loss = nn.functional.cross_entropy(logits_flat, target)
                cat_losses.append(loss)

                per_category_loss.setdefault(cat_idx, 0.0)
                per_category_loss[cat_idx] += loss.item()

            batch_loss = torch.stack(cat_losses).mean()
            if run_backward:
                batch_loss.backward()
            loss_sum += batch_loss.item()

        avg_loss = loss_sum / len(batch_tgt_events)
        avg_per_cat = [per_category_loss[i] / len(batch_tgt_events) for i in sorted(per_category_loss)]
        return avg_loss, avg_per_cat
