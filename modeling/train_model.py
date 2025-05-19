import argparse
import json
import os
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from data.events_loader import EventsLoader
from modeling.models.model import ModelTrainer


def load_config(path: str = 'training_config.json') -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Train the Event Prediction Model")
    p.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable CUDA even if available'
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(os.path.join(os.path.dirname(__file__), 'training_config.json'))

    # Device selection
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f"[INFO] Using device: {device}")

    full_dataset = EventsLoader(
        batch_size=cfg['batch_size'],
        src_seq_length=cfg['src_seq_len'],
        tgt_seq_length=cfg['tgt_seq_len'],
        id_category_size=cfg['id_category_size']
    )

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=2, pin_memory=True,
        collate_fn=lambda x: x[0]
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True,
        collate_fn=lambda x: x[0]
    )

    model_trainer = ModelTrainer(
        d_input=full_dataset.input_size,
        d_model=cfg['model']['d_model'],
        n_heads=cfg['model']['num_heads'],
        encoder_layers=cfg['model']['encoder_layers'],
        decoder_layers=cfg['model']['decoder_layers'],
        d_categories=list(full_dataset.category_fields.values()),
        encoders=full_dataset.encoder_streams,
        d_output=full_dataset.output_size
    ).to(device)

    optimizer = torch.optim.Adam(model_trainer.parameters(), lr=3e-4)

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_writer = SummaryWriter(log_dir=f"./runs/{run_id}/train")
    val_writer = SummaryWriter(log_dir=f"./runs/{run_id}/val")

    for epoch in range(cfg['epochs']):
        print(f"[EPOCH {epoch}]")

        model_trainer.eval()
        with torch.no_grad():
            for step, (tgt_list, src_list, mask_list) in enumerate(val_loader):
                # move to device
                tgt_list = [t.to(device) for t in tgt_list]
                src_list = [
                    {k: [seq.to(device) for seq in seqs] for k, seqs in src_dict.items()}
                    for src_dict in src_list
                ]
                mask_list = [
                    [m.to(device) for m in masks]
                    for masks in mask_list
                ]

                loss, per_cat = model_trainer(
                    src_list,    # batch_src_events
                    tgt_list,    # batch_tgt_events
                    mask_list,   # batch_masks
                    1,           # cross_encoder_token_length
                    False        # run_backward
                )
                val_writer.add_scalar(
                    "loss/total", loss, epoch * len(val_loader) + step
                )

        model_trainer.train()
        optimizer.zero_grad()
        for step, (tgt_list, src_list, mask_list) in enumerate(train_loader):
            tgt_list = [t.to(device) for t in tgt_list]
            src_list = [
                {k: [seq.to(device) for seq in seqs] for k, seqs in src_dict.items()}
                for src_dict in src_list
            ]
            mask_list = [
                [m.to(device) for m in masks]
                for masks in mask_list
            ]

            loss, per_cat = model_trainer(
                src_list,
                tgt_list,
                mask_list,
                1,    
                True  
            )
            optimizer.step()
            optimizer.zero_grad()

            train_writer.add_scalar(
                "loss/total", loss, epoch * len(train_loader) + step
            )

        ckpt_dir = f"./runs/{run_id}/ckpt"
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model_trainer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            os.path.join(ckpt_dir, f"epoch_{epoch}.pt")
        )

    torch.save(model_trainer.state_dict(), f"./runs/{run_id}/final.pt")
    print(f"[DONE] Saved final model to ./runs/{run_id}/final.pt")


if __name__ == '__main__':
    main()