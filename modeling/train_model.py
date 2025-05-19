# pytorch-model-challenge/modeling/train_model.py

import torch

from data.events_loader import EventsLoader

torch.set_float32_matmul_precision('high')
torch.set_default_dtype(torch.bfloat16)

import time
from datetime import datetime
import os

from modeling.models.model import ModelTrainer

from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

base_path = os.path.dirname(os.path.abspath(__file__))


def load_config(config_path=f'{base_path}/training_config.json'):
    import json
    return json.load(open(config_path))


def main():
    config = load_config()
    full_dataset = EventsLoader(
        batch_size=config['batch_size'],
        src_seq_length=config['src_seq_len'],
        tgt_seq_length=config['tgt_seq_len'],
        id_category_size=config['id_category_size']
    )

    train_size = int(0.9 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda x: x[0]
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda x: x[0]
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_trainer = ModelTrainer(
        d_input=full_dataset.input_size,
        d_categories=list(full_dataset.category_fields.values()),
        d_output=full_dataset.output_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['num_heads'],
        encoder_layers=config['model']['encoder_layers'],
        decoder_layers=config['model']['decoder_layers'],
        encoders=full_dataset.encoder_streams
    ).to(device)

    optimizer = torch.optim.Adam(model_trainer.parameters(), lr=3e-4)

    run_dir = f"/runs/{datetime.now():%Y%m%d-%H%M%S}"
    train_writer = SummaryWriter(log_dir=f"{run_dir}/train")
    test_writer  = SummaryWriter(log_dir=f"{run_dir}/test")

    for epoch in range(config['epochs']):
        print(f"Epoch {epoch}")
        # validation
        model_trainer.eval()
        with torch.no_grad():
            for step, (tgt, srcs, masks) in enumerate(test_loader):
                loss, per_cat = model_trainer([srcs], [tgt], [masks], run_backward=False)
                test_writer.add_scalar("loss/total", loss, epoch * len(test_loader) + step)

        # training
        model_trainer.train()
        optimizer.zero_grad()
        for step, (tgt, srcs, masks) in enumerate(train_loader):
            loss, per_cat = model_trainer([srcs], [tgt], [masks], run_backward=True)
            optimizer.step()
            optimizer.zero_grad()

            train_writer.add_scalar("loss/total", loss, epoch * len(train_loader) + step)

        # save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_trainer.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"{run_dir}/checkpoint_{epoch}.pt")

    torch.save(model_trainer.state_dict(), f"{run_dir}/final.pt")


if __name__ == '__main__':
    main()
