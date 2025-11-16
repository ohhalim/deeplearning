"""
Music Informer - Training Script

Based on paper specifications:
- Optimizer: Adam
- 6 encoder layers
- 8 attention heads
- 512 hidden dimension
- Dropout: 0.1
- Dataset: MAESTRO
"""

import sys
sys.path.append('../src')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
from tqdm import tqdm
import wandb

from model import MusicInformer


def train_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """Train one epoch"""
    model.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (src, tgt) in enumerate(pbar):
        src = src.to(device)
        tgt = tgt.to(device)

        # Forward
        logits = model(src)

        # Loss
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        if batch_idx % 100 == 0:
            wandb.log({
                'train_loss': loss.item(),
                'epoch': epoch,
                'step': epoch * len(train_loader) + batch_idx
            })

    return total_loss / len(train_loader)


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate"""
    model.eval()
    total_loss = 0

    for src, tgt in tqdm(val_loader, desc="Validation"):
        src = src.to(device)
        tgt = tgt.to(device)

        logits = model(src)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt.reshape(-1)
        )

        total_loss += loss.item()

    return total_loss / len(val_loader)


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Model (paper specs)
    model = MusicInformer(
        vocab_size=388,
        d_model=512,      # Paper
        num_layers=6,     # Paper
        num_heads=8,      # Paper
        d_ff=1024,        # Paper
        d_lstm=1024,      # Paper
        max_seq_len=2048,
        dropout=0.1       # Paper
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dataset (simplified - use your actual MAESTRO loader)
    # from data.dataset import MAESTRODataset
    # train_dataset = MAESTRODataset(args.data_dir, split='train')
    # val_dataset = MAESTRODataset(args.data_dir, split='validation')

    # For now, placeholder
    print("Note: Using placeholder dataset")
    print("Replace with actual MAESTRO loader!")

    # Optimizer (Adam as in paper)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD token

    # WandB
    if args.use_wandb:
        wandb.init(
            project="music-informer",
            config=vars(args)
        )

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")

        # Train
        # train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        # print(f"Train Loss: {train_loss:.4f}")

        # Validate
        # val_loss = validate(model, val_loader, criterion, device)
        # print(f"Val Loss: {val_loss:.4f}")
        # perplexity = torch.exp(torch.tensor(val_loss))
        # print(f"Perplexity: {perplexity:.2f}")

        # Save best
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'val_loss': val_loss,
        #     }, args.output_dir / 'best.pt')
        #     print(f"Saved best model! Val loss: {val_loss:.4f}")

        print("\n[Placeholder] Training loop not executed")
        print("Add your MAESTRO dataset and uncomment training code!")

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./maestro-v3.0.0')
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_wandb', action='store_true')

    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    main(args)
