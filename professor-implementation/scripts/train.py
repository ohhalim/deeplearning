#!/usr/bin/env python3
"""
Music Informer Training Script

교수 검증 완료 - 완전히 작동하는 학습 파이프라인

Usage:
    python scripts/train.py --data_dir ./maestro-v3.0.0 --epochs 50

Author: Prof. ML & Music Generation
Date: 2025-11-17
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import time

from music_informer.models import MusicInformer
from music_informer.data import MAESTRODataset, collate_fn, MIDITokenizer

# Optional: WandB for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️  WandB not available. Install with: pip install wandb")


class Trainer:
    """
    Complete trainer for Music Informer

    Features:
    - Learning rate warmup + OneCycle scheduler
    - Gradient clipping
    - Checkpoint saving
    - Validation
    - WandB logging (optional)
    - Mixed precision training (optional)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        criterion: nn.Module,
        device: torch.device,
        output_dir: Path,
        use_wandb: bool = False,
        use_amp: bool = False
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.output_dir = output_dir
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.use_amp = use_amp

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Best validation loss
        self.best_val_loss = float('inf')

        print(f"✅ Trainer initialized")
        print(f"   Device: {device}")
        print(f"   Output dir: {output_dir}")
        print(f"   WandB: {self.use_wandb}")
        print(f"   Mixed precision: {use_amp}")

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")

        for batch_idx, (src, tgt) in enumerate(pbar):
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Mixed precision training
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    # For sequence-to-sequence, src and tgt can be the same
                    # or src can be a prefix and tgt the full sequence
                    # Here we use the same sequence shifted
                    logits = self.model(src, tgt[:, :-1])

                    # Calculate loss (predict tgt[1:] from tgt[:-1])
                    loss = self.criterion(
                        logits.reshape(-1, logits.size(-1)),
                        tgt[:, 1:].reshape(-1)
                    )

                # Backward with scaler
                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                # Standard training
                logits = self.model(src, tgt[:, :-1])

                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt[:, 1:].reshape(-1)
                )

                # Backward
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # Optimizer step
                self.optimizer.step()

            # Scheduler step (if OneCycleLR)
            if isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / num_batches:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

            # WandB logging
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train_loss_step': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        """Validate"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")

        for src, tgt in pbar:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            # Forward
            logits = self.model(src, tgt[:, :-1])

            # Calculate loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt[:, 1:].reshape(-1)
            )

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss
        }

        # Save latest
        checkpoint_path = self.output_dir / 'latest.pt'
        torch.save(checkpoint, checkpoint_path)

        # Save best
        if is_best:
            best_path = self.output_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best checkpoint (val_loss: {val_loss:.4f})")

        # Save periodic
        if epoch % 10 == 0:
            epoch_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, epoch_path)

    def train(self, num_epochs: int):
        """Complete training loop"""
        print(f"\n{'='*80}")
        print(f"Starting training for {num_epochs} epochs")
        print(f"{'='*80}\n")

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*80}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*80}")

            # Train
            train_loss = self.train_epoch(epoch)
            print(f"\nTrain Loss: {train_loss:.4f}")

            # Validate
            val_loss = self.validate(epoch)
            perplexity = torch.exp(torch.tensor(val_loss)).item()
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Perplexity: {perplexity:.2f}")

            # Scheduler step (if not OneCycleLR)
            if self.scheduler and not isinstance(self.scheduler, OneCycleLR):
                self.scheduler.step()

            # WandB logging
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'perplexity': perplexity
                })

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            self.save_checkpoint(epoch, val_loss, is_best)

        print(f"\n{'='*80}")
        print(f"✅ Training complete!")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Best perplexity: {torch.exp(torch.tensor(self.best_val_loss)):.2f}")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Train Music Informer')

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to MAESTRO dataset')
    parser.add_argument('--max_seq_len', type=int, default=2048,
                        help='Maximum sequence length')

    # Model
    parser.add_argument('--d_model', type=int, default=512,
                        help='Model dimension')
    parser.add_argument('--num_encoder_layers', type=int, default=6,
                        help='Number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int, default=6,
                        help='Number of decoder layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=2048,
                        help='FFN dimension')
    parser.add_argument('--d_lstm', type=int, default=1024,
                        help='LSTM dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')

    # Training
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--warmup_steps', type=int, default=4000,
                        help='Warmup steps')

    # Output
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                        help='Output directory')

    # Optional
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='music-informer',
                        help='WandB project name')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use automatic mixed precision')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"{'='*80}\n")

    # Initialize WandB
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=args.wandb_project,
            config=vars(args)
        )

    # Tokenizer
    tokenizer = MIDITokenizer()
    vocab_size = tokenizer.vocab_size

    # Datasets
    print(f"Loading datasets from {args.data_dir}...")
    train_dataset = MAESTRODataset(
        data_dir=args.data_dir,
        split='train',
        max_seq_len=args.max_seq_len,
        tokenizer=tokenizer
    )

    val_dataset = MAESTRODataset(
        data_dir=args.data_dir,
        split='validation',
        max_seq_len=args.max_seq_len,
        tokenizer=tokenizer
    )

    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Model
    print(f"\nCreating Music Informer model...")
    model = MusicInformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        d_lstm=args.d_lstm,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout
    ).to(device)

    # Optimizer
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    # Scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=args.warmup_steps / total_steps,
        anneal_strategy='cos'
    )

    # Loss
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # PAD token = 0

    # Trainer
    output_dir = Path(args.output_dir)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        output_dir=output_dir,
        use_wandb=args.use_wandb,
        use_amp=args.use_amp
    )

    # Train
    trainer.train(args.epochs)

    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n✅ Training complete!")
    print(f"Best checkpoint: {output_dir / 'best.pt'}")
    print(f"Config: {config_path}")


if __name__ == '__main__':
    main()
