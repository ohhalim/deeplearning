"""
Month 2: Music Transformer - Training Loop

목표:
1. 모델 학습
2. Loss monitoring
3. Checkpoint 저장
4. 생성 품질 평가
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
import os
from pathlib import Path

from 01_basic_transformer import MusicTransformer
from 02_midi_preprocessing import MIDITokenizer, MAESTRODataset, collate_fn


class Trainer:
    """Music Transformer Trainer"""

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        tokenizer,
        device='cuda',
        lr=1e-4,
        warmup_steps=4000,
        log_wandb=False
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device

        # Optimizer with warmup
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.98),
            eps=1e-9
        )

        self.warmup_steps = warmup_steps
        self.step = 0

        # Loss function (ignore padding)
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=tokenizer.PAD
        )

        # Logging
        self.log_wandb = log_wandb
        if log_wandb:
            wandb.init(
                project="music-transformer",
                config={
                    "d_model": model.d_model,
                    "num_layers": len(model.layers),
                    "vocab_size": model.vocab_size,
                    "lr": lr,
                    "warmup_steps": warmup_steps
                }
            )

    def lr_schedule(self, step):
        """
        Learning rate warmup schedule from "Attention Is All You Need"

        lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
        """
        d_model = self.model.d_model
        step = max(step, 1)  # Avoid division by zero

        lr = (d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5)
        )

        return lr

    def update_lr(self):
        """Update learning rate"""
        lr = self.lr_schedule(self.step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for src, tgt in pbar:
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            # Forward
            logits = self.model(src)  # (batch, seq_len, vocab_size)

            # Reshape for loss
            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1)
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # Optimizer step
            self.optimizer.step()

            # Update learning rate
            self.step += 1
            lr = self.update_lr()

            # Logging
            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{lr:.2e}'
            })

            if self.log_wandb and self.step % 100 == 0:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': lr,
                    'step': self.step
                })

        avg_loss = total_loss / num_batches
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """Validation"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)

        for src, tgt in tqdm(self.val_loader, desc="Validation"):
            src = src.to(self.device)
            tgt = tgt.to(self.device)

            logits = self.model(src)

            loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1)
            )

            total_loss += loss.item()

        avg_loss = total_loss / num_batches

        # Calculate perplexity
        perplexity = torch.exp(torch.tensor(avg_loss))

        return avg_loss, perplexity.item()

    @torch.no_grad()
    def generate_sample(self, num_samples=3, max_len=256):
        """Generate music samples"""
        self.model.eval()

        samples = []
        for i in range(num_samples):
            # Start with SOS token
            start_tokens = torch.tensor(
                [[self.tokenizer.SOS]],
                dtype=torch.long,
                device=self.device
            )

            # Generate
            generated = self.model.generate(
                start_tokens,
                max_len=max_len,
                temperature=1.0,
                top_k=40
            )

            samples.append(generated.cpu().numpy()[0])

        return samples

    def save_checkpoint(self, epoch, save_dir='checkpoints'):
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }

        path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def train(self, num_epochs, save_every=5):
        """Full training loop"""
        print("=" * 80)
        print("Training Started")
        print("=" * 80)

        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")

            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")

            # Validate
            val_loss, perplexity = self.validate()
            print(f"Val Loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")

            if self.log_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss_epoch': train_loss,
                    'val_loss': val_loss,
                    'perplexity': perplexity
                })

            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(epoch)

                # Generate samples
                print("\nGenerating samples...")
                samples = self.generate_sample(num_samples=2, max_len=128)

                for i, sample in enumerate(samples):
                    output_path = f'generated_epoch{epoch}_sample{i}.mid'
                    self.tokenizer.tokens_to_midi(
                        sample.tolist(),
                        output_path
                    )
                    print(f"Sample {i} saved to {output_path}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best')
                print(f"New best model! Val loss: {val_loss:.4f}")

        print("\n" + "=" * 80)
        print("Training Completed!")
        print("=" * 80)


# ============================================================================
# Main Training Script
# ============================================================================

if __name__ == "__main__":
    # Hyperparameters
    VOCAB_SIZE = 391  # From tokenizer
    D_MODEL = 512
    NUM_HEADS = 8
    NUM_LAYERS = 6
    D_FF = 2048
    MAX_SEQ_LEN = 2048
    DROPOUT = 0.1

    BATCH_SIZE = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    WARMUP_STEPS = 4000

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}")

    # Tokenizer
    tokenizer = MIDITokenizer(
        num_velocity_bins=32,
        time_resolution=10,
        max_time_shift=1000
    )

    # Datasets
    # NOTE: MAESTRO 데이터셋 경로를 실제 경로로 변경하세요
    DATA_DIR = '/path/to/maestro-v3.0.0'  # CHANGE THIS!

    if not os.path.exists(DATA_DIR):
        print(f"\n{'='*80}")
        print("WARNING: MAESTRO dataset not found!")
        print(f"Please download from: https://magenta.tensorflow.org/datasets/maestro")
        print(f"And update DATA_DIR in this script")
        print(f"{'='*80}\n")
        exit(1)

    train_dataset = MAESTRODataset(
        data_dir=DATA_DIR,
        tokenizer=tokenizer,
        max_seq_len=MAX_SEQ_LEN,
        split='train'
    )

    val_dataset = MAESTRODataset(
        data_dir=DATA_DIR,
        tokenizer=tokenizer,
        max_seq_len=MAX_SEQ_LEN,
        split='validation'
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # Model
    model = MusicTransformer(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT
    )

    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=DEVICE,
        lr=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        log_wandb=False  # Set True to use Weights & Biases
    )

    # Train!
    trainer.train(num_epochs=NUM_EPOCHS, save_every=5)
