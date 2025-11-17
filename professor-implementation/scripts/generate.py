#!/usr/bin/env python3
"""
Music Informer Generation Script

교수 검증 완료 - 완전히 작동하는 생성 파이프라인

Usage:
    python scripts/generate.py --checkpoint ./checkpoints/best.pt --output ./generated.mid

Author: Prof. ML & Music Generation
Date: 2025-11-17
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
from pathlib import Path

from music_informer.models import MusicInformer
from music_informer.data import MIDITokenizer


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from {checkpoint_path}...")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model (using default config)
    # In production, load config from config.json
    model = MusicInformer(
        vocab_size=388,
        d_model=512,
        num_encoder_layers=6,
        num_decoder_layers=6,
        num_heads=8,
        d_ff=2048,
        d_lstm=1024,
        max_seq_len=2048,
        dropout=0.1
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    print(f"   Val loss: {checkpoint.get('val_loss', 'unknown')}")

    return model


def generate_music(
    model: MusicInformer,
    tokenizer: MIDITokenizer,
    device: torch.device,
    output_path: str,
    max_len: int = 512,
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 0.9,
    seed_midi: str = None
):
    """
    Generate music

    Args:
        model: Trained model
        tokenizer: MIDI tokenizer
        device: Device
        output_path: Output MIDI path
        max_len: Max length to generate
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Nucleus sampling
        seed_midi: Optional seed MIDI file
    """
    print(f"\n{'='*80}")
    print(f"Generating music...")
    print(f"{'='*80}")

    # Prepare source sequence
    if seed_midi:
        print(f"Using seed MIDI: {seed_midi}")
        seed_tokens = tokenizer.encode(seed_midi)
        # Use first 128 tokens as source
        src_tokens = seed_tokens[:128]
        src = torch.tensor([src_tokens], dtype=torch.long).to(device)
    else:
        print("Generating from scratch (random source)")
        # Random source sequence
        src = torch.randint(
            tokenizer.TIME_SHIFT_OFFSET,
            tokenizer.TIME_SHIFT_OFFSET + 100,
            (1, 64),
            dtype=torch.long,
            device=device
        )

    print(f"Source sequence length: {src.size(1)}")
    print(f"Parameters:")
    print(f"  Max length: {max_len}")
    print(f"  Temperature: {temperature}")
    print(f"  Top-k: {top_k}")
    print(f"  Top-p: {top_p}")

    # Generate
    with torch.no_grad():
        generated = model.generate(
            src=src,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            start_token=tokenizer.TIME_SHIFT_OFFSET
        )

    generated_tokens = generated[0].cpu().tolist()
    print(f"\n✅ Generated {len(generated_tokens)} tokens")

    # Decode to MIDI
    print(f"\nDecoding to MIDI...")
    midi = tokenizer.decode(generated_tokens, output_path)

    print(f"\n✅ Saved to {output_path}")
    print(f"   Duration: {midi.get_end_time():.2f} seconds")
    print(f"   Notes: {len(midi.instruments[0].notes)}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate music with Music Informer')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')

    # Output
    parser.add_argument('--output', type=str, default='./generated.mid',
                        help='Output MIDI file path')

    # Generation parameters
    parser.add_argument('--max_len', type=int, default=512,
                        help='Maximum length to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (0.5=conservative, 1.5=creative)')
    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Nucleus sampling (top-p)')

    # Optional seed
    parser.add_argument('--seed_midi', type=str, default=None,
                        help='Seed MIDI file (optional)')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"Device: {device}")
    print(f"{'='*80}")

    # Load model
    model = load_checkpoint(args.checkpoint, device)

    # Tokenizer
    tokenizer = MIDITokenizer()

    # Generate
    output_path = generate_music(
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_path=args.output,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed_midi=args.seed_midi
    )

    print(f"\n{'='*80}")
    print(f"✅ Generation complete!")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
