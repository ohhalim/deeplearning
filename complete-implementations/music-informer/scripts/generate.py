"""
Music Informer - Generation Script

실제로 작동하는 MIDI 생성 스크립트
"""

import sys
sys.path.append('../src')
sys.path.append('../data')

import torch
import argparse
from pathlib import Path

from model import MusicInformer
from tokenizer import MIDITokenizer


def generate_music(
    model_path: str,
    output_path: str,
    max_len: int = 512,
    temperature: float = 1.0,
    top_k: int = 40,
    top_p: float = 0.9,
    seed_midi: str = None
):
    """
    음악 생성

    Args:
        model_path: 학습된 모델 경로
        output_path: 출력 MIDI 파일 경로
        max_len: 생성할 최대 길이
        temperature: 샘플링 온도
        top_k: Top-k 샘플링
        top_p: Nucleus 샘플링
        seed_midi: 시드 MIDI 파일 (optional)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model = MusicInformer(
        vocab_size=388,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=1024,
        d_lstm=1024,
        max_seq_len=2048,
        dropout=0.1
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("✅ Model loaded!")

    # Tokenizer
    tokenizer = MIDITokenizer()

    # Start tokens
    if seed_midi:
        print(f"\nUsing seed MIDI: {seed_midi}")
        start_tokens = tokenizer.encode(seed_midi)
        start_tokens = start_tokens[:50]  # Use first 50 tokens as seed
    else:
        print("\nUsing random start tokens")
        # Start with TIME_SHIFT and NOTE_ON for C4
        start_tokens = [
            tokenizer.TIME_SHIFT_OFFSET,  # Time shift
            tokenizer.VELOCITY_OFFSET + 20,  # Velocity
            tokenizer.NOTE_ON_OFFSET + 60  # C4
        ]

    start_tokens = torch.tensor(start_tokens, dtype=torch.long).unsqueeze(0).to(device)

    print(f"Start tokens: {start_tokens.shape}")

    # Generate
    print(f"\nGenerating {max_len} tokens...")
    print(f"Temperature: {temperature}")
    print(f"Top-k: {top_k}")
    print(f"Top-p: {top_p}")

    with torch.no_grad():
        generated = model.generate(
            start_tokens=start_tokens,
            max_len=max_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )

    print(f"Generated: {generated.shape}")

    # Decode to MIDI
    print(f"\nDecoding to MIDI...")
    generated_tokens = generated[0].cpu().tolist()

    midi = tokenizer.decode(generated_tokens, output_path)

    print(f"✅ Saved to {output_path}")
    print(f"\nMIDI info:")
    print(f"  Duration: {midi.get_end_time():.2f} seconds")
    print(f"  Notes: {len(midi.instruments[0].notes)}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate music with Music Informer")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='./generated.mid',
                        help='Output MIDI file path')
    parser.add_argument('--max_len', type=int, default=512,
                        help='Max length to generate')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (0.5=conservative, 1.5=creative)')
    parser.add_argument('--top_k', type=int, default=40,
                        help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='Nucleus sampling')
    parser.add_argument('--seed_midi', type=str, default=None,
                        help='Seed MIDI file (optional)')

    args = parser.parse_args()

    generate_music(
        model_path=args.checkpoint,
        output_path=args.output,
        max_len=args.max_len,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed_midi=args.seed_midi
    )

    print("\n" + "=" * 80)
    print("Generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
