#!/usr/bin/env python3
"""
Music Informer - End-to-End Test

이 스크립트는 전체 파이프라인이 작동하는지 테스트합니다:
1. MIDI Tokenizer
2. Dataset Loader
3. Model Forward Pass
4. Training (1 epoch)
5. Generation
"""

import sys
import torch
import numpy as np
from pathlib import Path
import pretty_midi

# Add paths
sys.path.append('./src')
sys.path.append('./data')

from model import MusicInformer
from tokenizer import MIDITokenizer
from dataset import MAESTRODataset, collate_fn
from torch.utils.data import DataLoader


def test_tokenizer():
    """Test 1: MIDI Tokenizer"""
    print("=" * 80)
    print("Test 1: MIDI Tokenizer")
    print("=" * 80)

    tokenizer = MIDITokenizer()
    print(f"✅ Tokenizer created (vocab size: {tokenizer.vocab_size})")

    # Create test MIDI
    test_midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    # C major scale
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]
    for i, pitch in enumerate(pitches):
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=i * 0.5,
            end=(i + 1) * 0.5
        )
        piano.notes.append(note)

    test_midi.instruments.append(piano)
    test_midi.write('/tmp/test_input.mid')

    # Encode
    tokens = tokenizer.encode('/tmp/test_input.mid')
    print(f"✅ Encoded {len(tokens)} tokens")

    # Decode
    decoded_midi = tokenizer.decode(tokens, '/tmp/test_output.mid')
    print(f"✅ Decoded {len(decoded_midi.instruments[0].notes)} notes")

    print()
    return tokenizer


def test_model():
    """Test 2: Model Forward Pass"""
    print("=" * 80)
    print("Test 2: Model Forward Pass")
    print("=" * 80)

    model = MusicInformer(
        vocab_size=388,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=1024,
        d_lstm=1024,
        max_seq_len=2048,
        dropout=0.1
    )

    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created ({param_count:,} parameters)")

    # Test forward pass
    batch_size = 4
    seq_len = 128

    src = torch.randint(0, 388, (batch_size, seq_len))
    output = model(src)

    assert output.shape == (batch_size, seq_len, 388), f"Wrong output shape: {output.shape}"
    print(f"✅ Forward pass successful: {src.shape} → {output.shape}")

    # Test generation
    start_tokens = torch.randint(0, 388, (1, 10))
    generated = model.generate(start_tokens, max_len=50, temperature=1.0, top_k=40)

    assert generated.shape == (1, 50), f"Wrong generation shape: {generated.shape}"
    print(f"✅ Generation successful: {start_tokens.shape} → {generated.shape}")

    print()
    return model


def test_dataset():
    """Test 3: Dataset Loader"""
    print("=" * 80)
    print("Test 3: Dataset Loader")
    print("=" * 80)

    # Create dummy MAESTRO dataset for testing
    import json
    dummy_dir = Path('/tmp/dummy_maestro')
    dummy_dir.mkdir(exist_ok=True)

    # Create metadata
    metadata = []
    for i in range(5):
        metadata.append({
            'split': 'train',
            'midi_filename': f'dummy_train_{i}.mid'
        })

    with open(dummy_dir / 'maestro-v3.0.0.json', 'w') as f:
        json.dump(metadata, f)

    # Create dummy MIDI files
    for i in range(5):
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        for j in range(20):
            note = pretty_midi.Note(
                velocity=100,
                pitch=60 + (j % 12),
                start=j * 0.25,
                end=(j + 1) * 0.25
            )
            piano.notes.append(note)
        midi.instruments.append(piano)
        midi.write(str(dummy_dir / f'dummy_train_{i}.mid'))

    # Load dataset
    dataset = MAESTRODataset(
        data_dir=str(dummy_dir),
        split='train',
        max_seq_len=512
    )

    print(f"✅ Dataset loaded ({len(dataset)} samples)")

    # Test one sample
    src, tgt = dataset[0]
    assert src.shape == (511,), f"Wrong src shape: {src.shape}"
    assert tgt.shape == (511,), f"Wrong tgt shape: {tgt.shape}"
    print(f"✅ Sample shape correct: src={src.shape}, tgt={tgt.shape}")

    # Test DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    src_batch, tgt_batch = next(iter(dataloader))
    assert src_batch.shape == (2, 511), f"Wrong batch shape: {src_batch.shape}"
    print(f"✅ DataLoader works: batch shape={src_batch.shape}")

    print()
    return dataset, dataloader


def test_training(model, dataloader):
    """Test 4: Training Loop"""
    print("=" * 80)
    print("Test 4: Training Loop (1 iteration)")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    model.train()

    # One training step
    src, tgt = next(iter(dataloader))
    src = src.to(device)
    tgt = tgt.to(device)

    # Forward
    logits = model(src)
    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        tgt.reshape(-1)
    )

    # Backward
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    print(f"✅ Training step successful (loss: {loss.item():.4f})")

    print()
    return model


def test_generation(model, tokenizer):
    """Test 5: Music Generation"""
    print("=" * 80)
    print("Test 5: Music Generation")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # Start tokens
    start_tokens = torch.tensor([
        tokenizer.TIME_SHIFT_OFFSET,
        tokenizer.VELOCITY_OFFSET + 20,
        tokenizer.NOTE_ON_OFFSET + 60
    ], dtype=torch.long).unsqueeze(0).to(device)

    # Generate
    with torch.no_grad():
        generated = model.generate(
            start_tokens=start_tokens,
            max_len=100,
            temperature=1.0,
            top_k=40,
            top_p=0.9
        )

    print(f"✅ Generated {generated.shape[1]} tokens")

    # Decode to MIDI
    tokens = generated[0].cpu().tolist()
    midi = tokenizer.decode(tokens, '/tmp/generated.mid')

    print(f"✅ Decoded to MIDI ({len(midi.instruments[0].notes)} notes)")
    print(f"✅ Saved to /tmp/generated.mid")

    print()


def main():
    """Run all tests"""
    print("\n")
    print("🎼" * 40)
    print("Music Informer - End-to-End Test")
    print("🎼" * 40)
    print("\n")

    try:
        # Test 1: Tokenizer
        tokenizer = test_tokenizer()

        # Test 2: Model
        model = test_model()

        # Test 3: Dataset
        dataset, dataloader = test_dataset()

        # Test 4: Training
        model = test_training(model, dataloader)

        # Test 5: Generation
        test_generation(model, tokenizer)

        # Success!
        print("=" * 80)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("=" * 80)
        print("\nMusic Informer is fully functional!")
        print("\nNext steps:")
        print("1. Download MAESTRO dataset:")
        print("   wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip")
        print("   unzip maestro-v3.0.0-midi.zip")
        print("\n2. Train the model:")
        print("   python scripts/train.py --data_dir ./maestro-v3.0.0 --num_epochs 50")
        print("\n3. Generate music:")
        print("   python scripts/generate.py --checkpoint ./checkpoints/best.pt --output ./generated.mid")
        print("\n" + "=" * 80)

        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
