"""
MAESTRO Dataset Loader - 실제로 작동하는 구현
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from typing import List, Optional
import numpy as np
from .tokenizer import MIDITokenizer


class MAESTRODataset(Dataset):
    """
    MAESTRO 데이터셋 로더

    사용법:
        dataset = MAESTRODataset(
            data_dir='./maestro-v3.0.0',
            split='train',
            max_seq_len=2048
        )
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        max_seq_len: int = 2048,
        tokenizer: Optional[MIDITokenizer] = None
    ):
        """
        Args:
            data_dir: MAESTRO 데이터셋 디렉토리
            split: 'train', 'validation', 'test'
            max_seq_len: 최대 시퀀스 길이
            tokenizer: MIDI 토크나이저 (None이면 기본값 사용)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_seq_len = max_seq_len

        # Tokenizer
        self.tokenizer = tokenizer if tokenizer else MIDITokenizer()

        # MAESTRO metadata 로드
        metadata_path = self.data_dir / 'maestro-v3.0.0.json'
        if not metadata_path.exists():
            # Try alternative name
            metadata_path = self.data_dir / 'maestro-v2.0.0.json'
            if not metadata_path.exists():
                # Try without version
                metadata_paths = list(self.data_dir.glob('maestro*.json'))
                if len(metadata_paths) > 0:
                    metadata_path = metadata_paths[0]
                else:
                    raise FileNotFoundError(
                        f"MAESTRO metadata not found in {self.data_dir}\n"
                        f"Please download MAESTRO dataset from:\n"
                        f"https://magenta.tensorflow.org/datasets/maestro"
                    )

        print(f"Loading metadata from {metadata_path}")
        with open(metadata_path) as f:
            metadata = json.load(f)

        # Split에 해당하는 파일 필터링
        self.midi_files = []
        for entry in metadata:
            if entry['split'] == split:
                midi_path = self.data_dir / entry['midi_filename']
                if midi_path.exists():
                    self.midi_files.append(str(midi_path))

        if len(self.midi_files) == 0:
            raise ValueError(
                f"No MIDI files found for split '{split}' in {self.data_dir}\n"
                f"Check if MAESTRO dataset is properly downloaded."
            )

        print(f"Loaded {len(self.midi_files)} MIDI files for split '{split}'")

        # Cache for tokenized sequences
        self._cache = {}

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx):
        """
        Returns:
            src: (seq_len,) - input sequence
            tgt: (seq_len,) - target sequence (shifted by 1)
        """
        # Cache check
        if idx in self._cache:
            return self._cache[idx]

        # MIDI 파일 로드 & 토큰화
        midi_path = self.midi_files[idx]

        try:
            tokens = self.tokenizer.encode(midi_path)
        except Exception as e:
            print(f"Error encoding {midi_path}: {e}")
            # Return PAD tokens
            tokens = [self.tokenizer.PAD_TOKEN] * self.max_seq_len

        # Truncate or pad
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len]
        else:
            # Pad
            tokens = tokens + [self.tokenizer.PAD_TOKEN] * (self.max_seq_len - len(tokens))

        tokens = torch.tensor(tokens, dtype=torch.long)

        # Create source and target
        # Source: tokens[:-1]
        # Target: tokens[1:]
        src = tokens[:-1]
        tgt = tokens[1:]

        # Cache
        self._cache[idx] = (src, tgt)

        return src, tgt

    def get_vocab_size(self):
        """Vocabulary 크기"""
        return self.tokenizer.vocab_size


def collate_fn(batch):
    """
    Batch collation function

    Args:
        batch: List of (src, tgt) tuples

    Returns:
        src_batch: (batch_size, seq_len)
        tgt_batch: (batch_size, seq_len)
    """
    src_list, tgt_list = zip(*batch)

    src_batch = torch.stack(src_list, dim=0)
    tgt_batch = torch.stack(tgt_list, dim=0)

    return src_batch, tgt_batch


def test_dataset():
    """데이터셋 테스트"""
    print("=" * 80)
    print("MAESTRO Dataset Test")
    print("=" * 80)

    # Check if MAESTRO exists
    maestro_dir = Path('./maestro-v3.0.0')
    if not maestro_dir.exists():
        print("\n⚠️  MAESTRO dataset not found!")
        print("\nTo download MAESTRO:")
        print("  wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip")
        print("  unzip maestro-v3.0.0-midi.zip")
        print("\nFor now, using dummy data...")

        # Create dummy data for testing
        import pretty_midi
        dummy_dir = Path('/tmp/dummy_maestro')
        dummy_dir.mkdir(exist_ok=True)

        # Create dummy metadata
        metadata = [
            {
                'split': 'train',
                'midi_filename': 'dummy_train.mid'
            }
        ]

        with open(dummy_dir / 'maestro-v3.0.0.json', 'w') as f:
            json.dump(metadata, f)

        # Create dummy MIDI
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0)
        for i in range(10):
            note = pretty_midi.Note(
                velocity=100,
                pitch=60 + i,
                start=i * 0.5,
                end=(i + 1) * 0.5
            )
            piano.notes.append(note)
        midi.instruments.append(piano)
        midi.write(str(dummy_dir / 'dummy_train.mid'))

        maestro_dir = dummy_dir

    # Load dataset
    print(f"\nLoading dataset from {maestro_dir}")
    dataset = MAESTRODataset(
        data_dir=str(maestro_dir),
        split='train',
        max_seq_len=2048
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Vocab size: {dataset.get_vocab_size()}")

    # Test one sample
    print("\n" + "=" * 80)
    print("Testing one sample...")
    print("=" * 80)

    src, tgt = dataset[0]
    print(f"Source shape: {src.shape}")
    print(f"Target shape: {tgt.shape}")
    print(f"Source (first 20): {src[:20].tolist()}")
    print(f"Target (first 20): {tgt[:20].tolist()}")

    # Test DataLoader
    print("\n" + "=" * 80)
    print("Testing DataLoader...")
    print("=" * 80)

    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )

    src_batch, tgt_batch = next(iter(dataloader))
    print(f"Batch source shape: {src_batch.shape}")
    print(f"Batch target shape: {tgt_batch.shape}")

    print("\n" + "=" * 80)
    print("✅ Dataset works!")
    print("=" * 80)


if __name__ == "__main__":
    test_dataset()
