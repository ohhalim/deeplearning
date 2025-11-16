"""
Month 2: Music Transformer - MIDI 데이터 전처리

목표:
1. MIDI 파일 읽기 및 파싱
2. MIDI 이벤트를 토큰으로 변환
3. 데이터셋 구축
4. MAESTRO 데이터셋 처리

Required libraries:
pip install pretty_midi mido music21
"""

import os
import numpy as np
import pretty_midi
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import pickle


class MIDITokenizer:
    """
    MIDI를 토큰 시퀀스로 변환

    Token types:
    - NOTE_ON: 0-127 (MIDI note numbers)
    - NOTE_OFF: 128-255
    - TIME_SHIFT: 256-355 (100 time bins, 10ms each = 1초)
    - VELOCITY: 356-387 (32 velocity bins)
    - Special tokens: PAD, SOS, EOS
    """

    def __init__(
        self,
        num_velocity_bins=32,
        time_resolution=10,  # milliseconds
        max_time_shift=1000,  # milliseconds
    ):
        self.num_velocity_bins = num_velocity_bins
        self.time_resolution = time_resolution
        self.max_time_shift = max_time_shift
        self.num_time_bins = max_time_shift // time_resolution

        # Vocabulary
        # NOTE_ON: 0-127
        # NOTE_OFF: 128-255
        # TIME_SHIFT: 256-(256+num_time_bins-1)
        # VELOCITY: (256+num_time_bins)-(256+num_time_bins+num_velocity_bins-1)
        # Special: PAD, SOS, EOS

        self.note_on_offset = 0
        self.note_off_offset = 128
        self.time_shift_offset = 256
        self.velocity_offset = self.time_shift_offset + self.num_time_bins

        self.vocab_size = self.velocity_offset + self.num_velocity_bins + 3

        self.PAD = self.vocab_size - 3
        self.SOS = self.vocab_size - 2
        self.EOS = self.vocab_size - 1

        print(f"Vocabulary size: {self.vocab_size}")
        print(f"  - NOTE_ON: {self.note_on_offset}-{self.note_off_offset-1}")
        print(f"  - NOTE_OFF: {self.note_off_offset}-{self.time_shift_offset-1}")
        print(f"  - TIME_SHIFT: {self.time_shift_offset}-{self.velocity_offset-1}")
        print(f"  - VELOCITY: {self.velocity_offset}-{self.velocity_offset+self.num_velocity_bins-1}")
        print(f"  - PAD: {self.PAD}, SOS: {self.SOS}, EOS: {self.EOS}")

    def midi_to_tokens(self, midi_path: str) -> List[int]:
        """
        MIDI 파일을 토큰 시퀀스로 변환

        Args:
            midi_path: MIDI 파일 경로

        Returns:
            tokens: 토큰 시퀀스 리스트
        """
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            print(f"Error loading {midi_path}: {e}")
            return []

        # 모든 노트 이벤트 수집
        events = []
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue  # 드럼은 일단 제외

            for note in instrument.notes:
                # NOTE_ON event
                events.append({
                    'time': note.start,
                    'type': 'note_on',
                    'pitch': note.pitch,
                    'velocity': note.velocity
                })

                # NOTE_OFF event
                events.append({
                    'time': note.end,
                    'type': 'note_off',
                    'pitch': note.pitch
                })

        # 시간순 정렬
        events.sort(key=lambda x: x['time'])

        # 토큰으로 변환
        tokens = [self.SOS]
        current_time = 0.0

        for event in events:
            # Time shift
            time_diff = (event['time'] - current_time) * 1000  # seconds to ms
            if time_diff > 0:
                # Split into multiple time shifts if needed
                while time_diff > self.max_time_shift:
                    tokens.append(self.time_shift_offset + self.num_time_bins - 1)
                    time_diff -= self.max_time_shift
                    current_time += self.max_time_shift / 1000

                if time_diff > 0:
                    time_bin = int(time_diff / self.time_resolution)
                    time_bin = min(time_bin, self.num_time_bins - 1)
                    tokens.append(self.time_shift_offset + time_bin)
                    current_time = event['time']

            # Note event
            if event['type'] == 'note_on':
                # Velocity
                velocity_bin = int(event['velocity'] / 128 * self.num_velocity_bins)
                velocity_bin = min(velocity_bin, self.num_velocity_bins - 1)
                tokens.append(self.velocity_offset + velocity_bin)

                # Note on
                tokens.append(self.note_on_offset + event['pitch'])

            elif event['type'] == 'note_off':
                tokens.append(self.note_off_offset + event['pitch'])

        tokens.append(self.EOS)

        return tokens

    def tokens_to_midi(
        self,
        tokens: List[int],
        output_path: str,
        tempo: int = 120
    ):
        """
        토큰 시퀀스를 MIDI 파일로 변환

        Args:
            tokens: 토큰 시퀀스
            output_path: 출력 MIDI 파일 경로
            tempo: BPM
        """
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

        current_time = 0.0
        current_velocity = 64  # default velocity
        active_notes = {}  # {pitch: start_time}

        for token in tokens:
            if token == self.PAD or token == self.SOS:
                continue
            elif token == self.EOS:
                break

            # Time shift
            elif self.time_shift_offset <= token < self.velocity_offset:
                time_bin = token - self.time_shift_offset
                current_time += (time_bin * self.time_resolution) / 1000

            # Velocity
            elif self.velocity_offset <= token < self.PAD:
                velocity_bin = token - self.velocity_offset
                current_velocity = int((velocity_bin / self.num_velocity_bins) * 128)
                current_velocity = min(current_velocity, 127)

            # Note on
            elif self.note_on_offset <= token < self.note_off_offset:
                pitch = token - self.note_on_offset
                active_notes[pitch] = (current_time, current_velocity)

            # Note off
            elif self.note_off_offset <= token < self.time_shift_offset:
                pitch = token - self.note_off_offset
                if pitch in active_notes:
                    start_time, velocity = active_notes[pitch]
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time,
                        end=current_time
                    )
                    instrument.notes.append(note)
                    del active_notes[pitch]

        midi.instruments.append(instrument)
        midi.write(output_path)
        print(f"MIDI saved to {output_path}")


class MAESTRODataset(Dataset):
    """
    MAESTRO 데이터셋

    Download: https://magenta.tensorflow.org/datasets/maestro
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer: MIDITokenizer,
        max_seq_len: int = 2048,
        split: str = 'train'
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.split = split

        # MIDI 파일 찾기
        self.midi_files = self._find_midi_files()
        print(f"Found {len(self.midi_files)} MIDI files in {split} split")

    def _find_midi_files(self) -> List[Path]:
        """MIDI 파일 리스트 생성"""
        # MAESTRO 데이터셋 구조에 따라 수정 필요
        midi_files = list(self.data_dir.glob(f"**/{self.split}/*.mid"))
        midi_files += list(self.data_dir.glob(f"**/{self.split}/*.midi"))
        return midi_files

    def __len__(self):
        return len(self.midi_files)

    def __getitem__(self, idx):
        midi_path = self.midi_files[idx]

        # MIDI → tokens
        tokens = self.tokenizer.midi_to_tokens(str(midi_path))

        if len(tokens) == 0:
            # 빈 파일인 경우 패딩만
            tokens = [self.tokenizer.PAD] * self.max_seq_len

        # Sequence 길이 제한
        if len(tokens) > self.max_seq_len:
            # Random crop
            start_idx = np.random.randint(0, len(tokens) - self.max_seq_len)
            tokens = tokens[start_idx:start_idx + self.max_seq_len]
        else:
            # Padding
            tokens = tokens + [self.tokenizer.PAD] * (self.max_seq_len - len(tokens))

        return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch):
    """
    Batch collation function

    Args:
        batch: List of tensors (seq_len,)

    Returns:
        src: (batch_size, seq_len-1) - input sequence
        tgt: (batch_size, seq_len-1) - target sequence (shifted by 1)
    """
    # Stack batch
    sequences = torch.stack(batch)  # (batch_size, seq_len)

    # Input: [SOS, token1, token2, ..., tokenN-1]
    # Target: [token1, token2, ..., tokenN-1, EOS]
    src = sequences[:, :-1]
    tgt = sequences[:, 1:]

    return src, tgt


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MIDI Preprocessing Example")
    print("=" * 80)

    # Tokenizer 생성
    tokenizer = MIDITokenizer(
        num_velocity_bins=32,
        time_resolution=10,
        max_time_shift=1000
    )

    # Example: 간단한 MIDI 생성 및 토큰화
    print("\n" + "=" * 80)
    print("Creating a simple MIDI for testing")
    print("=" * 80)

    # 간단한 MIDI 파일 생성 (C major scale)
    test_midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    # C major scale: C, D, E, F, G, A, B, C
    notes = [60, 62, 64, 65, 67, 69, 71, 72]
    for i, pitch in enumerate(notes):
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=i * 0.5,
            end=(i + 1) * 0.5
        )
        piano.notes.append(note)

    test_midi.instruments.append(piano)
    test_midi.write('test_scale.mid')
    print("Created test_scale.mid")

    # Tokenize
    tokens = tokenizer.midi_to_tokens('test_scale.mid')
    print(f"\nTokenized sequence (length={len(tokens)}):")
    print(f"First 30 tokens: {tokens[:30]}")

    # Detokenize
    tokenizer.tokens_to_midi(tokens, 'reconstructed.mid')

    print("\n" + "=" * 80)
    print("다음 단계:")
    print("1. MAESTRO 데이터셋 다운로드:")
    print("   wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip")
    print("2. 압축 해제 후 경로 설정")
    print("3. Dataset 및 DataLoader 생성")
    print("4. Training loop 시작")
    print("=" * 80)

    # Dataset example (MAESTRO 다운로드 후 사용)
    # dataset = MAESTRODataset(
    #     data_dir='/path/to/maestro-v3.0.0',
    #     tokenizer=tokenizer,
    #     max_seq_len=2048,
    #     split='train'
    # )
    #
    # dataloader = DataLoader(
    #     dataset,
    #     batch_size=8,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     num_workers=4
    # )
    #
    # for src, tgt in dataloader:
    #     print(f"src: {src.shape}, tgt: {tgt.shape}")
    #     break
