"""
MIDI 전처리 스크립트

MIDI 파일을 토큰 시퀀스로 변환하여 학습 가능한 형태로 만듭니다.

사용법:
    python 03_midi_preprocessing.py \
        --input_dir data/raw/charlie_parker \
        --output_dir data/processed \
        --max_seq_len 512

작성자: Your Name
날짜: 2025-11-19
"""

import argparse
import glob
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pretty_midi
from tqdm import tqdm


class MIDITokenizer:
    """
    간단한 MIDI Tokenizer

    토큰 범위:
    - 0-127: NOTE_ON (MIDI 음높이)
    - 128-255: NOTE_OFF (128 + MIDI 음높이)
    - 256-355: TIME_SHIFT (100 steps, 10ms 단위)
    - 356-387: VELOCITY (32 levels, 4 단위)
    - 388: PAD
    - 389: BOS (Begin of Sequence)
    - 390: EOS (End of Sequence)

    총 어휘 크기: 391
    """

    def __init__(self):
        self.NOTE_ON_OFFSET = 0
        self.NOTE_OFF_OFFSET = 128
        self.TIME_SHIFT_OFFSET = 256
        self.VELOCITY_OFFSET = 356

        self.PAD_TOKEN = 388
        self.BOS_TOKEN = 389
        self.EOS_TOKEN = 390

        self.VOCAB_SIZE = 391

        # 타임 시프트: 10ms 단위, 최대 1초
        self.TIME_SHIFT_BINS = 100
        self.TIME_SHIFT_MS = 10

        # 벨로시티: 32 레벨 (0-127을 32개 구간으로)
        self.VELOCITY_BINS = 32

    def encode_midi(self, midi_path: str, max_seq_len: int = 512) -> List[int]:
        """
        MIDI 파일을 토큰 시퀀스로 변환

        Args:
            midi_path: MIDI 파일 경로
            max_seq_len: 최대 시퀀스 길이

        Returns:
            토큰 리스트
        """
        try:
            midi = pretty_midi.PrettyMIDI(midi_path)
        except Exception as e:
            print(f"Error loading {midi_path}: {e}")
            return None

        # 모든 노트 이벤트 수집
        events = []
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue  # 드럼 제외

            for note in instrument.notes:
                # NOTE_ON 이벤트
                events.append({
                    'time': note.start,
                    'type': 'note_on',
                    'pitch': note.pitch,
                    'velocity': note.velocity
                })

                # NOTE_OFF 이벤트
                events.append({
                    'time': note.end,
                    'type': 'note_off',
                    'pitch': note.pitch
                })

        # 시간순 정렬
        events.sort(key=lambda x: x['time'])

        if len(events) == 0:
            return None

        # 토큰 시퀀스 생성
        tokens = [self.BOS_TOKEN]
        current_time = 0

        for event in events:
            # 시간 차이 계산 (밀리초 단위)
            time_diff_ms = int((event['time'] - current_time) * 1000)

            # TIME_SHIFT 토큰 추가
            while time_diff_ms > 0:
                # 최대 1초까지만 한 번에 표현
                shift = min(time_diff_ms, self.TIME_SHIFT_MS * self.TIME_SHIFT_BINS)
                shift_token = self.TIME_SHIFT_OFFSET + (shift // self.TIME_SHIFT_MS)

                if shift_token >= self.TIME_SHIFT_OFFSET + self.TIME_SHIFT_BINS:
                    shift_token = self.TIME_SHIFT_OFFSET + self.TIME_SHIFT_BINS - 1

                tokens.append(shift_token)
                time_diff_ms -= shift
                current_time = event['time']

            # NOTE 토큰 추가
            if event['type'] == 'note_on':
                # Velocity 토큰
                velocity_bin = event['velocity'] // (128 // self.VELOCITY_BINS)
                velocity_token = self.VELOCITY_OFFSET + min(velocity_bin, self.VELOCITY_BINS - 1)
                tokens.append(velocity_token)

                # NOTE_ON 토큰
                note_on_token = self.NOTE_ON_OFFSET + event['pitch']
                tokens.append(note_on_token)

            elif event['type'] == 'note_off':
                # NOTE_OFF 토큰
                note_off_token = self.NOTE_OFF_OFFSET + event['pitch']
                tokens.append(note_off_token)

            # 최대 길이 체크
            if len(tokens) >= max_seq_len - 1:  # -1 for EOS
                break

        # EOS 토큰 추가
        tokens.append(self.EOS_TOKEN)

        # 패딩
        while len(tokens) < max_seq_len:
            tokens.append(self.PAD_TOKEN)

        return tokens[:max_seq_len]

    def decode_tokens(self, tokens: List[int], output_path: str):
        """
        토큰 시퀀스를 MIDI 파일로 변환

        Args:
            tokens: 토큰 리스트
            output_path: 출력 MIDI 파일 경로
        """
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

        current_time = 0
        active_notes = {}  # pitch -> start_time, velocity

        for token in tokens:
            if token == self.PAD_TOKEN or token == self.EOS_TOKEN:
                break
            if token == self.BOS_TOKEN:
                continue

            # TIME_SHIFT
            if self.TIME_SHIFT_OFFSET <= token < self.TIME_SHIFT_OFFSET + self.TIME_SHIFT_BINS:
                shift_ms = (token - self.TIME_SHIFT_OFFSET) * self.TIME_SHIFT_MS
                current_time += shift_ms / 1000.0  # 초 단위로 변환

            # VELOCITY
            elif self.VELOCITY_OFFSET <= token < self.VELOCITY_OFFSET + self.VELOCITY_BINS:
                current_velocity = (token - self.VELOCITY_OFFSET) * (128 // self.VELOCITY_BINS)

            # NOTE_ON
            elif self.NOTE_ON_OFFSET <= token < self.NOTE_ON_OFFSET + 128:
                pitch = token - self.NOTE_ON_OFFSET
                active_notes[pitch] = (current_time, current_velocity)

            # NOTE_OFF
            elif self.NOTE_OFF_OFFSET <= token < self.NOTE_OFF_OFFSET + 128:
                pitch = token - self.NOTE_OFF_OFFSET
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
        print(f"Saved MIDI to {output_path}")


def process_midi_files(input_dir: str, output_dir: str, max_seq_len: int = 512,
                       train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict:
    """
    디렉토리의 모든 MIDI 파일을 처리

    Args:
        input_dir: 입력 MIDI 디렉토리
        output_dir: 출력 디렉토리
        max_seq_len: 최대 시퀀스 길이
        train_ratio: 학습 데이터 비율
        val_ratio: 검증 데이터 비율

    Returns:
        통계 정보
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # MIDI 파일 찾기
    midi_files = glob.glob(os.path.join(input_dir, '**/*.mid'), recursive=True)
    midi_files += glob.glob(os.path.join(input_dir, '**/*.midi'), recursive=True)

    print(f"Found {len(midi_files)} MIDI files")

    if len(midi_files) == 0:
        print(f"No MIDI files found in {input_dir}")
        return None

    # Tokenizer 초기화
    tokenizer = MIDITokenizer()

    # MIDI 파일 처리
    all_sequences = []
    stats = {
        'total_files': len(midi_files),
        'processed': 0,
        'failed': 0,
        'too_short': 0,
        'sequence_lengths': [],
        'pitch_range': {'min': 127, 'max': 0},
        'files': []
    }

    for midi_file in tqdm(midi_files, desc="Processing MIDI files"):
        tokens = tokenizer.encode_midi(midi_file, max_seq_len)

        if tokens is None:
            stats['failed'] += 1
            continue

        # 너무 짧은 시퀀스 제외 (BOS, EOS 제외하고 50 토큰 미만)
        non_special_tokens = [t for t in tokens if t not in [
            tokenizer.BOS_TOKEN, tokenizer.EOS_TOKEN, tokenizer.PAD_TOKEN
        ]]

        if len(non_special_tokens) < 50:
            stats['too_short'] += 1
            continue

        all_sequences.append(tokens)
        stats['processed'] += 1
        stats['sequence_lengths'].append(len(non_special_tokens))
        stats['files'].append(os.path.basename(midi_file))

        # 음높이 범위 통계
        note_tokens = [t for t in tokens if tokenizer.NOTE_ON_OFFSET <= t < tokenizer.NOTE_ON_OFFSET + 128]
        if note_tokens:
            pitches = [t - tokenizer.NOTE_ON_OFFSET for t in note_tokens]
            stats['pitch_range']['min'] = min(stats['pitch_range']['min'], min(pitches))
            stats['pitch_range']['max'] = max(stats['pitch_range']['max'], max(pitches))

    print(f"\nProcessed: {stats['processed']} files")
    print(f"Failed: {stats['failed']} files")
    print(f"Too short: {stats['too_short']} files")

    if stats['processed'] == 0:
        print("No valid sequences to save!")
        return stats

    # 데이터 분할
    np.random.seed(42)
    indices = np.random.permutation(len(all_sequences))

    train_size = int(len(all_sequences) * train_ratio)
    val_size = int(len(all_sequences) * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_data = [all_sequences[i] for i in train_indices]
    val_data = [all_sequences[i] for i in val_indices]
    test_data = [all_sequences[i] for i in test_indices]

    # 저장
    with open(os.path.join(output_dir, 'train.pkl'), 'wb') as f:
        pickle.dump(train_data, f)
    print(f"Saved {len(train_data)} training sequences to train.pkl")

    with open(os.path.join(output_dir, 'val.pkl'), 'wb') as f:
        pickle.dump(val_data, f)
    print(f"Saved {len(val_data)} validation sequences to val.pkl")

    with open(os.path.join(output_dir, 'test.pkl'), 'wb') as f:
        pickle.dump(test_data, f)
    print(f"Saved {len(test_data)} test sequences to test.pkl")

    # 어휘 사전 저장
    vocab = {
        'vocab_size': tokenizer.VOCAB_SIZE,
        'pad_token': tokenizer.PAD_TOKEN,
        'bos_token': tokenizer.BOS_TOKEN,
        'eos_token': tokenizer.EOS_TOKEN,
        'note_on_offset': tokenizer.NOTE_ON_OFFSET,
        'note_off_offset': tokenizer.NOTE_OFF_OFFSET,
        'time_shift_offset': tokenizer.TIME_SHIFT_OFFSET,
        'velocity_offset': tokenizer.VELOCITY_OFFSET,
    }

    with open(os.path.join(output_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved vocabulary to vocab.json")

    # 통계 저장
    stats['split'] = {
        'train': len(train_data),
        'val': len(val_data),
        'test': len(test_data)
    }

    with open(os.path.join(output_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to stats.json")

    return stats


def main():
    parser = argparse.ArgumentParser(description='MIDI Preprocessing')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing MIDI files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for processed data')
    parser.add_argument('--max_seq_len', type=int, default=512,
                        help='Maximum sequence length (default: 512)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training data ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation data ratio (default: 0.1)')

    args = parser.parse_args()

    print("="*60)
    print("MIDI Preprocessing")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max sequence length: {args.max_seq_len}")
    print(f"Split ratio: {args.train_ratio:.1%} train, {args.val_ratio:.1%} val, "
          f"{1 - args.train_ratio - args.val_ratio:.1%} test")
    print("="*60)

    # 처리 실행
    stats = process_midi_files(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio
    )

    if stats:
        print("\n" + "="*60)
        print("Statistics:")
        print("="*60)
        print(f"Total files: {stats['total_files']}")
        print(f"Processed: {stats['processed']}")
        print(f"Failed: {stats['failed']}")
        print(f"Too short: {stats['too_short']}")
        if stats['sequence_lengths']:
            print(f"\nSequence length:")
            print(f"  Mean: {np.mean(stats['sequence_lengths']):.1f} tokens")
            print(f"  Std: {np.std(stats['sequence_lengths']):.1f} tokens")
            print(f"  Min: {np.min(stats['sequence_lengths'])} tokens")
            print(f"  Max: {np.max(stats['sequence_lengths'])} tokens")
        print(f"\nPitch range:")
        print(f"  Min: {stats['pitch_range']['min']} (MIDI note)")
        print(f"  Max: {stats['pitch_range']['max']} (MIDI note)")
        print(f"\nData split:")
        print(f"  Train: {stats['split']['train']} sequences")
        print(f"  Val: {stats['split']['val']} sequences")
        print(f"  Test: {stats['split']['test']} sequences")
        print("="*60)
        print("\nPreprocessing completed successfully!")
        print(f"Data saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
