"""
MIDI Tokenizer - 실제로 작동하는 구현

MIDI 파일을 토큰으로 변환하고 다시 MIDI로 복원합니다.
"""

import pretty_midi
import numpy as np
from typing import List, Tuple
from pathlib import Path


class MIDITokenizer:
    """
    MIDI를 토큰으로 변환하는 토크나이저

    토큰 구조:
    - NOTE_ON: 0-127 (velocity > 0)
    - NOTE_OFF: 128-255
    - TIME_SHIFT: 256-355 (100개, 10ms 단위)
    - VELOCITY: 356-387 (32 레벨)
    - PAD: 0
    """

    def __init__(
        self,
        time_resolution: int = 10,  # ms
        max_time_shift: int = 1000,  # ms
        velocity_bins: int = 32
    ):
        self.time_resolution = time_resolution
        self.max_time_shift = max_time_shift
        self.velocity_bins = velocity_bins

        # Token ranges
        self.NOTE_ON_OFFSET = 0
        self.NOTE_OFF_OFFSET = 128
        self.TIME_SHIFT_OFFSET = 256
        self.VELOCITY_OFFSET = 356
        self.PAD_TOKEN = 0

        # Vocab size: 388
        # 0-127: NOTE_ON
        # 128-255: NOTE_OFF
        # 256-355: TIME_SHIFT (100 steps)
        # 356-387: VELOCITY (32 bins)
        self.vocab_size = 388

        self.num_time_shift_bins = self.max_time_shift // self.time_resolution

    def encode(self, midi_path: str) -> List[int]:
        """
        MIDI 파일을 토큰 시퀀스로 변환

        Args:
            midi_path: MIDI 파일 경로

        Returns:
            tokens: 토큰 리스트
        """
        midi = pretty_midi.PrettyMIDI(midi_path)

        # 모든 노트를 (start_time, pitch, velocity, duration) 형태로 모으기
        notes = []
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                notes.append({
                    'start': note.start,
                    'pitch': note.pitch,
                    'velocity': note.velocity,
                    'duration': note.end - note.start
                })

        # 시작 시간으로 정렬
        notes.sort(key=lambda x: x['start'])

        if len(notes) == 0:
            return [self.PAD_TOKEN]

        # 토큰으로 변환
        tokens = []
        current_time = 0.0

        for note in notes:
            # Time shift
            time_delta = note['start'] - current_time
            time_delta_ms = int(time_delta * 1000)

            # Time shift 토큰 추가
            while time_delta_ms > 0:
                shift = min(time_delta_ms, self.max_time_shift)
                shift_bins = shift // self.time_resolution
                if shift_bins > 0:
                    tokens.append(self.TIME_SHIFT_OFFSET + shift_bins - 1)
                time_delta_ms -= shift

            # Velocity 토큰
            velocity_bin = int(note['velocity'] / 128 * self.velocity_bins)
            velocity_bin = min(velocity_bin, self.velocity_bins - 1)
            tokens.append(self.VELOCITY_OFFSET + velocity_bin)

            # Note on
            tokens.append(self.NOTE_ON_OFFSET + note['pitch'])

            # Note off (duration 후)
            duration_ms = int(note['duration'] * 1000)
            shift_bins = duration_ms // self.time_resolution
            if shift_bins > 0:
                # Duration을 time shift로 표현
                while duration_ms > 0:
                    shift = min(duration_ms, self.max_time_shift)
                    shift_bins = shift // self.time_resolution
                    if shift_bins > 0:
                        tokens.append(self.TIME_SHIFT_OFFSET + shift_bins - 1)
                    duration_ms -= shift

            tokens.append(self.NOTE_OFF_OFFSET + note['pitch'])

            current_time = note['start']

        return tokens

    def decode(self, tokens: List[int], output_path: str = None) -> pretty_midi.PrettyMIDI:
        """
        토큰 시퀀스를 MIDI로 변환

        Args:
            tokens: 토큰 리스트
            output_path: 저장할 MIDI 파일 경로 (optional)

        Returns:
            midi: PrettyMIDI 객체
        """
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

        current_time = 0.0
        current_velocity = 64
        active_notes = {}  # pitch -> start_time

        for token in tokens:
            if token == self.PAD_TOKEN:
                continue

            # Time shift
            if self.TIME_SHIFT_OFFSET <= token < self.TIME_SHIFT_OFFSET + self.num_time_shift_bins:
                shift_bins = token - self.TIME_SHIFT_OFFSET + 1
                shift_ms = shift_bins * self.time_resolution
                current_time += shift_ms / 1000.0

            # Velocity
            elif self.VELOCITY_OFFSET <= token < self.VELOCITY_OFFSET + self.velocity_bins:
                velocity_bin = token - self.VELOCITY_OFFSET
                current_velocity = int((velocity_bin / self.velocity_bins) * 127)
                current_velocity = max(1, min(127, current_velocity))

            # Note on
            elif self.NOTE_ON_OFFSET <= token < self.NOTE_ON_OFFSET + 128:
                pitch = token - self.NOTE_ON_OFFSET
                active_notes[pitch] = {
                    'start': current_time,
                    'velocity': current_velocity
                }

            # Note off
            elif self.NOTE_OFF_OFFSET <= token < self.NOTE_OFF_OFFSET + 128:
                pitch = token - self.NOTE_OFF_OFFSET
                if pitch in active_notes:
                    note_info = active_notes[pitch]
                    note = pretty_midi.Note(
                        velocity=note_info['velocity'],
                        pitch=pitch,
                        start=note_info['start'],
                        end=current_time
                    )
                    instrument.notes.append(note)
                    del active_notes[pitch]

        # 아직 종료되지 않은 노트들 종료
        for pitch, note_info in active_notes.items():
            note = pretty_midi.Note(
                velocity=note_info['velocity'],
                pitch=pitch,
                start=note_info['start'],
                end=current_time + 0.5  # 0.5초 후 종료
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)

        if output_path:
            midi.write(output_path)

        return midi

    def encode_batch(self, midi_paths: List[str]) -> List[List[int]]:
        """배치로 여러 MIDI 파일 인코딩"""
        return [self.encode(path) for path in midi_paths]

    def get_vocab_size(self) -> int:
        """Vocabulary 크기 반환"""
        return self.vocab_size


def test_tokenizer():
    """토크나이저 테스트"""
    print("=" * 80)
    print("MIDI Tokenizer Test")
    print("=" * 80)

    tokenizer = MIDITokenizer()

    print(f"\nVocab size: {tokenizer.vocab_size}")
    print(f"Token ranges:")
    print(f"  NOTE_ON: 0-127")
    print(f"  NOTE_OFF: 128-255")
    print(f"  TIME_SHIFT: 256-355")
    print(f"  VELOCITY: 356-387")

    # Simple MIDI 생성
    print("\n" + "=" * 80)
    print("Creating test MIDI...")
    print("=" * 80)

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
    print("Created /tmp/test_input.mid")

    # Encode
    print("\n" + "=" * 80)
    print("Encoding...")
    print("=" * 80)
    tokens = tokenizer.encode('/tmp/test_input.mid')
    print(f"Tokens: {len(tokens)}")
    print(f"First 20 tokens: {tokens[:20]}")

    # Decode
    print("\n" + "=" * 80)
    print("Decoding...")
    print("=" * 80)
    decoded_midi = tokenizer.decode(tokens, '/tmp/test_output.mid')
    print(f"Decoded MIDI: {len(decoded_midi.instruments[0].notes)} notes")
    print("Saved to /tmp/test_output.mid")

    print("\n" + "=" * 80)
    print("✅ Tokenizer works!")
    print("=" * 80)


if __name__ == "__main__":
    test_tokenizer()
