# Month 2: Music Transformer 기초

## 학습 목표
- Transformer 아키텍처 완벽 이해
- MIDI 데이터 처리
- 음악 생성의 기본 파이프라인

## 파일 구성

### 01_basic_transformer.py
- Vanilla Transformer from scratch 구현
- Multi-head Attention
- Positional Encoding
- Music Transformer 모델

**실행**:
```bash
python 01_basic_transformer.py
```

### 02_midi_preprocessing.py
- MIDI 토크나이저 구현
- MAESTRO 데이터셋 로더
- Token vocabulary 설계

**실행**:
```bash
python 02_midi_preprocessing.py
```

### 03_training.py
- 전체 학습 파이프라인
- Learning rate warmup
- Checkpointing
- Sample generation

**실행** (MAESTRO 다운로드 후):
```bash
python 03_training.py
```

## 필수 라이브러리

```bash
pip install torch torchvision torchaudio
pip install pretty_midi mido music21
pip install tqdm wandb
```

## 데이터셋 다운로드

```bash
# MAESTRO v3.0.0 (약 50GB)
wget https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip
unzip maestro-v3.0.0-midi.zip
```

## 학습 체크리스트

- [ ] "Attention Is All You Need" 논문 3회 읽기
- [ ] "Music Transformer" 논문 정독
- [ ] Transformer 구현 완료
- [ ] MAESTRO 데이터셋 처리
- [ ] 모델 학습 (최소 20 epoch)
- [ ] 8-bar 멜로디 생성 성공
- [ ] Perplexity < 10 달성

## 참고 자료

- [Music Transformer 논문](https://arxiv.org/abs/1809.04281)
- [MAESTRO 데이터셋](https://magenta.tensorflow.org/datasets/maestro)
- [참고 구현](https://github.com/jason9693/MusicTransformer-pytorch)
