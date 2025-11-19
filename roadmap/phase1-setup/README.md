# Phase 1: 환경 세팅 + 데이터 수집 (Month 1)

**목표**: 완전 무료로 개발 환경 세팅하고 찰리 파커 MIDI 데이터 수집

**예산**: $0 (완전 무료!)

**소요 시간**: 주말 2-3일 (총 10-15시간)

---

## 📋 체크리스트

- [ ] **Week 1**: Google Colab 세팅 (2시간)
- [ ] **Week 2**: MIDI 데이터 수집 50-100곡 (4시간)
- [ ] **Week 3**: 데이터 전처리 파이프라인 구축 (6시간)
- [ ] **Week 4**: 데이터 탐색 및 분석 (3시간)

**완료 조건**: 찰리 파커 MIDI 50곡이 토큰으로 변환되어 저장됨

---

## 🎯 학습 목표

이번 Phase에서 배울 것들:
1. ✅ Google Colab 무료로 GPU 사용하는 법
2. ✅ MIDI 파일 다운로드 및 관리
3. ✅ MIDI → 토큰 변환 (Tokenization)
4. ✅ 데이터 통계 분석 (평균 길이, 음역대, 리듬 패턴 등)

---

## 📁 파일 구조

```
phase1-setup/
├── README.md                      # 이 파일
├── 01_colab_setup.md             # Colab 세팅 가이드
├── 02_data_collection.md         # 데이터 수집 가이드
├── 03_midi_preprocessing.py      # MIDI 전처리 코드
├── 04_data_exploration.ipynb     # 데이터 탐색 노트북
├── requirements.txt              # 필요한 라이브러리
└── data/                         # 데이터 폴더 (gitignore)
    ├── raw/                      # 원본 MIDI 파일
    ├── processed/                # 전처리된 데이터
    └── stats.json                # 데이터 통계
```

---

## 🚀 Step-by-Step 가이드

### Step 1: Google Colab 세팅 (30분)

**왜 Colab?**
- ✅ 완전 무료
- ✅ GPU 제공 (T4 GPU, 12시간/일)
- ✅ 환경 설정 필요 없음
- ✅ 언제 어디서나 접속 가능

**따라하기**:
1. [01_colab_setup.md](./01_colab_setup.md) 읽기
2. https://colab.research.google.com/ 접속
3. Google 계정으로 로그인
4. 첫 노트북 만들기

**확인 방법**:
```python
# Colab 노트북에서 실행
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

출력 예시:
```
PyTorch version: 2.0.1+cu118
CUDA available: True
GPU: Tesla T4
```

---

### Step 2: 라이브러리 설치 (10분)

**필요한 라이브러리**:
```bash
pip install pretty_midi
pip install mido
pip install numpy
pip install matplotlib
pip install tqdm
```

또는 `requirements.txt` 사용:
```bash
pip install -r requirements.txt
```

**[requirements.txt](./requirements.txt)** 파일 참고

---

### Step 3: MIDI 데이터 수집 (2-4시간)

**찰리 파커 MIDI 다운로드**:

**옵션 1: 무료 MIDI 사이트**
- https://www.midiworld.com/ (검색: "Charlie Parker")
- https://freemidi.org/ (Jazz 섹션)
- https://www.kunstderfuge.com/ (클래식 포함)

**옵션 2: 자동 다운로드 스크립트**
- [02_data_collection.md](./02_data_collection.md)에서 스크립트 제공
- **주의**: 저작권 확인 후 사용

**목표**: 최소 50곡, 이상적으로 100곡

**다운로드 후 정리**:
```bash
roadmap/phase1-setup/data/raw/charlie_parker/
├── confirmation.mid
├── ornithology.mid
├── anthropology.mid
├── scrapple_from_the_apple.mid
└── ... (총 50-100곡)
```

**체크리스트**:
- [ ] 찰리 파커 MIDI 50곡 다운로드
- [ ] `data/raw/charlie_parker/` 폴더에 저장
- [ ] 파일명을 영어로 정리 (한글 파일명 피하기)

---

### Step 4: MIDI 전처리 (3-4시간)

**MIDI → 토큰 변환**

**왜 토큰화?**
- MIDI 파일은 그대로 학습 불가
- 토큰 = 모델이 이해할 수 있는 형태 (정수 시퀀스)

**토큰화 방법** (간단한 버전):
```python
# 토큰 종류:
# 0-127: NOTE_ON (MIDI 음높이)
# 128-255: NOTE_OFF
# 256-355: TIME_SHIFT (100개, 0-10초를 100ms 간격)
# 356-387: VELOCITY (32 levels)
# 388: PAD
# 389: BOS (Begin of Sequence)
# 390: EOS (End of Sequence)

# 총 어휘 크기: 391
```

**전처리 스크립트 실행**:
```bash
python 03_midi_preprocessing.py \
  --input_dir data/raw/charlie_parker \
  --output_dir data/processed \
  --max_seq_len 512
```

**[03_midi_preprocessing.py](./03_midi_preprocessing.py)** 파일 참고

**결과**:
```bash
data/processed/
├── train.pkl          # 학습 데이터 (80%)
├── val.pkl            # 검증 데이터 (10%)
├── test.pkl           # 테스트 데이터 (10%)
└── vocab.json         # 어휘 사전
```

**체크리스트**:
- [ ] `03_midi_preprocessing.py` 실행
- [ ] `train.pkl`, `val.pkl`, `test.pkl` 생성 확인
- [ ] 파일 크기 확인 (train.pkl이 가장 커야 함)

---

### Step 5: 데이터 탐색 (2-3시간)

**데이터 통계 확인**

**Jupyter Notebook 실행**:
```bash
# Colab에서 04_data_exploration.ipynb 열기
```

**[04_data_exploration.ipynb](./04_data_exploration.ipynb)** 파일 참고

**확인할 통계**:
1. **시퀀스 길이 분포**
   - 평균: ~400 토큰
   - 최소/최대: 100 ~ 512 토큰

2. **음높이 분포**
   - 찰리 파커는 주로 C4-C6 (MIDI 60-84)
   - 히스토그램으로 시각화

3. **리듬 패턴**
   - TIME_SHIFT 토큰 분포
   - 스윙 리듬 확인 (8분음표 타이밍)

4. **벨로시티 분포**
   - 평균: 70-90 (mf-f)
   - 다이나믹 레인지 확인

**출력 예시**:
```
Dataset Statistics:
==================
Total sequences: 87
Train: 69 (80%)
Val: 9 (10%)
Test: 9 (10%)

Sequence Length:
  Mean: 398.5 tokens
  Std: 112.3 tokens
  Min: 156 tokens
  Max: 512 tokens

Pitch Range:
  Mean: 72.3 (C5)
  Min: 52 (E3)
  Max: 91 (G6)
  Most common: 67 (G4), 69 (A4), 72 (C5)

Rhythm:
  8th notes: 45%
  16th notes: 30%
  Quarter notes: 20%
  Other: 5%

Velocity:
  Mean: 82.5
  Std: 15.2
```

**체크리스트**:
- [ ] 데이터 통계 확인
- [ ] 히스토그램 시각화
- [ ] 이상한 값 없는지 체크 (예: 음높이 0 또는 127 너무 많으면 문제)

---

## ✅ Phase 1 완료 조건

다음을 모두 완료하면 Phase 1 성공:

1. ✅ Google Colab에서 GPU 사용 가능
2. ✅ 찰리 파커 MIDI 50곡 이상 수집
3. ✅ `train.pkl`, `val.pkl`, `test.pkl` 생성
4. ✅ 데이터 통계 확인 및 시각화
5. ✅ GitHub에 코드 업로드 (데이터는 gitignore)

**확인 방법**:
```python
import pickle

# 데이터 로드 테스트
with open('data/processed/train.pkl', 'rb') as f:
    train_data = pickle.load(f)

print(f"Train sequences: {len(train_data)}")
print(f"First sequence length: {len(train_data[0])}")
print(f"First 20 tokens: {train_data[0][:20]}")
```

출력 예시:
```
Train sequences: 69
First sequence length: 412
First 20 tokens: [389, 67, 356, 256, 69, 358, 257, 72, 360, 258, 128+67, 256, 74, 362, ...]
```

---

## 🎓 학습 포인트

**이번 Phase에서 배운 것**:
1. ✅ **Google Colab**: 무료 GPU 클라우드 사용법
2. ✅ **MIDI 처리**: pretty_midi 라이브러리 활용
3. ✅ **토큰화**: 음악을 정수 시퀀스로 변환
4. ✅ **데이터 분석**: 통계 및 시각화

**다음 Phase 준비**:
- Train/Val/Test 데이터 준비 완료
- 이제 모델 학습 준비됨
- Phase 2에서 첫 번째 모델 학습!

---

## 🐛 문제 해결

### Q1: MIDI 파일이 제대로 로드 안 됨
```python
# 문제: "Could not open MIDI file"
# 해결: MIDI 파일 형식 확인

import mido
mid = mido.MidiFile('problematic.mid')
print(mid)  # 에러 메시지 확인
```

### Q2: 토큰 시퀀스가 너무 짧음 (<100 토큰)
```python
# 문제: 매우 짧은 MIDI 파일
# 해결: 최소 길이 필터링

# 03_midi_preprocessing.py에서
MIN_SEQ_LEN = 100  # 100 토큰 미만 제거
```

### Q3: Colab GPU가 할당 안 됨
```
문제: "GPU not available"
해결:
1. 런타임 > 런타임 유형 변경 > GPU 선택
2. 무료 할당량 초과 → 다음날 재시도
3. Colab Pro 고려 ($10/월, 더 많은 GPU 시간)
```

---

## 📚 추가 학습 자료

**MIDI 이해하기**:
- https://en.wikipedia.org/wiki/MIDI
- https://www.midi.org/specifications

**pretty_midi 튜토리얼**:
- https://craffel.github.io/pretty-midi/
- https://github.com/craffel/pretty-midi

**음악 이론 기초** (선택):
- 스케일, 코드, 리듬
- 재즈는 몰라도 학습 가능하지만, 알면 더 좋음

---

## 🎯 다음 단계

Phase 1 완료했으면:
- ✅ GitHub에 푸시
- ✅ 체크리스트 모두 완료 확인
- ✅ **Phase 2로 이동** → 첫 번째 모델 학습!

**→ [Phase 2 시작하기](../phase2-small-model/README.md)** 🚀

---

**축하합니다! Phase 1 완료!** 🎉

이제 실제 데이터가 있고, 학습 준비가 끝났어요.
Phase 2에서 첫 번째 AI 모델을 학습해봅시다!
