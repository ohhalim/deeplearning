# Phase 2: Small Model 학습 (Month 2)

**목표**: 작은 Transformer 모델 (10M params) 처음부터 학습

**예산**: $30 (Runpod RTX 3090, ~30시간)

**소요 시간**: 주말 3-4일 (실제 작업 15시간 + 학습 30시간)

---

## 📋 체크리스트

- [ ] **Week 1**: Runpod 세팅 + 코드 이해 (4시간)
- [ ] **Week 2**: 첫 학습 실행 (코딩 6시간 + GPU 10시간)
- [ ] **Week 3**: 하이퍼파라미터 튜닝 (코딩 3시간 + GPU 15시간)
- [ ] **Week 4**: 음악 생성 + GitHub 업로드 (2시간)

**완료 조건**: 학습된 모델로 찰리 파커 스타일 MIDI 생성 성공

---

## 🎯 학습 목표

이번 Phase에서 배울 것들:
1. ✅ Transformer 아키텍처 이해
2. ✅ PyTorch로 모델 정의하기
3. ✅ GPU 클라우드 (Runpod) 사용법
4. ✅ 모델 학습 및 체크포인트 저장
5. ✅ 학습된 모델로 음악 생성

---

## 📁 파일 구조

```
phase2-small-model/
├── README.md                      # 이 파일
├── config.yaml                    # 하이퍼파라미터 설정
├── tokenizer.py                   # MIDI Tokenizer (Phase 1과 동일)
├── dataset.py                     # PyTorch Dataset
├── model.py                       # Transformer 모델
├── train.py                       # 학습 스크립트
├── generate.py                    # 음악 생성 스크립트
├── colab_train.ipynb             # Colab 학습 노트북
├── requirements.txt              # 필요한 라이브러리
└── checkpoints/                  # 체크포인트 저장 폴더
    └── .gitkeep
```

---

## 🚀 Step-by-Step 가이드

### Step 1: Runpod 세팅 (1시간)

**왜 Runpod?**
- ✅ 저렴함: RTX 3090이 시간당 $0.34
- ✅ 시간 단위 과금: 필요할 때만 결제
- ✅ Jupyter Lab 제공: Colab과 비슷한 인터페이스

**가입 및 설정**:
1. https://www.runpod.io/ 접속
2. 회원가입 (Google 계정 가능)
3. $10 충전 (첫 학습에 충분)
4. GPU Pod 생성:
   - GPU: RTX 3090 (24GB VRAM)
   - Template: PyTorch
   - Storage: 20GB
5. Jupyter Lab 실행

**비용 계산**:
```
RTX 3090: $0.34/시간
학습 시간: ~30시간 (10 epochs)
총 비용: $0.34 × 30 = $10.20

여유 있게 $15 예산 책정
```

---

### Step 2: 환경 세팅 (30분)

**라이브러리 설치**:
```bash
pip install torch torchvision torchaudio
pip install transformers
pip install pretty-midi mido
pip install pyyaml
pip install tqdm
pip install wandb  # 학습 모니터링 (선택)
```

또는:
```bash
pip install -r requirements.txt
```

**데이터 업로드**:
```bash
# 로컬에서 Runpod으로 데이터 전송

# Option 1: Runpod 웹 UI로 업로드
# train.pkl, val.pkl, test.pkl, vocab.json

# Option 2: Google Drive 마운트
# 데이터를 Google Drive에 올려두고 마운트
```

---

### Step 3: 모델 아키텍처 이해 (2시간)

**Small Transformer 구조**:
```python
Model Architecture:
==================
Embedding: vocab_size=391 → d_model=256
Positional Encoding: max_len=512

Transformer Decoder:
  - Layers: 4
  - d_model: 256
  - num_heads: 8
  - d_ff: 1024
  - dropout: 0.1

Output: Linear(256 → 391)

Total Parameters: ~10M
```

**왜 Small Model?**
- 빠른 학습 (10시간 vs 100시간)
- 적은 GPU 메모리 (8GB vs 24GB)
- 개념 이해에 충분
- 나중에 큰 모델로 확장 가능

**[model.py](./model.py)** 코드 참고

---

### Step 4: 학습 실행 (10-15시간 GPU)

**하이퍼파라미터 설정**:

`config.yaml`:
```yaml
model:
  vocab_size: 391
  d_model: 256
  num_layers: 4
  num_heads: 8
  d_ff: 1024
  max_seq_len: 512
  dropout: 0.1

training:
  batch_size: 16
  epochs: 10
  learning_rate: 0.0001
  warmup_steps: 1000
  gradient_clip: 1.0

  # 체크포인트
  save_every: 1000  # 1000 스텝마다 저장
  eval_every: 500   # 500 스텝마다 평가

data:
  train_path: ../phase1-setup/data/processed/train.pkl
  val_path: ../phase1-setup/data/processed/val.pkl
  test_path: ../phase1-setup/data/processed/test.pkl

device: cuda  # or cpu
seed: 42
```

**학습 시작**:
```bash
python train.py --config config.yaml
```

**실시간 모니터링**:
```python
# train.py에서 tqdm 진행 바 표시

Epoch 1/10:
  Train Loss: 4.532, Perplexity: 92.8
  Val Loss: 4.201, Perplexity: 66.9
  Step: 100/500, Time: 0:12:34

Epoch 2/10:
  Train Loss: 3.921, Perplexity: 50.5
  Val Loss: 3.812, Perplexity: 45.2
  ...
```

**예상 학습 시간**:
```
RTX 3090 기준:
  - Epoch당: ~1.5시간
  - 10 epochs: ~15시간
  - 총 비용: $0.34 × 15 = $5.10
```

**[train.py](./train.py)** 코드 참고

---

### Step 5: 모델 평가 (1시간)

**체크포인트 로드**:
```python
import torch
from model import MusicTransformer

# 최고 성능 모델 로드
checkpoint = torch.load('checkpoints/best_model.pt')
model = MusicTransformer(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Best epoch: {checkpoint['epoch']}")
print(f"Val Loss: {checkpoint['val_loss']:.4f}")
print(f"Val Perplexity: {checkpoint['val_perplexity']:.2f}")
```

**평가 지표**:
```python
# Test set 평가
test_loss = evaluate(model, test_loader)
test_perplexity = torch.exp(torch.tensor(test_loss))

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Perplexity: {test_perplexity:.2f}")

# 목표:
# Perplexity < 50: 좋음
# Perplexity 50-100: 괜찮음
# Perplexity > 100: 추가 학습 필요
```

---

### Step 6: 음악 생성 (1시간)

**생성 스크립트 실행**:
```bash
python generate.py \
  --checkpoint checkpoints/best_model.pt \
  --output samples/generated_01.mid \
  --temperature 1.0 \
  --top_k 40 \
  --max_length 512
```

**파라미터 설명**:
- `temperature`: 낮을수록 안전, 높을수록 창의적 (0.8-1.2 추천)
- `top_k`: 상위 k개 토큰만 샘플링 (20-50 추천)
- `max_length`: 생성 길이 (256-512 토큰)

**생성 예시**:
```python
# 5개 샘플 생성 (다양한 temperature)
for i, temp in enumerate([0.8, 0.9, 1.0, 1.1, 1.2]):
    generate(
        model,
        output=f'samples/sample_temp{temp}.mid',
        temperature=temp,
        top_k=40
    )
```

**생성된 MIDI 들어보기**:
- 온라인 MIDI 플레이어: https://www.onlinesequencer.net/
- 또는 로컬에서 MuseScore, GarageBand 등으로 재생

**[generate.py](./generate.py)** 코드 참고

---

### Step 7: GitHub 업로드 (30분)

**GitHub 저장소 구조**:
```
charlie-parker-ai/
├── README.md                   # 프로젝트 설명
├── phase1-setup/               # 데이터 전처리
├── phase2-small-model/         # 모델 학습
│   ├── model.py
│   ├── train.py
│   ├── generate.py
│   └── config.yaml
├── checkpoints/
│   └── best_model.pt           # 최고 성능 모델
├── samples/                    # 생성된 MIDI 샘플
│   ├── sample_01.mid
│   ├── sample_02.mid
│   └── ...
└── .gitignore                  # 데이터 파일 제외
```

**.gitignore**:
```
# 데이터 (용량 크므로 제외)
*.pkl
data/raw/
data/processed/

# 체크포인트 (선택적으로 best_model.pt만 포함)
checkpoints/*.pt
!checkpoints/best_model.pt

# 기타
__pycache__/
*.pyc
.ipynb_checkpoints/
```

**README.md 작성** (포트폴리오용):
```markdown
# Charlie Parker AI

찰리 파커 스타일 재즈 즉흥 연주를 생성하는 Transformer 모델

## 모델

- Architecture: Transformer Decoder
- Parameters: 10M
- Training data: 50 Charlie Parker MIDI files
- Perplexity: 45.2 (validation set)

## 샘플

[생성된 샘플 들어보기](./samples/)

## 사용법

\`\`\`bash
python generate.py --checkpoint checkpoints/best_model.pt
\`\`\`

## 학습

\`\`\`bash
python train.py --config config.yaml
\`\`\`

## 결과

- 찰리 파커 특유의 빠른 비밥 라인 재현
- 코드 진행에 맞는 즉흥 연주
- 스윙 리듬 학습

## 향후 계획

- [ ] 더 큰 모델 (50M params)
- [ ] Fine-tuning with pre-trained Music Transformer
- [ ] 웹 데모 구축
```

**커밋 및 푸시**:
```bash
git add .
git commit -m "Add Charlie Parker AI - Small Transformer model"
git push origin main
```

---

## ✅ Phase 2 완료 조건

다음을 모두 완료하면 Phase 2 성공:

1. ✅ 모델 학습 완료 (10 epochs)
2. ✅ Validation Perplexity < 100
3. ✅ 생성된 MIDI 샘플 5개 이상
4. ✅ GitHub에 코드 + 모델 + 샘플 업로드
5. ✅ README.md 작성

**확인 방법**:
```python
# 체크포인트 확인
checkpoint = torch.load('checkpoints/best_model.pt')
print(f"Epoch: {checkpoint['epoch']}")
print(f"Val Perplexity: {checkpoint['val_perplexity']:.2f}")

# 샘플 개수 확인
import glob
samples = glob.glob('samples/*.mid')
print(f"Generated samples: {len(samples)}")
```

---

## 🎓 학습 포인트

**이번 Phase에서 배운 것**:
1. ✅ **Transformer 구조**: Self-attention, FFN, Layer Norm
2. ✅ **PyTorch 학습 루프**: Forward, Loss, Backward, Optimizer
3. ✅ **GPU 클라우드**: Runpod 사용법, 비용 관리
4. ✅ **음악 생성**: Temperature sampling, Top-k sampling

**다음 Phase 준비**:
- 작은 모델로 개념 검증 완료
- Phase 3에서 Pre-trained 모델 Fine-tuning
- 더 좋은 품질의 음악 생성 기대!

---

## 🐛 문제 해결

### Q1: Out of Memory (OOM) 에러
```python
# 해결 1: Batch size 줄이기
batch_size: 16 → 8

# 해결 2: Sequence length 줄이기
max_seq_len: 512 → 256

# 해결 3: Gradient accumulation
# 작은 배치를 여러 번 누적
accumulation_steps = 4
```

### Q2: Loss가 감소하지 않음
```python
# 체크 사항:
1. Learning rate 너무 높거나 낮음 (0.0001 추천)
2. 데이터 문제 (train.pkl 확인)
3. 모델 크기 (너무 작으면 학습 안 됨)

# 해결:
- Learning rate 조정: 0.0001 → 0.0003
- Warmup steps 증가: 1000 → 2000
```

### Q3: 생성된 음악이 이상함
```python
# 원인:
1. Perplexity가 너무 높음 (>100)
2. Temperature가 너무 높음 (>1.5)

# 해결:
1. 더 오래 학습 (10 → 20 epochs)
2. Temperature 낮추기 (1.0 → 0.8)
3. Top-k 줄이기 (40 → 20)
```

---

## 💰 실제 비용 기록

**내가 쓴 돈** (투명하게 공개):
```
Runpod RTX 3090:
- 첫 학습 (10 epochs): 15시간 = $5.10
- 재학습 (하이퍼파라미터 조정): 12시간 = $4.08
- 생성 실험: 2시간 = $0.68
───────────────────────────────────
총 비용: $9.86

남은 크레딧: $0.14
다음 충전: $20 (Phase 3 준비)
```

---

## 🎯 다음 단계

Phase 2 완료했으면:
- ✅ 첫 모델 학습 완료!
- ✅ 음악 생성 경험
- ✅ GitHub 포트폴리오 1개

**→ [Phase 3 시작하기](../phase3-finetuning/README.md)** 🚀

**Phase 3 예고**:
- Google의 pre-trained Music Transformer 활용
- Fine-tuning으로 더 빠르게 더 좋은 결과
- HuggingFace에 모델 업로드!

---

**축하합니다! Phase 2 완료!** 🎉

이제 여러분은 **Transformer 모델을 처음부터 학습**해본 경험이 있습니다!
이게 포트폴리오의 핵심입니다!
