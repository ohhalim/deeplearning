# 🚀 Efficient Fine-tuning: LoRA & QLoRA 완전 정복

> **목표**: LoRA와 QLoRA를 완벽히 이해하고 실전 음악 생성 모델에 적용

---

## 📋 개요

대규모 언어 모델(LLM)과 음악 생성 모델을 **효율적으로 fine-tuning**하는 방법을 학습합니다.

### 왜 LoRA/QLoRA인가?

| 문제 | 해결 |
|------|------|
| 💸 Full fine-tuning 비용이 너무 높음 (65B 모델: 8× A100 필요) | LoRA: 학습 파라미터 99% 감소 |
| 💾 GPU 메모리 부족 (130GB 필요) | QLoRA: 메모리 75% 절감 (단일 GPU 가능!) |
| 🐌 학습 시간 길음 | LoRA/QLoRA: 25% 빠름 |
| 📦 여러 task 지원 어려움 | LoRA adapter: 5-10MB만 저장, 교체 가능 |

---

## 📁 파일 구성

```
efficient-finetuning-lora-qlora/
│
├── README.md (이 파일)
│
├── 01_lora_implementation.py
│   - LoRA from scratch 구현
│   - Multi-head attention with LoRA
│   - Transformer block with LoRA
│   - 파라미터 수 계산 및 비교
│
├── 02_qlora_implementation.py
│   - 4-bit NF4 quantization
│   - Double quantization
│   - QLoRA linear layer
│   - Paged optimizer
│   - 메모리 절감 시뮬레이션
│
├── 03_comparison_and_best_practices.md
│   - LoRA vs QLoRA 완전 비교
│   - 하이퍼파라미터 가이드
│   - 언제 무엇을 사용할지
│   - 실전 팁 & 트릭
│
└── 04_practical_training.py
    - HuggingFace 활용 실전 코드
    - LoRA/QLoRA fine-tuning
    - Inference & merge
    - CLI 인터페이스
```

---

## 🎯 핵심 개념

### LoRA (Low-Rank Adaptation)

**수식**:
```
W' = W_0 + ΔW = W_0 + B·A

where:
- W_0: frozen pre-trained weight (d × k)
- A: trainable (k × r), r << k
- B: trainable (d × r), r << d
```

**파라미터 감소 예시** (d=4096, k=4096):
```
Original: 4096 × 4096 = 16.7M params

LoRA (r=8):
  A: 4096 × 8 = 32K
  B: 8 × 4096 = 32K
  Total: 64K params (0.38% !)

Reduction: 99.62%
```

**장점**:
- ✅ 학습 파라미터 99% 감소
- ✅ 메모리 사용량 ~3배 감소
- ✅ 학습 속도 25% 향상
- ✅ 여러 adapter 교체 가능
- ✅ Inference 속도 동일 (merge 시)

---

### QLoRA (Quantized LoRA)

**핵심 아이디어**:
```
Base Model: FP16 (16-bit) → 4-bit NF4
LoRA Adapter: FP16 유지

메모리 = Base (4-bit) + LoRA (16-bit) + Optimizer states
```

**메모리 절감 예시** (LLaMA-65B):
```
Full FT:  130 GB (base FP16)
LoRA:     131 GB (base FP16 + LoRA)
QLoRA:     34 GB (base 4-bit + LoRA)

→ 75% 절감!
→ 단일 A100 (48GB)로 가능!
```

**핵심 기술**:
1. **NF4 (NormalFloat4)**: 정규분포 최적화 4-bit
2. **Double Quantization**: Scale factors도 양자화
3. **Paged Optimizers**: CPU-GPU paging

---

## 🚀 빠른 시작

### 1. 설치

```bash
# Python 3.10 권장
pip install torch transformers peft bitsandbytes accelerate datasets
```

### 2. LoRA로 GPT-2 Fine-tuning

```bash
python 04_practical_training.py \
  --mode lora \
  --model gpt2 \
  --dataset ./jazz_dataset \
  --output ./lora-gpt2 \
  --rank 8 \
  --alpha 16 \
  --lr 2e-4 \
  --epochs 3
```

**메모리 사용량**: ~2 GB

---

### 3. QLoRA로 LLaMA-7B Fine-tuning

```bash
python 04_practical_training.py \
  --mode qlora \
  --model meta-llama/Llama-2-7b-hf \
  --dataset ./jazz_dataset \
  --output ./qlora-llama7b \
  --rank 64 \
  --alpha 128 \
  --lr 2e-4 \
  --epochs 3
```

**메모리 사용량**: ~6 GB (vs 14 GB without QLoRA)

---

### 4. Inference

```bash
python 04_practical_training.py \
  --mode inference \
  --model gpt2 \
  --output ./lora-gpt2
```

---

### 5. Merge LoRA (Fast Inference)

```bash
python 04_practical_training.py \
  --mode merge \
  --model gpt2 \
  --output ./lora-gpt2
```

---

## 💻 코드 예시

### Python API 사용

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model with 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# Prepare for k-bit training
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

# QLoRA config
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply QLoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# → trainable params: ~67M || all params: 7B || trainable%: 0.95%

# Train!
# ...
```

---

## 📊 하이퍼파라미터 가이드

### Rank (r) 선택

| Task 복잡도 | Rank | 예시 |
|------------|------|------|
| Simple | r=4-8 | 감정 분석, 분류 |
| Medium | r=8-16 | Instruction tuning |
| Complex | r=16-64 | 수학, 코딩, reasoning |
| Very Complex | r=64-128 | Domain shift 큼 (의학, 재즈!) |

**재즈 음악 생성**: `r=64` 권장 (복잡한 화성, 리듬)

---

### Alpha (α) 선택

**수식**: `scaling = α / r`

| α 설정 | Scaling | 용도 |
|--------|---------|------|
| α = r | 1.0× | Conservative (base 보존) |
| **α = 2×r** | **2.0×** | **일반 권장** |
| α = 4×r | 4.0× | Aggressive (빠른 adaptation) |

**재즈 음악 생성**: `α = 128` (r=64일 때, scaling=2.0)

---

### Learning Rate

| Method | LR | 이유 |
|--------|-----|------|
| Full FT | 1e-5 ~ 5e-5 | 모든 파라미터 학습 |
| LoRA | 1e-4 ~ 3e-4 | LoRA만 학습 (10배 높게) |
| QLoRA | 1e-4 ~ 5e-4 | 양자화 보상 |

**재즈 음악 생성**: `2e-4` (QLoRA)

---

### Target Modules

**Attention Q, K, V, O** (필수):
```python
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

**Attention + FFN** (권장):
```python
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"      # FFN
]
```

---

## 🎓 학습 순서

### Week 1: 이론 학습
- [ ] LoRA 논문 읽기 ([arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685))
- [ ] QLoRA 논문 읽기 ([arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314))
- [ ] `01_lora_implementation.py` 실행 및 이해

### Week 2: 구현 이해
- [ ] `02_qlora_implementation.py` 실행
- [ ] NF4 quantization 원리 이해
- [ ] `03_comparison_and_best_practices.md` 정독

### Week 3: 실전 적용
- [ ] `04_practical_training.py`로 GPT-2 fine-tuning
- [ ] 작은 데이터셋으로 실험
- [ ] Hyperparameter tuning

### Week 4: 음악 모델 적용
- [ ] Magenta RealTime에 LoRA/QLoRA 적용
- [ ] 재즈 데이터로 fine-tuning
- [ ] 성능 평가

---

## 📈 성능 벤치마크

### 메모리 사용량 (LLaMA-7B)

| Method | Model Memory | Training Memory | Total |
|--------|-------------|-----------------|-------|
| Full FT | 14 GB | 28 GB | **42 GB** |
| LoRA | 14 GB | 1 GB | **15 GB** |
| QLoRA | 4 GB | 2 GB | **6 GB** |

**결론**: QLoRA는 **85% 메모리 절감**

---

### 학습 속도 (tokens/sec)

| Method | Speed | Relative |
|--------|-------|----------|
| Full FT | 1000 | 1.0× |
| LoRA | 1250 | 1.25× ⚡ |
| QLoRA | 1200 | 1.2× ⚡ |

---

### 성능 (MMLU Accuracy)

| Model | Full FT | LoRA (r=16) | QLoRA (r=64) |
|-------|---------|-------------|--------------|
| 7B | 35.1% | 34.9% | 35.0% |
| 13B | 46.9% | 46.6% | 46.7% |
| 65B | 63.4% | 63.2% | 63.1% |

**결론**: 성능 차이 < 0.5% (거의 동일!)

---

## 🎺 재즈 음악 생성 적용

### JazzFlow-RT Fine-tuning 전략

```python
# Magenta RealTime (800M) → Jazz-specific model

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=64,                    # 재즈 복잡도 → 높은 rank
    lora_alpha=128,          # 2×r
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none"
)

# 예상 결과:
# - 메모리: ~4 GB (vs 16 GB)
# - 학습 시간: ~3일 (A100, 700h data)
# - Adapter 크기: ~100 MB
# - 성능: FAD < 3.0 목표
```

---

## 🐛 트러블슈팅

### OOM (Out of Memory)

```python
# 해결 방법:
# 1. Batch size 줄이기
per_device_train_batch_size = 1

# 2. Gradient accumulation
gradient_accumulation_steps = 16

# 3. Gradient checkpointing
model.gradient_checkpointing_enable()

# 4. QLoRA 사용
load_in_4bit = True
```

---

### 학습이 안 됨 (Loss 정체)

```python
# 원인 1: Learning rate 너무 낮음
learning_rate = 2e-4  # LoRA용 (Full FT의 10배)

# 원인 2: Rank 너무 낮음
r = 16  # 복잡한 task는 16-64

# 원인 3: Target modules 누락
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
```

---

## 📚 추가 자료

### 논문
- [LoRA (ICLR 2022)](https://arxiv.org/abs/2106.09685)
- [QLoRA (NeurIPS 2023)](https://arxiv.org/abs/2305.14314)
- [Prefix Tuning](https://arxiv.org/abs/2101.00190)
- [Adapter Layers](https://arxiv.org/abs/1902.00751)

### 라이브러리
- [PEFT (HuggingFace)](https://github.com/huggingface/peft)
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)

### 튜토리얼
- [HuggingFace PEFT Docs](https://huggingface.co/docs/peft)
- [QLoRA Blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

---

## 🎯 체크리스트

- [ ] LoRA 개념 이해
- [ ] QLoRA 개념 이해
- [ ] 01_lora_implementation.py 실행
- [ ] 02_qlora_implementation.py 실행
- [ ] 03_comparison_and_best_practices.md 읽기
- [ ] 04_practical_training.py로 실습
- [ ] GPT-2 fine-tuning 성공
- [ ] Hyperparameter tuning 경험
- [ ] 음악 모델에 적용 준비 완료

---

## 💡 핵심 요약

```
✅ LoRA:
  - 학습 파라미터 99% 감소
  - 메모리 ~3배 절감
  - 성능 거의 동일 (< 0.5% 차이)

✅ QLoRA:
  - LoRA + 4-bit quantization
  - 메모리 75% 절감
  - 65B 모델을 48GB GPU 하나로!

✅ 재즈 음악 생성:
  - r=64, α=128 권장
  - QLoRA로 메모리 절약
  - 700h 데이터로 ~3일 학습
```

---

**작성일**: 2025-11-16
**최종 수정**: 2025-11-16
**버전**: 1.0

**Let's efficiently fine-tune SOTA models! 🚀**
