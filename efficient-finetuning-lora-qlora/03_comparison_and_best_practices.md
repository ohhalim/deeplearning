# LoRA vs QLoRA - 완전 비교 가이드

> **목표**: LoRA와 QLoRA를 완벽히 이해하고 실전에 적용

---

## 📊 한눈에 보는 비교표

| 특징 | Full Fine-tuning | LoRA | QLoRA |
|------|-----------------|------|-------|
| **메모리 (LLaMA-65B)** | ~130 GB | ~130 GB | ~48 GB |
| **학습 파라미터** | 65B (100%) | ~120M (0.2%) | ~120M (0.2%) |
| **GPU 요구사항** | 8× A100 (80GB) | 8× A100 (80GB) | 1× A100 (48GB) |
| **학습 속도** | 1x (baseline) | 1.25x (빠름) | 1.2x (빠름) |
| **성능 (vs Full)** | 100% | ~99% | ~97-99% |
| **다중 task 지원** | ✗ | ✓ (adapter swap) | ✓ (adapter swap) |
| **추론 오버헤드** | 없음 | 없음 (merge 시) | 약간 (dequantize) |

---

## 🔬 상세 비교

### 1. 메모리 사용량

#### Full Fine-tuning
```python
# 모든 파라미터 학습
Memory = Model_params × 2 (FP16)
        + Optimizer_states × 8 (Adam: m, v)
        + Gradients × 2
        ≈ Model_params × 12 bytes

# LLaMA-65B 예시
65B × 12 bytes = 780 GB (!!)
→ 다중 GPU 필수
```

#### LoRA
```python
# Base model frozen, LoRA adapter만 학습
Memory = Base_params × 2 (frozen, FP16)
        + LoRA_params × 2
        + LoRA_optimizer × 8
        + LoRA_gradients × 2

# LLaMA-65B + LoRA (r=8) 예시
Base: 65B × 2 bytes = 130 GB
LoRA trainable: ~120M
LoRA overhead: 120M × 12 = 1.44 GB

Total: ~131 GB
→ 여전히 다중 GPU 필요
```

#### QLoRA
```python
# Base model 4-bit, LoRA adapter FP16
Memory = Base_params × 0.5 (4-bit)
        + Scale_factors × 2
        + LoRA_params × 2
        + LoRA_optimizer × 8
        + LoRA_gradients × 2

# LLaMA-65B + QLoRA (r=8) 예시
Base (4-bit): 65B × 0.5 bytes = 32.5 GB
Scale: ~300 MB
LoRA overhead: 1.44 GB

Total: ~34 GB
→ 단일 A100 (48GB) 가능! 🎉
```

---

### 2. 학습 속도

| 방법 | Throughput (tokens/sec) | 상대 속도 |
|------|------------------------|---------|
| Full Fine-tuning | 1000 | 1.0x |
| LoRA | 1250 | 1.25x ⚡ |
| QLoRA | 1200 | 1.2x ⚡ |

**왜 LoRA/QLoRA가 더 빠른가?**
- 학습 파라미터 수 감소
- Backward pass 연산량 감소
- Optimizer step 빠름

---

### 3. 성능 비교 (벤치마크)

#### MMLU (Massive Multitask Language Understanding)

| Model | Full FT | LoRA (r=8) | QLoRA (r=64) |
|-------|---------|-----------|--------------|
| LLaMA-7B | 35.1% | 34.9% (-0.2) | 35.0% (-0.1) |
| LLaMA-13B | 46.9% | 46.6% (-0.3) | 46.7% (-0.2) |
| LLaMA-33B | 57.8% | 57.4% (-0.4) | 57.5% (-0.3) |
| LLaMA-65B | 63.4% | 63.2% (-0.2) | 63.1% (-0.3) |

**결론**: 성능 차이는 미미 (< 0.5%)

---

## 🎯 언제 무엇을 사용할까?

### Full Fine-tuning 사용 시

✅ **사용 권장**:
- 작은 모델 (< 1B parameters)
- GPU 리소스 충분 (8× A100)
- 최고 성능 필요
- Domain이 완전히 다름 (medical → code)

❌ **비추천**:
- 대형 모델 (> 10B)
- GPU 제한적
- 빠른 실험 필요

**예시**:
```python
# BERT-base (110M) fine-tuning
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")

# 모든 파라미터 학습
for param in model.parameters():
    param.requires_grad = True

optimizer = AdamW(model.parameters(), lr=2e-5)
# → GPU 1개로 가능, 성능 최고
```

---

### LoRA 사용 시

✅ **사용 권장**:
- 중/대형 모델 (1B-100B)
- 여러 task 동시 지원 (adapter swap)
- 빠른 실험
- Base model의 일반 지식 유지

❌ **비추천**:
- 극도로 큰 모델 + GPU 부족 (→ QLoRA)
- Domain shift 극심

**예시**:
```python
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

# LoRA config
config = LoraConfig(
    r=8,                          # Rank
    lora_alpha=16,                # Scaling
    target_modules=["q_proj", "v_proj"],  # Attention만
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
model.print_trainable_parameters()
# → trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622
```

---

### QLoRA 사용 시

✅ **사용 권장**:
- 대형 모델 (> 10B)
- GPU 메모리 부족 (< 80GB)
- 단일 GPU에서 65B 모델 fine-tuning
- 성능과 효율성 균형

❌ **비추천**:
- Inference 속도가 critical (dequantization overhead)
- 작은 모델 (오히려 복잡도 증가)

**예시**:
```python
from transformers import BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,    # Double quantization
    bnb_4bit_quant_type="nf4",         # NormalFloat4
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA on top
lora_config = LoraConfig(r=64, lora_alpha=128, ...)
model = get_peft_model(model, lora_config)
# → 48GB GPU 하나로 70B 모델 fine-tuning!
```

---

## 🛠️ 하이퍼파라미터 가이드

### LoRA Rank (r)

| Rank | 파라미터 수 | 성능 | 추천 용도 |
|------|-----------|------|---------|
| r=4 | 최소 | 85-90% | 빠른 프로토타입 |
| **r=8** | 낮음 | 95-97% | **일반 권장** |
| r=16 | 중간 | 97-99% | 복잡한 task |
| r=64 | 높음 | 99%+ | 성능 critical |
| r=128 | 매우 높음 | 99.5%+ | 극한 성능 |

**선택 가이드**:
```python
# 간단한 task (감정 분석, 분류)
r = 4-8

# 일반적인 instruction tuning
r = 8-16

# 복잡한 reasoning (수학, 코딩)
r = 16-64

# Domain shift 큼 (의학, 법률)
r = 64-128
```

---

### Alpha (α)

**수식**: `scaling = α / r`

| α | 설정 | 효과 |
|---|------|------|
| α = r | 1.0× | LoRA 기여도 낮음 |
| **α = 2×r** | **2.0×** | **일반 권장** |
| α = 4×r | 4.0× | LoRA 기여도 높음 |

**예시**:
```python
# Conservative (base model 보존)
LoraConfig(r=8, lora_alpha=8)   # scaling=1

# Recommended (균형)
LoraConfig(r=8, lora_alpha=16)  # scaling=2

# Aggressive (빠른 adaptation)
LoraConfig(r=8, lora_alpha=32)  # scaling=4
```

---

### Target Modules

**Attention 레이어**:
```python
# Minimal (가장 효율적)
target_modules = ["q_proj", "v_proj"]

# Balanced (일반 권장)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Maximal (최고 성능)
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
    "gate_proj", "up_proj", "down_proj"      # FFN
]
```

**모델별 이름**:
```python
# LLaMA/Mistral
["q_proj", "k_proj", "v_proj", "o_proj"]

# GPT-2/GPT-Neo
["c_attn", "c_proj"]

# BLOOM
["query_key_value", "dense"]

# T5
["q", "k", "v", "o"]
```

---

### Learning Rate

| Method | LR | 이유 |
|--------|-----|-----|
| Full FT | 1e-5 ~ 5e-5 | 모든 파라미터 |
| LoRA | 1e-4 ~ 3e-4 | LoRA만 학습 (더 높게) |
| QLoRA | 1e-4 ~ 5e-4 | 양자화 보상 (더 높게) |

**스케줄러**:
```python
from transformers import get_cosine_schedule_with_warmup

# Warmup: 10% of total steps
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)
```

---

## 💻 실전 코드 예시

### 1. 음악 생성 모델에 LoRA 적용

```python
"""
Magenta RealTime을 재즈 데이터로 fine-tuning (LoRA)
"""

from transformers import AutoModel
from peft import LoraConfig, get_peft_model, TaskType
import torch

# Load Magenta RealTime (가정)
model = AutoModel.from_pretrained("google/magenta-realtime")

# LoRA config for music generation
lora_config = LoraConfig(
    r=16,                      # 음악은 복잡 → rank 높게
    lora_alpha=32,
    target_modules=[
        "attention.q_proj",
        "attention.k_proj",
        "attention.v_proj",
        "attention.o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA
model = get_peft_model(model, lora_config)

# Check trainable parameters
trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

# Train on jazz dataset
# ...

# Save LoRA adapter only (매우 작음!)
model.save_pretrained("./magenta-jazz-lora")
# → adapter_model.bin: ~5-10 MB (vs 800M base model!)
```

---

### 2. QLoRA로 대형 음악 모델 Fine-tuning

```python
"""
Lyria (Google DeepMind) 같은 대형 모델을 QLoRA로 fine-tuning
"""

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import torch

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load large music model with 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "google/lyria-large",  # 가정: 50B+ parameters
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Prepare for k-bit training
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)

# QLoRA config
qlora_config = LoraConfig(
    r=64,                    # QLoRA는 rank 높게 (성능 보상)
    lora_alpha=128,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply QLoRA
model = get_peft_model(model, qlora_config)
model.print_trainable_parameters()

# Train with jazz dataset (700h)
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./lyria-jazz-qlora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,  # Effective batch=32
    learning_rate=2e-4,              # QLoRA: lr 높게
    num_train_epochs=3,
    fp16=False,                      # QLoRA uses bf16
    bf16=True,
    optim="paged_adamw_8bit",        # Paged optimizer
    logging_steps=10,
    save_strategy="steps",
    save_steps=500,
    warmup_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=jazz_dataset,
    data_collator=data_collator
)

# Train!
trainer.train()

# Save QLoRA adapter
model.save_pretrained("./lyria-jazz-qlora-final")
# → Only adapter: ~50-100 MB
```

---

### 3. 여러 LoRA Adapter 교체 (Multi-task)

```python
"""
하나의 base model + 여러 LoRA adapter
- Bebop adapter
- Swing adapter
- Cool jazz adapter
"""

from peft import PeftModel

# Load base model
base_model = AutoModel.from_pretrained("google/magenta-realtime")

# Load bebop adapter
bebop_model = PeftModel.from_pretrained(base_model, "./lora-bebop")

# Generate bebop
bebop_output = bebop_model.generate(prompt="bebop solo")

# Switch to swing adapter
bebop_model.set_adapter("swing")  # Swap!
swing_output = bebop_model.generate(prompt="swing rhythm")

# Or load different adapter
cool_model = PeftModel.from_pretrained(base_model, "./lora-cool-jazz")
cool_output = cool_model.generate(prompt="cool jazz")

# Adapter 크기: 각각 5-10 MB
# Base model은 한 번만 로드!
```

---

## 📈 성능 최적화 팁

### 1. Gradient Checkpointing

```python
# 메모리 절약 (속도 약간 희생)
model.gradient_checkpointing_enable()

# Training args
training_args = TrainingArguments(
    gradient_checkpointing=True,
    ...
)
```

**효과**:
- 메모리: -40%
- 속도: -15%

---

### 2. Mixed Precision (BF16)

```python
# BF16 (권장, A100/H100)
training_args = TrainingArguments(
    bf16=True,
    ...
)

# FP16 (V100, RTX)
training_args = TrainingArguments(
    fp16=True,
    ...
)
```

**효과**:
- 속도: +50-100%
- 메모리: -50%

---

### 3. Flash Attention 2

```python
# Install
# pip install flash-attn --no-build-isolation

# Load model with flash attention
model = AutoModel.from_pretrained(
    "model-name",
    attn_implementation="flash_attention_2"
)
```

**효과**:
- 속도: +2-3x (long sequence)
- 메모리: -30-50%

---

### 4. Efficient Data Loading

```python
from datasets import load_dataset

# Streaming (큰 데이터셋)
dataset = load_dataset("path", streaming=True)

# Preprocessing
def preprocess(example):
    # Tokenize
    tokens = tokenizer(example["text"], truncation=True, max_length=2048)
    return tokens

dataset = dataset.map(
    preprocess,
    batched=True,
    num_proc=8,  # 병렬 처리
    remove_columns=dataset.column_names
)
```

---

## 🐛 자주 하는 실수 & 해결

### 1. OOM (Out of Memory)

**문제**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**해결**:
```python
# 1. Batch size 줄이기
per_device_train_batch_size = 1  # or 2

# 2. Gradient accumulation
gradient_accumulation_steps = 16  # Effective batch = 1×16 = 16

# 3. Gradient checkpointing
model.gradient_checkpointing_enable()

# 4. QLoRA 사용
quantization_config = BitsAndBytesConfig(load_in_4bit=True, ...)

# 5. Offload optimizer to CPU
optim="paged_adamw_8bit"
```

---

### 2. 학습이 안 됨 (Loss 안 떨어짐)

**문제**:
```
Loss: 5.234 → 5.231 → 5.229 ... (거의 변화 없음)
```

**원인 & 해결**:

**1) Learning rate 너무 낮음**
```python
# ❌ Full FT용 lr
learning_rate = 1e-5

# ✅ LoRA용 lr (10배 높게)
learning_rate = 1e-4
```

**2) LoRA가 적용 안 됨**
```python
# Check trainable parameters
model.print_trainable_parameters()
# → 0%면 문제!

# Target modules 확인
print(model)  # 실제 layer 이름 확인
```

**3) Rank 너무 낮음**
```python
# ❌ 복잡한 task에 r=4
r = 4

# ✅ r=16 or 32
r = 16
```

---

### 3. Inference가 느림

**문제**:
```python
# QLoRA로 학습 후 inference 느림
output = model.generate(...)  # 매우 느림
```

**해결**:

**Option 1: Merge LoRA weights**
```python
from peft import PeftModel

# Load base + LoRA
base_model = AutoModel.from_pretrained("base-model")
lora_model = PeftModel.from_pretrained(base_model, "lora-adapter")

# Merge!
merged_model = lora_model.merge_and_unload()

# Save merged model
merged_model.save_pretrained("./merged-model")
# → Inference 빠름, 하지만 FP16 (메모리 큼)
```

**Option 2: Keep 4-bit + LoRA**
```python
# 4-bit model + LoRA (작지만 느림)
# Trade-off: 메모리 vs 속도
```

---

## 🎓 추가 학습 자료

### 논문
1. **LoRA**: [arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)
2. **QLoRA**: [arxiv.org/abs/2305.14314](https://arxiv.org/abs/2305.14314)
3. **Prefix Tuning**: [arxiv.org/abs/2101.00190](https://arxiv.org/abs/2101.00190)
4. **Adapter Tuning**: [arxiv.org/abs/1902.00751](https://arxiv.org/abs/1902.00751)

### 코드 & 라이브러리
- **PEFT** (HuggingFace): [github.com/huggingface/peft](https://github.com/huggingface/peft)
- **bitsandbytes**: [github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- **Flash Attention**: [github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

### 튜토리얼
- [HuggingFace PEFT Docs](https://huggingface.co/docs/peft)
- [QLoRA Blog](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

---

## 🎯 2026 논문 제출용 Best Practice

### JazzFlow-RT Fine-tuning 전략

```python
"""
최종 추천: QLoRA (r=64, α=128)
- Magenta RealTime 800M params
- 재즈 데이터 700h
- GPU: 1× A100 (48GB)
"""

# Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

qlora_config = LoraConfig(
    r=64,                    # 높은 rank (재즈 복잡도)
    lora_alpha=128,          # 2×r
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"      # FFN
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="./jazzflow-rt-qlora",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=10,
    bf16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    logging_steps=10,
    save_strategy="steps",
    save_steps=1000,
    warmup_ratio=0.1
)

# 예상 결과:
# - 학습 시간: ~3-4일 (A100)
# - 메모리: ~35 GB
# - Adapter 크기: ~100 MB
# - 성능: FAD < 3.0 (목표)
```

---

**작성일**: 2025-11-16
**최종 수정**: 2025-11-16
**버전**: 1.0
