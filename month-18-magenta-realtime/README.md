# Month 18: Magenta RealTime - 재즈 Fine-tuning

> **🎉 좋은 소식: Magenta RealTime은 지금 사용 가능합니다!**

## 📦 프로젝트 정보

- **GitHub**: https://github.com/magenta/magenta-realtime
- **공식 페이지**: https://magenta.withgoogle.com/magenta-realtime
- **공개일**: 2025년 6월 12일
- **라이선스**: Apache 2.0 (코드), CC BY 4.0 (모델)

## 🎯 핵심 사양

| 항목 | 사양 |
|------|------|
| **파라미터** | 800M |
| **오디오 품질** | 48kHz stereo |
| **생성 방식** | 2초 청크 스트리밍 |
| **컨텍스트** | 10초 과거 오디오 |
| **RTF** | 1.6x (A100 TPU) |
| **프롬프트** | 텍스트 + 오디오 |

## 🚀 빠른 시작

### 방법 1: Google Colab (추천! 무료 TPU)

```python
# 1. Colab 노트북에서 Runtime → TPU 선택

# 2. 설치
!git clone https://github.com/magenta/magenta-realtime.git
%cd magenta-realtime
!pip install -e .

# 3. 사용
from magenta_realtime import MagentaRT
model = MagentaRT.from_pretrained('google/magenta-realtime')

# 4. 생성!
audio = model.generate(
    prompt="bebop jazz piano solo",
    duration=30.0
)

# 5. 저장
import soundfile as sf
sf.write('jazz.wav', audio, 48000)
```

**소요 시간**: 5분
**비용**: 무료!

---

### 방법 2: 로컬 GPU (A100 권장)

#### 요구사항
- Python 3.12
- GPU: 40GB+ VRAM (A100, A6000)
- RAM: 32GB+
- CUDA 11.8+

#### 설치

```bash
# 1. Clone
git clone https://github.com/magenta/magenta-realtime.git
cd magenta-realtime

# 2. Python 환경
conda create -n magenta-rt python=3.12
conda activate magenta-rt

# 3. 설치
pip install -e .
pip install tf2jax==0.3.8

# 4. GPU 패치 (README 참고)
# t5x 패치 필요
```

---

### 방법 3: Docker (가장 쉬움)

```bash
# 빌드
docker build -t magenta-rt .

# 실행 (GPU 필수)
docker run --gpus all -it magenta-rt

# Python
from magenta_realtime import MagentaRT
model = MagentaRT.from_pretrained('google/magenta-realtime')
```

---

## 💻 파일 구성

```
month-18-magenta-realtime/
│
├── README.md (이 파일)
│
├── 01_finetune_jazz.py
│   - LoRA/QLoRA fine-tuning 템플릿
│   - 재즈 데이터셋 로더
│
└── 02_magenta_rt_실전가이드.py (NEW!)
    - 실제 Magenta RT 사용법
    - 텍스트/오디오 프롬프트
    - 실시간 스트리밍
    - 재즈 fine-tuning 가이드
```

---

## 🎺 재즈 Fine-tuning 전략

### QLoRA 설정 (권장)

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4-bit 양자화
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Magenta RT 로드
model = MagentaRT.from_pretrained(
    'google/magenta-realtime',
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA 적용
lora_config = LoraConfig(
    r=64,           # 재즈 복잡도 → 높은 rank
    lora_alpha=128, # 2×r
    target_modules=[
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj"
    ],
    lora_dropout=0.05
)

model = get_peft_model(model, lora_config)

# 학습!
# ... (TrainingArguments, Trainer)
```

### 예상 결과

| 항목 | 값 |
|------|-----|
| **메모리 사용** | ~4 GB (vs 16 GB) |
| **학습 시간** | ~3-4일 (A100, 700h 데이터) |
| **Adapter 크기** | ~100 MB |
| **성능 목표** | FAD < 3.0 |

---

## 📊 사용 예시

### 1. 텍스트 프롬프트

```python
from magenta_realtime import MagentaRT

model = MagentaRT.from_pretrained('google/magenta-realtime')

# 생성
audio = model.generate(
    prompt="fast bebop jazz piano with complex chord progressions",
    duration=30.0,
    temperature=1.2  # 창의성
)

# 저장
import soundfile as sf
sf.write('bebop.wav', audio, 48000)
```

---

### 2. 오디오 프롬프트 (스타일 변환)

```python
import librosa

# 참고 오디오 로드
reference, _ = librosa.load('reference_jazz.wav', sr=48000)

# 스타일 변환
audio = model.generate(
    audio_prompt=reference,
    duration=60.0,
    style_blend_weight=0.8  # 0=무시, 1=완전 모방
)
```

---

### 3. 실시간 스트리밍

```python
def on_chunk(chunk):
    """2초 청크마다 호출"""
    # 실시간 재생
    play_audio(chunk)
    # 또는 저장
    save_to_buffer(chunk)

# 60초 실시간 생성
model.generate_stream(
    prompt="smooth cool jazz",
    duration=60.0,
    callback=on_chunk
)
```

---

## 🎯 학습 체크리스트

### Week 1: 설치 & 테스트
- [ ] Colab에서 Magenta RT 실행
- [ ] 텍스트 프롬프트로 음악 생성
- [ ] 오디오 프롬프트 실험
- [ ] 실시간 스트리밍 테스트

### Week 2: 데이터 준비
- [ ] 재즈 오디오 50시간 수집 (작게 시작)
- [ ] 48kHz 스테레오로 변환
- [ ] Train/Val split (90/10)
- [ ] 데이터 증강 (pitch shift, tempo)

### Week 3-4: Fine-tuning
- [ ] QLoRA 설정
- [ ] 학습 시작 (3-4일)
- [ ] Checkpoints 저장
- [ ] 생성 샘플 평가

### Week 5: 평가
- [ ] FAD score 계산
- [ ] Jazz theory compliance 측정
- [ ] 재즈 뮤지션 피드백
- [ ] Ablation study

---

## 🐛 트러블슈팅

### OOM (Out of Memory)

```python
# 해결 1: Batch size 줄이기
per_device_train_batch_size = 1

# 해결 2: Gradient checkpointing
model.gradient_checkpointing_enable()

# 해결 3: QLoRA 사용 (4-bit)
load_in_4bit = True

# 해결 4: Colab TPU 사용 (무료!)
```

### 생성 속도 느림

```python
# 해결 1: TPU 사용 (RTF 1.6x)
# Colab에서 Runtime → TPU

# 해결 2: Batch inference
# 여러 샘플 동시 생성

# 해결 3: ONNX 변환
# (고급, 나중에)
```

---

## 📚 추가 자료

### 공식 문서
- [GitHub Repository](https://github.com/magenta/magenta-realtime)
- [Colab Demo](https://colab.research.google.com/...) (README에서 확인)
- [Model Card](https://huggingface.co/google/magenta-realtime)

### 논문
- Magenta RealTime 기술 보고서 (2025)
- SpectroStream: High-Fidelity Audio Streaming
- MusicCoCa: Text-Audio Style Blending

### 커뮤니티
- Magenta Google Group
- r/MachineLearning
- Discord (Magenta 채널)

---

## 🎉 결론: 지금 바로 시작 가능!

```bash
# 오늘 안에 할 수 있는 것:

# 1. Colab에서 실행 (10분)
https://colab.research.google.com/...

# 2. 재즈 음악 생성 (5분)
audio = model.generate(prompt="bebop jazz", duration=30)

# 3. 스타일 실험 (30분)
# 다양한 프롬프트 테스트

# → 오늘 안에 음악 생성 경험!
```

**Magenta RealTime은 지금 사용 가능합니다. 바로 시작하세요!** 🚀🎺

---

**작성일**: 2025-11-16
**최종 수정**: 2025-11-16 (Magenta RT 공개 확인!)
**버전**: 2.0
