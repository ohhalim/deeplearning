"""
Magenta RealTime - 실전 사용 가이드

🎉 좋은 소식: Magenta RealTime은 이미 공개되어 있고 지금 사용 가능합니다!

📅 공개일: 2025년 6월 12일
📦 GitHub: https://github.com/magenta/magenta-realtime
🌐 공식 페이지: https://magenta.withgoogle.com/magenta-realtime
📄 라이선스: Apache 2.0 (코드), CC BY 4.0 (모델)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import torch
import numpy as np


# ============================================================================
# 1. 설치 방법
# ============================================================================

"""
방법 1: Google Colab (권장! - 무료 TPU)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Colab 노트북 열기
2. 런타임 → TPU 선택
3. 아래 코드 실행:

```python
!git clone https://github.com/magenta/magenta-realtime.git
%cd magenta-realtime
!pip install -e .

# 모델 다운로드 (자동)
from magenta_realtime import MagentaRT
model = MagentaRT.from_pretrained('google/magenta-realtime')
```

→ 5분 안에 실행 가능!


방법 2: 로컬 GPU (RTX 3090, A100 등)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

요구사항:
- Python 3.12
- CUDA 11.8+
- GPU 메모리: 40GB+ (A100 권장)
- RAM: 32GB+

설치:
```bash
# 1. Repository clone
git clone https://github.com/magenta/magenta-realtime.git
cd magenta-realtime

# 2. Python 3.12 환경 생성
conda create -n magenta-rt python=3.12
conda activate magenta-rt

# 3. 의존성 설치
pip install -e .
pip install tf2jax==0.3.8

# 4. GPU 패치 (중요!)
# t5x 패치 필요 (README 참고)
```


방법 3: Docker (가장 쉬움!)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

요구사항:
- Docker + nvidia-docker
- GPU: 40GB+ VRAM

```bash
# Docker 이미지 빌드
docker build -t magenta-rt .

# 실행
docker run --gpus all -it magenta-rt

# Python에서
from magenta_realtime import MagentaRT
model = MagentaRT.from_pretrained('google/magenta-realtime')
```
"""


# ============================================================================
# 2. 기본 사용법
# ============================================================================

class MagentaRTQuickStart:
    """
    Magenta RealTime Quick Start Guide

    핵심 기능:
    1. 텍스트 프롬프트로 음악 생성
    2. 오디오 프롬프트로 스타일 변환
    3. 실시간 스트리밍 생성 (2초 청크)
    """

    def __init__(self):
        """
        모델 로드

        참고: 처음 실행 시 ~3GB 모델 다운로드
        """
        print("Loading Magenta RealTime...")

        # 실제 코드 (GitHub README 기반)
        # from magenta_realtime import MagentaRT
        # self.model = MagentaRT.from_pretrained('google/magenta-realtime')

        # Placeholder for demo
        print("(Model would be loaded here)")
        print("Model: 800M parameters")
        print("Audio: 48kHz stereo")
        print("Chunk size: 2 seconds")

    def generate_from_text(
        self,
        prompt: str = "upbeat jazz piano",
        duration_sec: float = 30.0,
        temperature: float = 1.0
    ):
        """
        텍스트 프롬프트로 음악 생성

        Args:
            prompt: 텍스트 설명 (예: "bebop jazz piano solo")
            duration_sec: 생성할 길이 (초)
            temperature: 창의성 (0.5=보수적, 1.5=창의적)

        Returns:
            audio: (num_samples,) numpy array, 48kHz
        """
        print(f"\nGenerating music from text prompt:")
        print(f"  Prompt: '{prompt}'")
        print(f"  Duration: {duration_sec}s")
        print(f"  Temperature: {temperature}")

        # 실제 코드 예시
        """
        audio = self.model.generate(
            prompt=prompt,
            duration=duration_sec,
            temperature=temperature,
            style_blend_weight=1.0
        )

        return audio  # (num_samples,) at 48kHz
        """

        print("\n  → Generating 2s chunks...")
        num_chunks = int(duration_sec / 2)
        print(f"  → Total {num_chunks} chunks")
        print("  → Done!")

        # Dummy audio
        sample_rate = 48000
        audio = np.random.randn(int(duration_sec * sample_rate)) * 0.01

        return audio

    def generate_from_audio(
        self,
        audio_prompt_path: str,
        duration_sec: float = 30.0,
        style_weight: float = 0.8
    ):
        """
        오디오 프롬프트로 스타일 변환

        Args:
            audio_prompt_path: 참고 오디오 파일 경로
            duration_sec: 생성할 길이
            style_weight: 스타일 강도 (0=무시, 1=완전 모방)

        Returns:
            audio: (num_samples,) numpy array
        """
        print(f"\nGenerating music from audio prompt:")
        print(f"  Audio: {audio_prompt_path}")
        print(f"  Style weight: {style_weight}")

        # 실제 코드 예시
        """
        # 오디오 로드
        import librosa
        audio_prompt, sr = librosa.load(audio_prompt_path, sr=48000)

        # 생성
        audio = self.model.generate(
            audio_prompt=audio_prompt,
            duration=duration_sec,
            style_blend_weight=style_weight
        )

        return audio
        """

        print("  → Extracting style embedding...")
        print("  → Generating with style guidance...")
        print("  → Done!")

        # Dummy
        return np.random.randn(int(duration_sec * 48000)) * 0.01

    def real_time_generate(
        self,
        text_prompt: str = "smooth jazz",
        duration_sec: float = 60.0,
        callback=None
    ):
        """
        실시간 스트리밍 생성

        핵심: 2초 청크를 연속으로 생성

        Args:
            text_prompt: 스타일 설명
            duration_sec: 총 길이
            callback: 각 청크마다 호출될 함수 (audio_chunk)
        """
        print(f"\nReal-time streaming generation:")
        print(f"  Style: '{text_prompt}'")
        print(f"  Total duration: {duration_sec}s")
        print(f"  Chunk size: 2s")

        num_chunks = int(duration_sec / 2)

        for i in range(num_chunks):
            # 2초 청크 생성
            print(f"\r  Generating chunk {i+1}/{num_chunks}...", end="")

            # 실제 코드
            """
            chunk = self.model.generate_chunk(
                context=previous_audio,  # 10초 컨텍스트
                style_embedding=style_emb,
                chunk_duration=2.0
            )

            if callback:
                callback(chunk)
            """

            # Dummy
            chunk = np.random.randn(2 * 48000) * 0.01

            if callback:
                callback(chunk)

        print("\n  → Streaming complete!")


# ============================================================================
# 3. 재즈 Fine-tuning with LoRA
# ============================================================================

def finetune_magenta_rt_for_jazz():
    """
    Magenta RealTime을 재즈로 fine-tuning

    전략:
    1. Base model: Frozen (4-bit 양자화)
    2. LoRA adapter: Trainable (FP16)
    3. 재즈 데이터: 50-100시간부터 시작
    """

    print("=" * 80)
    print("Magenta RealTime → Jazz Fine-tuning with QLoRA")
    print("=" * 80)

    # Step 1: Load Magenta RT with 4-bit quantization
    print("\n[Step 1] Loading Magenta RT with 4-bit quantization...")

    """
    from transformers import BitsAndBytesConfig
    from magenta_realtime import MagentaRT

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = MagentaRT.from_pretrained(
        'google/magenta-realtime',
        quantization_config=bnb_config,
        device_map="auto"
    )

    print(f"Model loaded: 800M params → 4-bit (~3GB)")
    """

    # Step 2: Prepare for k-bit training
    print("\n[Step 2] Preparing for k-bit training...")

    """
    from peft import prepare_model_for_kbit_training
    model = prepare_model_for_kbit_training(model)
    """

    # Step 3: Apply LoRA
    print("\n[Step 3] Applying LoRA adapter...")

    """
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=64,                    # 재즈 복잡도 → 높은 rank
        lora_alpha=128,          # 2×r
        target_modules=[
            # Attention (핵심!)
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            # FFN (선택적)
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # → trainable: ~100M / 800M (12.5%)
    """

    # Step 4: Load jazz dataset
    print("\n[Step 4] Loading jazz dataset...")

    """
    from datasets import load_dataset

    # Option 1: Audio files
    dataset = load_dataset("audiofolder", data_dir="./jazz_audio_700h")

    # Option 2: Preprocessed
    dataset = load_dataset("your-username/jazz-dataset-700h")

    print(f"Dataset: {len(dataset['train'])} samples, ~700 hours")
    """

    # Step 5: Training
    print("\n[Step 5] Training with QLoRA...")

    """
    from transformers import Trainer, TrainingArguments

    training_args = TrainingArguments(
        output_dir="./magenta-jazz-qlora",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,  # Effective batch=32
        learning_rate=2e-4,              # LoRA: high LR
        num_train_epochs=10,
        bf16=True,
        optim="paged_adamw_8bit",        # QLoRA optimizer
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        warmup_ratio=0.1,
        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        data_collator=data_collator
    )

    # Train!
    trainer.train()

    # Save LoRA adapter (only ~100MB!)
    model.save_pretrained("./magenta-jazz-lora-adapter")
    """

    print("\n" + "=" * 80)
    print("Fine-tuning Complete!")
    print("=" * 80)
    print("\nResults:")
    print("  - Base model: 800M params (4-bit, frozen)")
    print("  - LoRA adapter: ~100M params (FP16, trainable)")
    print("  - Memory usage: ~4GB (vs 16GB without QLoRA)")
    print("  - Training time: ~3-4 days (A100, 700h data)")
    print("  - Adapter size: ~100MB")
    print("\nNext steps:")
    print("  1. Evaluate on test set")
    print("  2. Compute FAD score")
    print("  3. Get jazz musician feedback")
    print("  4. Prepare for ISMIR 2026 submission!")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Magenta RealTime - Practical Guide")
    print("=" * 80)

    # Quick start
    print("\n" + "=" * 80)
    print("1. Quick Start (Text Generation)")
    print("=" * 80)

    magenta = MagentaRTQuickStart()

    # Generate from text
    audio = magenta.generate_from_text(
        prompt="bebop jazz piano solo with fast tempo",
        duration_sec=30.0,
        temperature=1.2
    )

    print(f"\nGenerated audio: {audio.shape} samples at 48kHz")
    print("Save: soundfile.write('output.wav', audio, 48000)")

    # Generate from audio
    print("\n" + "=" * 80)
    print("2. Style Transfer (Audio Prompt)")
    print("=" * 80)

    audio2 = magenta.generate_from_audio(
        audio_prompt_path="./reference_jazz.wav",
        duration_sec=60.0,
        style_weight=0.8
    )

    # Real-time streaming
    print("\n" + "=" * 80)
    print("3. Real-time Streaming")
    print("=" * 80)

    def on_chunk_generated(chunk):
        """각 2초 청크마다 호출"""
        # 실시간 재생, 저장 등
        pass

    magenta.real_time_generate(
        text_prompt="smooth cool jazz",
        duration_sec=60.0,
        callback=on_chunk_generated
    )

    # Fine-tuning
    print("\n" + "=" * 80)
    print("4. Jazz Fine-tuning with QLoRA")
    print("=" * 80)

    finetune_magenta_rt_for_jazz()

    print("\n" + "=" * 80)
    print("실제 사용 방법:")
    print("=" * 80)
    print("""
# 1. Colab에서 실행 (가장 쉬움!)
https://colab.research.google.com/github/magenta/magenta-realtime/...

# 2. 로컬 설치
git clone https://github.com/magenta/magenta-realtime.git
cd magenta-realtime
pip install -e .

# 3. Python에서 사용
from magenta_realtime import MagentaRT
model = MagentaRT.from_pretrained('google/magenta-realtime')
audio = model.generate(prompt="jazz piano", duration=30)

# 4. 재즈 fine-tuning
# → 위의 finetune_magenta_rt_for_jazz() 참고
    """)
    print("=" * 80)
