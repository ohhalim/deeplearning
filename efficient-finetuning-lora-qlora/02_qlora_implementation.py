"""
QLoRA (Quantized Low-Rank Adaptation) - Complete Implementation

📚 핵심 개념:
LoRA + 4-bit Quantization = 메모리 사용량 극도로 절감

💡 핵심 아이디어:
1. Base model을 4-bit로 양자화 (16GB → 4GB)
2. LoRA adapter만 FP16/BF16으로 학습
3. 65B 모델을 단일 GPU(48GB)에서 fine-tuning 가능!

🔥 혁신 포인트:
- 4-bit NormalFloat (NF4) quantization
- Double quantization
- Paged optimizers

📊 성능:
- 메모리: LoRA 대비 1/4
- 품질: Full fine-tuning과 거의 동일
- 속도: LoRA와 비슷

📄 논문:
- "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- https://arxiv.org/abs/2305.14314

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
import math


# ============================================================================
# 4-bit Quantization (NF4 - NormalFloat4)
# ============================================================================

class NF4Quantizer:
    """
    4-bit NormalFloat (NF4) Quantization

    핵심 아이디어:
    - 일반 신경망 가중치는 정규분포를 따름
    - 정규분포에 최적화된 4-bit 양자화
    - 균일 분포보다 더 나은 정보 보존

    NF4 값:
        [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848,
         -0.0911,  0.0,  0.0911,  0.1848,  0.2844,  0.3949,
          0.5251,  0.6962,  1.0, inf]
    """

    # NF4 lookup table (16개 값)
    NF4_VALUES = torch.tensor([
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0
    ])

    def __init__(self, blocksize: int = 64):
        """
        Args:
            blocksize: Quantization block size (일반적으로 64 또는 128)
        """
        self.blocksize = blocksize

    def quantize(self, weight: torch.Tensor) -> tuple:
        """
        4-bit NF4 quantization

        Args:
            weight: (out_features, in_features) FP16/BF16 weight

        Returns:
            quantized: (out_features, in_features) uint8 (4-bit packed)
            scale: (num_blocks,) FP16 scale factors
        """
        # Flatten weight
        original_shape = weight.shape
        weight_flat = weight.flatten()

        # Number of blocks
        num_elements = weight_flat.numel()
        num_blocks = (num_elements + self.blocksize - 1) // self.blocksize

        # Pad to multiple of blocksize
        pad_size = num_blocks * self.blocksize - num_elements
        if pad_size > 0:
            weight_flat = F.pad(weight_flat, (0, pad_size))

        # Reshape to blocks
        weight_blocks = weight_flat.view(num_blocks, self.blocksize)

        # Compute scale per block (absmax)
        scale = weight_blocks.abs().max(dim=1).values

        # Avoid division by zero
        scale = scale.clamp(min=1e-8)

        # Normalize to [-1, 1]
        normalized = weight_blocks / scale.unsqueeze(1)

        # Quantize to NF4
        nf4_values = self.NF4_VALUES.to(weight.device)
        quantized_indices = torch.zeros_like(normalized, dtype=torch.uint8)

        for i, val in enumerate(nf4_values[:-1]):
            next_val = nf4_values[i + 1]
            threshold = (val + next_val) / 2
            mask = (normalized >= threshold)
            quantized_indices[mask] = i + 1

        # Pack 2 indices into 1 uint8 (4-bit each)
        # TODO: Actual bit-packing for memory efficiency
        # For now, store as uint8 (will use 8 bits instead of 4)

        return quantized_indices.view(original_shape), scale

    def dequantize(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        original_shape: tuple
    ) -> torch.Tensor:
        """
        Dequantize from 4-bit NF4

        Args:
            quantized: uint8 quantized indices
            scale: FP16 scale factors
            original_shape: Original weight shape

        Returns:
            weight: (out_features, in_features) FP16/BF16 weight
        """
        # Flatten
        quantized_flat = quantized.flatten()

        # Reshape to blocks
        num_blocks = len(scale)
        quantized_blocks = quantized_flat[:num_blocks * self.blocksize].view(
            num_blocks, self.blocksize
        )

        # Look up NF4 values
        nf4_values = self.NF4_VALUES.to(quantized.device)
        dequantized = nf4_values[quantized_blocks]

        # Scale back
        dequantized = dequantized * scale.unsqueeze(1)

        # Reshape
        weight = dequantized.flatten()[:np.prod(original_shape)].view(original_shape)

        return weight


# ============================================================================
# Double Quantization
# ============================================================================

class DoubleQuantizer:
    """
    Double Quantization

    핵심 아이디어:
    - Scale factors도 양자화!
    - 추가 메모리 절약 (0.37 bits/param)

    과정:
    1. Weight → 4-bit (NF4) + FP16 scale
    2. Scale → 8-bit + FP16 scale_scale
    """

    def __init__(self, blocksize: int = 64, scale_blocksize: int = 256):
        self.nf4_quantizer = NF4Quantizer(blocksize)
        self.scale_blocksize = scale_blocksize

    def quantize_scale(self, scale: torch.Tensor) -> tuple:
        """
        Quantize scale factors to 8-bit

        Args:
            scale: (num_blocks,) FP16

        Returns:
            scale_quantized: (num_blocks,) uint8
            scale_scale: (num_scale_blocks,) FP16
        """
        num_blocks = len(scale)
        num_scale_blocks = (num_blocks + self.scale_blocksize - 1) // self.scale_blocksize

        # Pad
        pad_size = num_scale_blocks * self.scale_blocksize - num_blocks
        if pad_size > 0:
            scale = F.pad(scale, (0, pad_size))

        # Reshape to scale blocks
        scale_blocks = scale.view(num_scale_blocks, self.scale_blocksize)

        # Compute scale of scale
        scale_scale = scale_blocks.abs().max(dim=1).values.clamp(min=1e-8)

        # Normalize to [0, 1]
        normalized = scale_blocks / scale_scale.unsqueeze(1)

        # Quantize to 8-bit (0-255)
        scale_quantized = (normalized * 255).round().clamp(0, 255).to(torch.uint8)

        return scale_quantized.flatten()[:num_blocks], scale_scale

    def dequantize_scale(
        self,
        scale_quantized: torch.Tensor,
        scale_scale: torch.Tensor,
        num_blocks: int
    ) -> torch.Tensor:
        """
        Dequantize scale factors

        Args:
            scale_quantized: (num_blocks,) uint8
            scale_scale: (num_scale_blocks,) FP16
            num_blocks: Original number of blocks

        Returns:
            scale: (num_blocks,) FP16
        """
        num_scale_blocks = len(scale_scale)

        # Pad
        pad_size = num_scale_blocks * self.scale_blocksize - num_blocks
        if pad_size > 0:
            scale_quantized = F.pad(scale_quantized.float(), (0, pad_size))

        # Reshape
        scale_blocks = scale_quantized.view(num_scale_blocks, self.scale_blocksize)

        # Denormalize
        scale = (scale_blocks.float() / 255.0) * scale_scale.unsqueeze(1)

        return scale.flatten()[:num_blocks]


# ============================================================================
# QLoRA Linear Layer
# ============================================================================

class QLoRALinear(nn.Module):
    """
    QLoRA Linear Layer

    구조:
        y = Quantized_4bit(W_0) @ x + (B @ A) @ x

    메모리:
        - W_0: 4-bit (+ 16-bit scale)
        - A, B: 16-bit (trainable)

    메모리 절감 예시 (d=4096, k=4096, r=8):
        Original FP16: 4096×4096×2 bytes = 32MB
        QLoRA:
            - W_0 (4-bit): 4096×4096×0.5 = 8MB
            - Scale: ~256KB
            - LoRA (A+B): 8×(4096+4096)×2 = 128KB
            Total: ~8.4MB (26% of original!)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        bias: bool = True,
        quantize_base: bool = True
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.quantize_base = quantize_base

        # Base weight (will be quantized)
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02,
            requires_grad=False  # Frozen
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        # Quantizer
        self.quantizer = NF4Quantizer(blocksize=64)

        # Quantized weights (will be populated in quantize())
        self.register_buffer('quantized_weight', None)
        self.register_buffer('weight_scale', None)

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize LoRA
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def quantize_weights(self):
        """
        Quantize base weights to 4-bit NF4
        """
        if self.quantize_base:
            quantized, scale = self.quantizer.quantize(self.base_weight.data)
            self.quantized_weight = quantized
            self.weight_scale = scale

            # Free original weight to save memory
            # self.base_weight.data = torch.tensor([])  # Optional

            print(f"Quantized {self.out_features}×{self.in_features} to 4-bit")

    def dequantize_weights(self) -> torch.Tensor:
        """
        Dequantize for forward pass

        Returns:
            weight: FP16 weight
        """
        if self.quantized_weight is not None:
            return self.quantizer.dequantize(
                self.quantized_weight,
                self.weight_scale,
                (self.out_features, self.in_features)
            )
        else:
            return self.base_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch, seq_len, in_features)

        Returns:
            output: (batch, seq_len, out_features)
        """
        # Dequantize base weight
        weight = self.dequantize_weights()

        # Base output (frozen)
        output = F.linear(x, weight, self.bias)

        # LoRA output (trainable)
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B
        lora_output = lora_output * self.scaling

        return output + lora_output


# ============================================================================
# Paged Optimizer (simulated)
# ============================================================================

class PagedAdamW:
    """
    Paged AdamW Optimizer

    핵심 아이디어:
    - Optimizer states를 CPU와 GPU 사이에서 paging
    - OOM 방지
    - bitsandbytes 라이브러리 사용 (실제 구현)

    Note: 이것은 simplified version
    실제로는 bitsandbytes.optim.PagedAdamW 사용
    """

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps

        # States (simplified)
        self.state = {}

        for param in self.params:
            self.state[param] = {
                'step': 0,
                'exp_avg': torch.zeros_like(param),
                'exp_avg_sq': torch.zeros_like(param)
            }

    def step(self):
        """
        Optimizer step

        실제 구현에서는:
        - CPU로 offload된 states를 GPU로 가져옴
        - Update
        - 다시 CPU로 보냄
        """
        for param in self.params:
            if param.grad is None:
                continue

            grad = param.grad
            state = self.state[param]

            state['step'] += 1

            # Update biased first/second moment estimates
            state['exp_avg'].mul_(self.betas[0]).add_(grad, alpha=1 - self.betas[0])
            state['exp_avg_sq'].mul_(self.betas[1]).addcmul_(
                grad, grad, value=1 - self.betas[1]
            )

            # Bias correction
            bias_correction1 = 1 - self.betas[0] ** state['step']
            bias_correction2 = 1 - self.betas[1] ** state['step']

            step_size = self.lr / bias_correction1

            denom = (state['exp_avg_sq'].sqrt() / math.sqrt(bias_correction2)).add_(self.eps)

            # Update parameters
            param.data.addcdiv_(state['exp_avg'], denom, value=-step_size)

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("QLoRA (Quantized Low-Rank Adaptation) - Implementation")
    print("=" * 80)

    # ========== Example 1: NF4 Quantization ==========
    print("\n[Example 1] 4-bit NF4 Quantization")
    print("-" * 80)

    # Create weight matrix
    weight = torch.randn(4096, 4096, dtype=torch.float16)
    original_size = weight.element_size() * weight.numel()

    print(f"Original weight: {weight.shape}")
    print(f"Original size: {original_size / (1024**2):.2f} MB (FP16)")

    # Quantize
    quantizer = NF4Quantizer(blocksize=64)
    quantized, scale = quantizer.quantize(weight)

    quantized_size = quantized.element_size() * quantized.numel()
    scale_size = scale.element_size() * scale.numel()
    total_size = quantized_size + scale_size

    print(f"\nQuantized: {quantized.shape}, dtype: {quantized.dtype}")
    print(f"Scale: {scale.shape}")
    print(f"Quantized size: {quantized_size / (1024**2):.2f} MB")
    print(f"Scale size: {scale_size / 1024:.2f} KB")
    print(f"Total: {total_size / (1024**2):.2f} MB")
    print(f"Compression ratio: {original_size / total_size:.2f}x")

    # Dequantize
    dequantized = quantizer.dequantize(quantized, scale, weight.shape)
    error = (weight - dequantized).abs().mean()
    print(f"\nDequantization error (MAE): {error:.6f}")

    # ========== Example 2: QLoRA Linear ==========
    print("\n[Example 2] QLoRA Linear Layer")
    print("-" * 80)

    # Create QLoRA linear
    qlora_linear = QLoRALinear(
        in_features=4096,
        out_features=4096,
        rank=8,
        alpha=16,
        quantize_base=True
    )

    # Quantize base weights
    qlora_linear.quantize_weights()

    # Count parameters
    base_params = qlora_linear.base_weight.numel()
    lora_params = qlora_linear.lora_A.numel() + qlora_linear.lora_B.numel()
    total_params = base_params + lora_params

    print(f"Base params: {base_params:,} (frozen, 4-bit)")
    print(f"LoRA params: {lora_params:,} (trainable, FP16)")
    print(f"Trainable ratio: {lora_params / total_params * 100:.3f}%")

    # Memory estimation
    base_memory = (base_params * 0.5 + len(qlora_linear.weight_scale) * 2) / (1024**2)
    lora_memory = lora_params * 2 / (1024**2)
    print(f"\nMemory usage:")
    print(f"  Base (4-bit): {base_memory:.2f} MB")
    print(f"  LoRA (FP16): {lora_memory:.2f} MB")
    print(f"  Total: {base_memory + lora_memory:.2f} MB")

    # Forward pass
    x = torch.randn(2, 128, 4096)
    output = qlora_linear(x)
    print(f"\nForward pass:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")

    # ========== Example 3: Full Model Comparison ==========
    print("\n[Example 3] Memory Comparison (LLaMA-65B scale)")
    print("-" * 80)

    # LLaMA-65B specs
    num_layers = 80
    d_model = 8192
    d_ff = 22016

    # Attention params per layer: 4 × (8192 × 8192)
    # FFN params per layer: 2 × (8192 × 22016)
    attn_params = 4 * d_model * d_model
    ffn_params = d_model * d_ff + d_ff * d_model

    total_params = num_layers * (attn_params + ffn_params)

    # Full FP16
    fp16_memory = total_params * 2 / (1024**3)

    # LoRA (rank=8)
    lora_base_memory = total_params * 2 / (1024**3)  # FP16 frozen
    lora_trainable = num_layers * (
        4 * 8 * (d_model + d_model) +  # Attention
        2 * 8 * (d_model + d_ff)       # FFN
    )
    lora_trainable_memory = lora_trainable * 2 / (1024**3)

    # QLoRA (rank=8, 4-bit base)
    qlora_base_memory = total_params * 0.5 / (1024**3)  # 4-bit
    qlora_trainable_memory = lora_trainable * 2 / (1024**3)  # Same as LoRA

    print(f"LLaMA-65B: {total_params / 1e9:.1f}B parameters\n")
    print(f"Full Fine-tuning (FP16):")
    print(f"  Memory: {fp16_memory:.1f} GB")
    print(f"  Trainable: {total_params / 1e9:.1f}B params\n")
    print(f"LoRA (rank=8):")
    print(f"  Memory: {lora_base_memory + lora_trainable_memory:.1f} GB")
    print(f"  Trainable: {lora_trainable / 1e6:.1f}M params")
    print(f"  Reduction: {(1 - lora_trainable / total_params) * 100:.1f}%\n")
    print(f"QLoRA (rank=8, 4-bit):")
    print(f"  Memory: {qlora_base_memory + qlora_trainable_memory:.1f} GB")
    print(f"  Trainable: {lora_trainable / 1e6:.1f}M params")
    print(f"  Memory reduction vs Full: {(1 - (qlora_base_memory + qlora_trainable_memory) / fp16_memory) * 100:.1f}%")
    print(f"  Memory reduction vs LoRA: {(1 - (qlora_base_memory + qlora_trainable_memory) / (lora_base_memory + lora_trainable_memory)) * 100:.1f}%")

    print("\n" + "=" * 80)
    print("핵심 포인트:")
    print("1. QLoRA = LoRA + 4-bit quantization")
    print("2. 65B 모델을 48GB GPU 하나로 fine-tuning 가능!")
    print("3. NF4 (NormalFloat4)는 정규분포에 최적화된 4-bit")
    print("4. Double quantization으로 추가 메모리 절약")
    print("5. 성능은 full fine-tuning과 거의 동일")
    print("=" * 80)

    print("\n실제 사용 (HuggingFace + bitsandbytes):")
    print("=" * 80)
    print("""
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model with 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto"
)

# LoRA config
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Train!
# ...
""")
    print("=" * 80)
