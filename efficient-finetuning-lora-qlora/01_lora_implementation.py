"""
LoRA (Low-Rank Adaptation) - Complete Implementation

📚 핵심 개념:
대규모 언어 모델(LLM)을 효율적으로 fine-tuning하는 방법

💡 핵심 아이디어:
- 기존 가중치는 freeze
- 낮은 rank의 행렬 2개(A, B)만 학습
- W' = W + BA (W는 frozen, B·A만 학습)

📊 장점:
- 학습 파라미터 수 90% 이상 감소
- 메모리 사용량 3배 감소
- 학습 속도 25% 향상
- 여러 LoRA adapter를 교체 가능

📄 논문:
- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- https://arxiv.org/abs/2106.09685

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ============================================================================
# LoRA Core Implementation
# ============================================================================

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) Layer

    수식:
        h = W_0·x + ΔW·x = W_0·x + (B·A)·x

    where:
        - W_0: frozen pre-trained weight (d × k)
        - A: trainable matrix (k × r)  ← 낮은 rank r
        - B: trainable matrix (d × r)
        - r << min(d, k)  ← 핵심!

    파라미터 감소:
        Original: d × k
        LoRA: d×r + r×k = r(d+k)
        Reduction ratio: r(d+k) / (d×k)

    예시 (d=4096, k=4096, r=8):
        Original: 16,777,216 params
        LoRA: 65,536 params (0.39% !)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Scaling factor
        self.scaling = alpha / rank

        # Dropout (optional)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming, B with zeros
        self.reset_parameters()

    def reset_parameters(self):
        """
        초기화 전략:
        - A: Kaiming uniform (학습 시작 시 분산 유지)
        - B: Zero initialization (처음엔 ΔW = 0, 즉 원본 모델과 동일)
        """
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: (batch, seq_len, in_features)

        Returns:
            lora_output: (batch, seq_len, out_features)
        """
        # x @ A @ B
        lora_output = self.dropout(x) @ self.lora_A @ self.lora_B
        lora_output = lora_output * self.scaling

        return lora_output


class LinearWithLoRA(nn.Module):
    """
    Linear layer with LoRA adaptation

    y = (W + BA)x = Wx + BAx
      = original_linear(x) + lora(x)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()

        # Original linear (frozen)
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA adapter
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)

        # Freeze original weights
        self.linear.weight.requires_grad = False
        if bias and self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, in_features)

        Returns:
            output: (batch, seq_len, out_features)
        """
        # Original output
        original_output = self.linear(x)

        # LoRA adaptation
        lora_output = self.lora(x)

        return original_output + lora_output

    def merge_weights(self):
        """
        LoRA 가중치를 원본에 병합 (inference 시 속도 향상)

        W_merged = W_0 + B @ A
        """
        if self.lora.rank > 0:
            # Compute ΔW = B @ A
            delta_w = (self.lora.lora_B @ self.lora.lora_A.T) * self.lora.scaling

            # Merge into original weights
            self.linear.weight.data += delta_w.T

            # Zero out LoRA (이미 병합됨)
            self.lora.lora_A.data.zero_()
            self.lora.lora_B.data.zero_()

    def unmerge_weights(self):
        """
        병합된 가중치 복원 (학습 재개 시)
        """
        # 주의: 원본 W_0를 저장해야 복원 가능
        # 일반적으로 merge는 inference용이므로 unmerge는 잘 안 씀
        raise NotImplementedError("Unmerge requires storing original weights")


# ============================================================================
# Attention Layer with LoRA
# ============================================================================

class MultiHeadAttentionWithLoRA(nn.Module):
    """
    Multi-Head Attention with LoRA on Q, K, V projections

    일반적으로 LoRA를 적용하는 위치:
    1. Q, K, V projection (가장 효과적)
    2. Output projection (선택적)
    3. FFN layers (선택적)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V with LoRA
        self.W_q = LinearWithLoRA(d_model, d_model, lora_rank, lora_alpha, dropout)
        self.W_k = LinearWithLoRA(d_model, d_model, lora_rank, lora_alpha, dropout)
        self.W_v = LinearWithLoRA(d_model, d_model, lora_rank, lora_alpha, dropout)

        # Output projection (선택적으로 LoRA 적용)
        self.W_o = LinearWithLoRA(d_model, d_model, lora_rank, lora_alpha, dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len, seq_len)

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size = x.size(0)

        # Q, K, V projections with LoRA
        Q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Output projection with LoRA
        output = self.W_o(attn_output)

        return output


# ============================================================================
# Transformer Block with LoRA
# ============================================================================

class TransformerBlockWithLoRA(nn.Module):
    """
    Transformer block with LoRA applied to:
    - Attention Q, K, V projections
    - Attention output projection
    - FFN layers (optional)
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        dropout: float = 0.1,
        apply_lora_to_ffn: bool = True  # FFN에도 LoRA 적용 여부
    ):
        super().__init__()

        # Multi-head attention with LoRA
        self.attention = MultiHeadAttentionWithLoRA(
            d_model, num_heads, lora_rank, lora_alpha, dropout
        )

        # Feed-forward network
        if apply_lora_to_ffn:
            # FFN with LoRA
            self.ffn = nn.Sequential(
                LinearWithLoRA(d_model, d_ff, lora_rank, lora_alpha, dropout),
                nn.GELU(),
                LinearWithLoRA(d_ff, d_model, lora_rank, lora_alpha, dropout)
            )
        else:
            # Original FFN (frozen)
            ffn_linear1 = nn.Linear(d_model, d_ff)
            ffn_linear2 = nn.Linear(d_ff, d_model)
            ffn_linear1.weight.requires_grad = False
            ffn_linear2.weight.requires_grad = False

            self.ffn = nn.Sequential(
                ffn_linear1,
                nn.GELU(),
                nn.Dropout(dropout),
                ffn_linear2
            )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len, seq_len)

        Returns:
            output: (batch, seq_len, d_model)
        """
        # Attention with residual
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # FFN with residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x


# ============================================================================
# Example: Apply LoRA to Pre-trained Model
# ============================================================================

def apply_lora_to_model(
    model: nn.Module,
    target_modules: list = ["q_proj", "k_proj", "v_proj"],
    rank: int = 8,
    alpha: int = 16
):
    """
    기존 모델에 LoRA를 적용하는 유틸리티 함수

    Args:
        model: Pre-trained model
        target_modules: LoRA를 적용할 모듈 이름 리스트
        rank: LoRA rank
        alpha: LoRA alpha

    Returns:
        model: LoRA가 적용된 모델
    """
    for name, module in model.named_modules():
        # Check if this module should have LoRA
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Get parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_submodule(parent_name) if parent_name else model

                # Create LoRA linear
                lora_linear = LinearWithLoRA(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    rank=rank,
                    alpha=alpha,
                    bias=module.bias is not None
                )

                # Copy original weights
                lora_linear.linear.weight.data = module.weight.data.clone()
                if module.bias is not None:
                    lora_linear.linear.bias.data = module.bias.data.clone()

                # Replace module
                setattr(parent, child_name, lora_linear)

                print(f"Applied LoRA to {name}")

    # Freeze all non-LoRA parameters
    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    return model


def count_parameters(model: nn.Module):
    """
    모델의 파라미터 수 계산

    Returns:
        total: 전체 파라미터 수
        trainable: 학습 가능한 파라미터 수
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return total, trainable


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("LoRA (Low-Rank Adaptation) - Complete Implementation")
    print("=" * 80)

    # ========== Example 1: Simple Linear with LoRA ==========
    print("\n[Example 1] Linear Layer with LoRA")
    print("-" * 80)

    d_in, d_out = 4096, 4096

    # Original linear
    original_linear = nn.Linear(d_in, d_out)
    original_params = sum(p.numel() for p in original_linear.parameters())

    # Linear with LoRA (rank=8)
    lora_linear = LinearWithLoRA(d_in, d_out, rank=8, alpha=16)
    total_params, trainable_params = count_parameters(lora_linear)

    print(f"Original Linear: {original_params:,} parameters")
    print(f"LoRA Linear: {total_params:,} total, {trainable_params:,} trainable")
    print(f"Reduction: {trainable_params / original_params * 100:.2f}%")
    print(f"Savings: {(1 - trainable_params / original_params) * 100:.1f}%")

    # Forward pass
    x = torch.randn(2, 128, d_in)
    output = lora_linear(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # ========== Example 2: Transformer Block with LoRA ==========
    print("\n[Example 2] Transformer Block with LoRA")
    print("-" * 80)

    # Without LoRA
    original_block = TransformerBlockWithLoRA(
        d_model=512,
        num_heads=8,
        d_ff=2048,
        lora_rank=0,  # No LoRA
        apply_lora_to_ffn=False
    )

    # With LoRA
    lora_block = TransformerBlockWithLoRA(
        d_model=512,
        num_heads=8,
        d_ff=2048,
        lora_rank=8,
        lora_alpha=16,
        apply_lora_to_ffn=True
    )

    original_total, original_trainable = count_parameters(original_block)
    lora_total, lora_trainable = count_parameters(lora_block)

    print(f"Original Block: {original_total:,} parameters")
    print(f"LoRA Block: {lora_total:,} total, {lora_trainable:,} trainable")
    print(f"Trainable ratio: {lora_trainable / lora_total * 100:.2f}%")

    # Forward pass
    x = torch.randn(2, 128, 512)
    output = lora_block(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # ========== Example 3: 실제 크기 모델 시뮬레이션 ==========
    print("\n[Example 3] Large Model Simulation (GPT-2 scale)")
    print("-" * 80)

    # GPT-2 medium: ~350M params
    # 24 layers, d_model=1024, d_ff=4096

    num_layers = 24
    d_model = 1024
    d_ff = 4096

    # Estimate parameters
    # Attention: 4 × (d_model × d_model) = 4 × 1024^2 per layer
    # FFN: 2 × (d_model × d_ff) = 2 × 1024 × 4096 per layer
    attn_params_per_layer = 4 * d_model * d_model
    ffn_params_per_layer = 2 * d_model * d_ff
    total_per_layer = attn_params_per_layer + ffn_params_per_layer

    total_original = num_layers * total_per_layer

    # With LoRA (rank=8)
    rank = 8
    # Attention: 4 × rank × (d_model + d_model)
    # FFN: 2 × rank × (d_model + d_ff)
    lora_attn_per_layer = 4 * rank * (d_model + d_model)
    lora_ffn_per_layer = 2 * rank * (d_model + d_ff)
    lora_trainable_per_layer = lora_attn_per_layer + lora_ffn_per_layer

    lora_trainable_total = num_layers * lora_trainable_per_layer

    print(f"Original model: {total_original / 1e6:.1f}M parameters")
    print(f"LoRA trainable: {lora_trainable_total / 1e6:.2f}M parameters")
    print(f"Reduction: {(1 - lora_trainable_total / total_original) * 100:.1f}%")
    print(f"Memory saving: ~{(1 - lora_trainable_total / total_original) * 100:.0f}%")

    print("\n" + "=" * 80)
    print("핵심 포인트:")
    print("1. LoRA는 대규모 모델의 fine-tuning 비용을 90% 이상 절감")
    print("2. Rank는 8-16이 일반적 (trade-off: 성능 vs 효율)")
    print("3. Alpha는 scaling factor (일반적으로 rank의 2배)")
    print("4. Attention Q, K, V에 적용이 가장 효과적")
    print("5. Inference 시 merge_weights()로 속도 손실 없음")
    print("=" * 80)
