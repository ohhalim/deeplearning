"""
Month 2: Music Transformer - Basic Transformer Implementation

목표:
1. Vanilla Transformer from scratch 구현
2. Multi-head attention 이해
3. Positional encoding 구현
4. 음악 시퀀스 생성

참고 논문:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Music Transformer" (Huang et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PositionalEncoding(nn.Module):
    """
    위치 인코딩 - 시퀀스의 순서 정보를 인코딩

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 위치 인코딩 계산
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-Attention

    핵심 아이디어:
    - Query, Key, Value를 여러 head로 분할
    - 각 head가 다른 representation subspace 학습
    - 병렬로 attention 계산 후 concat
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V projection layers
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

        Args:
            Q, K, V: (batch_size, num_heads, seq_len, d_k)
            mask: (batch_size, 1, 1, seq_len) or (batch_size, 1, seq_len, seq_len)

        Returns:
            output: (batch_size, num_heads, seq_len, d_k)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # QK^T
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Masking (for autoregressive generation)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Weighted sum of values
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query, key, value: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, seq_len, seq_len)

        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)

        # Linear projection and split into num_heads
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention
        attn_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concat heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # Final projection
        output = self.W_o(attn_output)

        return output, attention_weights


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network

    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    각 위치에 독립적으로 적용되는 2-layer MLP
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer

    구조:
    1. Masked Multi-Head Attention (self-attention)
    2. Add & Norm
    3. Position-wise Feed-Forward
    4. Add & Norm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, seq_len, seq_len)
        """
        # Self-attention with residual connection
        attn_output, attention_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x, attention_weights


class MusicTransformer(nn.Module):
    """
    Music Transformer (simplified version)

    음악 시퀀스 생성을 위한 Transformer 기반 모델

    Args:
        vocab_size: MIDI 이벤트 vocabulary 크기 (예: 388)
        d_model: 모델 차원 (예: 512)
        num_heads: Attention head 수 (예: 8)
        num_layers: Decoder layer 수 (예: 6)
        d_ff: Feed-forward 차원 (예: 2048)
        max_seq_len: 최대 시퀀스 길이 (예: 2048)
        dropout: Dropout 비율 (예: 0.1)
    """

    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=2048,
        dropout=0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz):
        """
        Causal mask for autoregressive generation

        마스크는 미래 토큰을 볼 수 없도록 함

        Returns:
            mask: (sz, sz) upper triangular matrix
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, src_mask=None):
        """
        Forward pass

        Args:
            src: (batch_size, seq_len) - MIDI event indices
            src_mask: (seq_len, seq_len) - causal mask

        Returns:
            output: (batch_size, seq_len, vocab_size) - logits
        """
        # Token embedding
        src = self.embedding(src) * math.sqrt(self.d_model)

        # Positional encoding
        src = self.pos_encoding(src)

        # Create causal mask if not provided
        if src_mask is None:
            device = src.device
            src_mask = self.generate_square_subsequent_mask(src.size(1)).to(device)

        # Pass through decoder layers
        attention_weights_list = []
        for layer in self.layers:
            src, attention_weights = layer(src, src_mask)
            attention_weights_list.append(attention_weights)

        # Output projection
        output = self.fc_out(src)

        return output

    @torch.no_grad()
    def generate(
        self,
        start_tokens,
        max_len=512,
        temperature=1.0,
        top_k=None,
        top_p=None
    ):
        """
        Autoregressive generation (sampling)

        Args:
            start_tokens: (1, start_len) - 시작 토큰들
            max_len: 생성할 최대 길이
            temperature: Sampling temperature (높을수록 diverse)
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            generated: (1, max_len) - 생성된 시퀀스
        """
        self.eval()

        generated = start_tokens

        for _ in range(max_len - start_tokens.size(1)):
            # Forward pass
            logits = self.forward(generated)

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float('Inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)

        return generated


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 388  # MIDI events (128 notes + controls)
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_len = 2048
    batch_size = 4
    seq_len = 128

    print("=" * 80)
    print("Music Transformer - Basic Implementation")
    print("=" * 80)

    # Create model
    model = MusicTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len
    )

    print(f"\n모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # Dummy input (random MIDI events)
    src = torch.randint(0, vocab_size, (batch_size, seq_len))

    print(f"\n입력 shape: {src.shape}")

    # Forward pass
    output = model(src)

    print(f"출력 shape: {output.shape}")
    print(f"출력 range: [{output.min():.2f}, {output.max():.2f}]")

    # Generation example
    print("\n" + "=" * 80)
    print("Generation Example")
    print("=" * 80)

    start_tokens = torch.randint(0, vocab_size, (1, 10))
    print(f"\n시작 토큰: {start_tokens.shape}")

    generated = model.generate(
        start_tokens,
        max_len=100,
        temperature=1.0,
        top_k=40
    )

    print(f"생성된 시퀀스: {generated.shape}")
    print(f"생성된 토큰 (처음 20개): {generated[0, :20].tolist()}")

    print("\n" + "=" * 80)
    print("다음 단계:")
    print("1. MAESTRO 데이터셋 다운로드")
    print("2. MIDI 전처리 파이프라인 구현")
    print("3. Training loop 작성")
    print("4. 생성된 음악 평가")
    print("=" * 80)
