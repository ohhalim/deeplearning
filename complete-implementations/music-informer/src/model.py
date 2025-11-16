"""
Music Informer - Complete Implementation

Paper: "Music informer as an efficient model for music generation"
Authors: Sun, H., Wang, X., Wang, Y. et al.
Published: Nature Scientific Reports, Volume 15, 2025
DOI: 10.1038/s41598-025-02792-4

Key Innovation:
- ProbSparse Self-Attention (21.73% computational reduction)
- Relative Local Attention
- LSTM integration for sequential modeling

Performance:
- 21.73% faster than Music Transformer
- 31.87% faster than Performance RNN
- 41.33% faster than Multi-Track Music Transformer
- Better metrics: Pitch Class Entropy, Number of Pitches, etc.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple


# ============================================================================
# ProbSparse Self-Attention (핵심 혁신!)
# ============================================================================

class ProbSparseSelfAttention(nn.Module):
    """
    ProbSparse Self-Attention from Informer

    핵심 아이디어:
    1. Query의 "sparsity measurement" 계산
    2. Top-u개의 중요한 query만 선택
    3. 나머지는 mean pooling으로 근사

    이를 통해:
    - 시간 복잡도: O(L log L) (vs O(L²) in standard attention)
    - 메모리: 대폭 절감
    - 성능: 거의 동일 또는 더 좋음
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        sampling_factor: int = 5,  # c in paper
        dropout: float = 0.1
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.sampling_factor = sampling_factor

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _prob_QK(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        sample_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate sparsity measurement M(q_i, K)

        M(q_i, K) = max{qK^T} - mean{qK^T}

        Args:
            Q: (batch, heads, L_Q, d_k)
            K: (batch, heads, L_K, d_k)
            sample_k: number of samples

        Returns:
            M_top: (batch, heads, u, d_k) - top-u queries
            M_top_index: (batch, heads, u) - indices
        """
        # Sample a subset of keys
        L_K = K.shape[2]
        L_Q = Q.shape[2]

        # Random sampling of K
        if sample_k < L_K:
            # Randomly sample sample_k keys
            index_sample = torch.randint(
                0, L_K, (sample_k,), device=K.device
            )
            K_sample = K[:, :, index_sample, :]
        else:
            K_sample = K

        # Calculate Q·K^T for sampled K
        # (batch, heads, L_Q, sample_k)
        Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1))

        # Sparsity measurement
        M = Q_K_sample.max(dim=-1)[0] - Q_K_sample.mean(dim=-1)

        # Select top-u queries
        u = int(self.sampling_factor * np.log(L_Q))
        u = min(u, L_Q)

        M_top, M_top_index = torch.topk(M, u, dim=-1, sorted=False)

        return M_top_index

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len, seq_len)

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Q, K, V projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # ProbSparse Attention
        # 1. Get top-u query indices
        sample_k = int(self.sampling_factor * np.log(seq_len))
        sample_k = min(sample_k, seq_len)

        u_index = self._prob_QK(Q, K, sample_k)  # (batch, heads, u)

        # 2. Gather top-u queries
        # (batch, heads, u, d_k)
        Q_top = torch.gather(
            Q,
            dim=2,
            index=u_index.unsqueeze(-1).expand(-1, -1, -1, self.d_k)
        )

        # 3. Calculate attention for top-u queries
        # (batch, heads, u, seq_len)
        scores_top = torch.matmul(Q_top, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # Expand mask for top-u
            mask_top = torch.gather(
                mask.expand(batch_size, self.num_heads, seq_len, seq_len),
                dim=2,
                index=u_index.unsqueeze(-1).expand(-1, -1, -1, seq_len)
            )
            scores_top = scores_top.masked_fill(mask_top == 0, -1e9)

        attn_top = F.softmax(scores_top, dim=-1)
        attn_top = self.dropout(attn_top)

        # (batch, heads, u, d_k)
        context_top = torch.matmul(attn_top, V)

        # 4. For other queries, use mean pooling (approximation)
        # (batch, heads, seq_len, d_k)
        context = torch.zeros_like(Q)

        # Fill top-u positions
        context.scatter_(
            dim=2,
            index=u_index.unsqueeze(-1).expand(-1, -1, -1, self.d_k),
            src=context_top
        )

        # Fill remaining with mean of V
        V_mean = V.mean(dim=2, keepdim=True).expand_as(V)

        # Create mask for non-top positions
        top_mask = torch.zeros(batch_size, self.num_heads, seq_len, 1, device=x.device)
        top_mask.scatter_(dim=2, index=u_index.unsqueeze(-1), value=1.0)

        context = context + V_mean * (1 - top_mask)

        # 5. Concatenate and output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)

        return output


# ============================================================================
# Relative Local Attention (Music Transformer의 핵심)
# ============================================================================

class RelativeLocalAttention(nn.Module):
    """
    Relative Local Attention

    음악에서 중요한 relative position 정보 활용
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        max_relative_position: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_relative_position = max_relative_position

        # Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Relative position embeddings
        self.relative_position_k = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, self.d_k)
        )
        self.relative_position_v = nn.Parameter(
            torch.randn(2 * max_relative_position + 1, self.d_k)
        )

        self.dropout = nn.Dropout(dropout)

    def _get_relative_embeddings(self, seq_len: int) -> torch.Tensor:
        """
        Get relative position embeddings

        Returns:
            relative_embeddings: (seq_len, seq_len, d_k)
        """
        # Relative positions: i - j
        positions = torch.arange(seq_len, device=self.relative_position_k.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)

        # Clip to max_relative_position
        relative_positions = torch.clamp(
            relative_positions,
            -self.max_relative_position,
            self.max_relative_position
        )

        # Shift to [0, 2*max_relative_position]
        relative_positions += self.max_relative_position

        # Get embeddings
        relative_embeddings = self.relative_position_k[relative_positions]

        return relative_embeddings

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len, seq_len)

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Standard attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Add relative position bias
        relative_k = self._get_relative_embeddings(seq_len)  # (seq_len, seq_len, d_k)

        # (batch, heads, seq_len, d_k) @ (seq_len, seq_len, d_k)
        # → (batch, heads, seq_len, seq_len)
        relative_scores = torch.einsum('bhld,lrd->bhlr', Q, relative_k.unsqueeze(0))
        scores = scores + relative_scores / math.sqrt(self.d_k)

        # Mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Attention
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Context
        context = torch.matmul(attn, V)

        # Output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)

        return output


# ============================================================================
# Music Informer Layer
# ============================================================================

class MusicInformerLayer(nn.Module):
    """
    Music Informer Encoder Layer

    구조:
    1. ProbSparse Self-Attention
    2. Relative Local Attention (선택적)
    3. LSTM
    4. Feed-Forward
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 1024,
        d_lstm: int = 1024,
        use_relative: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        # ProbSparse Attention
        self.prob_attn = ProbSparseSelfAttention(d_model, num_heads, dropout=dropout)

        # Relative Local Attention (optional)
        self.use_relative = use_relative
        if use_relative:
            self.rel_attn = RelativeLocalAttention(d_model, num_heads, dropout=dropout)

        # LSTM (2 layers as in paper)
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_lstm,
            num_layers=2,
            dropout=dropout,
            batch_first=True
        )
        self.lstm_proj = nn.Linear(d_lstm, d_model)

        # Feed-Forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),  # Paper uses ReLU
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        if use_relative:
            self.norm_rel = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, 1, seq_len, seq_len)

        Returns:
            output: (batch, seq_len, d_model)
        """
        # 1. ProbSparse Attention
        attn_out = self.prob_attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # 2. Relative Attention (optional)
        if self.use_relative:
            rel_out = self.rel_attn(x, mask)
            x = self.norm_rel(x + self.dropout(rel_out))

        # 3. LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_proj(lstm_out)
        x = self.norm2(x + self.dropout(lstm_out))

        # 4. FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))

        return x


# ============================================================================
# Complete Music Informer Model
# ============================================================================

class MusicInformer(nn.Module):
    """
    Complete Music Informer Model

    Paper specifications:
    - 6 encoder layers
    - 8 attention heads
    - 512 hidden dimension
    - 1024 feedforward dimension
    - 2-layer LSTM (512 input, 1024 hidden)
    - Dropout: 0.1
    - Optimizer: Adam
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 1024,
        d_lstm: int = 1024,
        max_seq_len: int = 2048,
        use_relative: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

        # Encoder layers
        self.layers = nn.ModuleList([
            MusicInformerLayer(
                d_model, num_heads, d_ff, d_lstm, use_relative, dropout
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Initialize
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Causal mask"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: (batch, seq_len) - token indices
            src_mask: (seq_len, seq_len) - causal mask

        Returns:
            output: (batch, seq_len, vocab_size) - logits
        """
        # Embedding
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src)

        # Causal mask
        if src_mask is None:
            device = src.device
            src_mask = self.generate_square_subsequent_mask(src.size(1)).to(device)

        # Encoder layers
        for layer in self.layers:
            src = layer(src, src_mask)

        # Output
        output = self.fc_out(src)

        return output

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_len: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> torch.Tensor:
        """
        Autoregressive generation

        Args:
            start_tokens: (1, start_len)
            max_len: max length to generate
            temperature: sampling temperature
            top_k: top-k sampling
            top_p: nucleus sampling

        Returns:
            generated: (1, max_len)
        """
        self.eval()
        generated = start_tokens

        for _ in range(max_len - start_tokens.size(1)):
            logits = self.forward(generated)
            logits = logits[:, -1, :] / temperature

            # Top-k
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')

            # Top-p
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

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

            # Append
            generated = torch.cat([generated, next_token], dim=1)

        return generated


class PositionalEncoding(nn.Module):
    """Positional Encoding"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Music Informer - Complete Implementation")
    print("=" * 80)

    # Create model (paper specifications)
    model = MusicInformer(
        vocab_size=388,      # MIDI events
        d_model=512,         # Paper spec
        num_layers=6,        # Paper spec
        num_heads=8,         # Paper spec
        d_ff=1024,           # Paper spec
        d_lstm=1024,         # Paper spec
        max_seq_len=2048,
        use_relative=True,
        dropout=0.1
    )

    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    batch_size = 4
    seq_len = 128

    src = torch.randint(0, 388, (batch_size, seq_len))

    print(f"\nInput: {src.shape}")

    output = model(src)

    print(f"Output: {output.shape}")

    # Test generation
    start_tokens = torch.randint(0, 388, (1, 10))
    generated = model.generate(start_tokens, max_len=100, temperature=1.0, top_k=40)

    print(f"\nGenerated: {generated.shape}")

    print("\n" + "=" * 80)
    print("Key Features:")
    print("1. ProbSparse Attention: 21.73% computational reduction")
    print("2. Relative Local Attention: Music-specific positional encoding")
    print("3. LSTM: Sequential modeling")
    print("4. 6 layers, 8 heads, 512 hidden (Nature paper spec)")
    print("=" * 80)
