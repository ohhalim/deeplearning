"""
Music Transformer Model

작은 Transformer 기반 음악 생성 모델

작성자: Your Name
날짜: 2025-11-19
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """
    Positional Encoding for Transformer
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, :x.size(1), :]


class MusicTransformer(nn.Module):
    """
    Small Transformer for Music Generation

    Architecture:
    - Embedding + Positional Encoding
    - Transformer Decoder Layers
    - Output Linear Layer

    Args:
        vocab_size: Vocabulary size (default: 391)
        d_model: Model dimension (default: 256)
        num_layers: Number of transformer layers (default: 4)
        num_heads: Number of attention heads (default: 8)
        d_ff: Feed-forward dimension (default: 1024)
        max_seq_len: Maximum sequence length (default: 512)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        vocab_size=391,
        d_model=256,
        num_layers=4,
        num_heads=8,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers

        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        # Output layer
        self.output_linear = nn.Linear(d_model, vocab_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_linear.bias.data.zero_()
        self.output_linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt_mask=None):
        """
        Forward pass

        Args:
            src: Input token IDs (batch_size, seq_len)
            tgt_mask: Causal mask for autoregressive generation

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = src.shape

        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(seq_len).to(src.device)

        # Embedding
        x = self.embedding(src) * math.sqrt(self.d_model)  # Scale by sqrt(d_model)

        # Positional encoding
        x = self.pos_encoding(x)

        # Dropout
        x = self.dropout(x)

        # Transformer decoder
        # Note: Using memory=x (self-attention only)
        output = self.transformer_decoder(
            tgt=x,
            memory=x,  # Self-attention
            tgt_mask=tgt_mask
        )

        # Output projection
        logits = self.output_linear(output)

        return logits

    def _generate_square_subsequent_mask(self, sz):
        """
        Generate causal mask

        Args:
            sz: Sequence length

        Returns:
            mask: (sz, sz) with True where attention is not allowed
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask

    def generate(
        self,
        start_tokens,
        max_length=512,
        temperature=1.0,
        top_k=40,
        top_p=0.9,
        device='cuda'
    ):
        """
        Generate music tokens autoregressively

        Args:
            start_tokens: Starting tokens (list or tensor)
            max_length: Maximum length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling (0 = disabled)
            top_p: Nucleus sampling (1.0 = disabled)
            device: Device to use

        Returns:
            generated_tokens: List of generated token IDs
        """
        self.eval()

        # Convert to tensor if needed
        if isinstance(start_tokens, list):
            tokens = torch.tensor([start_tokens], dtype=torch.long, device=device)
        else:
            tokens = start_tokens.clone().to(device)

        with torch.no_grad():
            for _ in range(max_length - tokens.size(1)):
                # Forward pass
                logits = self.forward(tokens)  # (batch, seq_len, vocab)

                # Get last token logits
                next_token_logits = logits[0, -1, :] / temperature  # (vocab,)

                # Top-k sampling
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = -float('Inf')

                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append to sequence
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)

                # Check for EOS token (assuming EOS=390)
                if next_token.item() == 390:  # EOS token
                    break

        return tokens[0].cpu().tolist()

    def get_num_params(self):
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters())


def create_model(config):
    """
    Create model from config dict

    Args:
        config: Config dictionary

    Returns:
        model: MusicTransformer instance
    """
    model_config = config.get('model', {})

    model = MusicTransformer(
        vocab_size=model_config.get('vocab_size', 391),
        d_model=model_config.get('d_model', 256),
        num_layers=model_config.get('num_layers', 4),
        num_heads=model_config.get('num_heads', 8),
        d_ff=model_config.get('d_ff', 1024),
        max_seq_len=model_config.get('max_seq_len', 512),
        dropout=model_config.get('dropout', 0.1)
    )

    return model


# Test code
if __name__ == '__main__':
    # Create small model
    model = MusicTransformer(
        vocab_size=391,
        d_model=256,
        num_layers=4,
        num_heads=8,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1
    )

    print(f"Model created:")
    print(f"  Parameters: {model.get_num_params():,}")
    print(f"  Memory: ~{model.get_num_params() * 4 / 1024 / 1024:.1f} MB (FP32)")

    # Test forward pass
    batch_size = 2
    seq_len = 128
    dummy_input = torch.randint(0, 391, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(dummy_input)

    print(f"\nForward pass test:")
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Output shape: {logits.shape}")

    # Test generation
    start_tokens = [389]  # BOS token
    generated = model.generate(start_tokens, max_length=50, device='cpu')

    print(f"\nGeneration test:")
    print(f"  Start tokens: {start_tokens}")
    print(f"  Generated length: {len(generated)}")
    print(f"  Generated tokens: {generated[:20]}...")

    print("\nModel ready for training!")
