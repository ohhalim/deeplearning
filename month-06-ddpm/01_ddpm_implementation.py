"""
Month 6: DDPM (Denoising Diffusion Probabilistic Models)

목표:
1. Diffusion forward/reverse process 이해
2. Noise schedule 설계
3. 음악 스펙트로그램 생성

참고 논문:
- "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Time step embeddings for diffusion"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SimpleUNet(nn.Module):
    """
    Simplified U-Net for diffusion denoising

    Input: Noisy spectrogram + time embedding
    Output: Predicted noise
    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        time_emb_dim=128,
        hidden_dims=[64, 128, 256, 512]
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )

        # Encoder (down)
        self.encoder = nn.ModuleList()
        prev_dim = in_channels

        for h_dim in hidden_dims:
            self.encoder.append(nn.Sequential(
                nn.Conv2d(prev_dim, h_dim, 3, padding=1),
                nn.GroupNorm(8, h_dim),
                nn.GELU(),
                nn.Conv2d(h_dim, h_dim, 3, padding=1),
                nn.GroupNorm(8, h_dim),
                nn.GELU(),
                nn.MaxPool2d(2)
            ))
            prev_dim = h_dim

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1] * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_dims[-1] * 2),
            nn.GELU(),
            nn.Conv2d(hidden_dims[-1] * 2, hidden_dims[-1], 3, padding=1),
            nn.GroupNorm(8, hidden_dims[-1]),
            nn.GELU()
        )

        # Decoder (up)
        self.decoder = nn.ModuleList()
        hidden_dims_reversed = list(reversed(hidden_dims))

        for i in range(len(hidden_dims_reversed) - 1):
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims_reversed[i] * 2,  # *2 for skip connection
                    hidden_dims_reversed[i + 1],
                    2,
                    stride=2
                ),
                nn.GroupNorm(8, hidden_dims_reversed[i + 1]),
                nn.GELU(),
                nn.Conv2d(hidden_dims_reversed[i + 1], hidden_dims_reversed[i + 1], 3, padding=1),
                nn.GroupNorm(8, hidden_dims_reversed[i + 1]),
                nn.GELU()
            ))

        # Final layer
        self.final = nn.Conv2d(hidden_dims[0] * 2, out_channels, 1)

    def forward(self, x, t):
        """
        Args:
            x: (batch, 1, H, W) - noisy input
            t: (batch,) - time step

        Returns:
            noise: (batch, 1, H, W) - predicted noise
        """
        # Time embedding
        t_emb = self.time_mlp(t)

        # Encoder
        skip_connections = []
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = list(reversed(skip_connections))
        for i, decoder_block in enumerate(self.decoder):
            x = torch.cat([x, skip_connections[i]], dim=1)
            x = decoder_block(x)

        # Final
        x = torch.cat([x, skip_connections[-1]], dim=1)
        noise_pred = self.final(x)

        return noise_pred


class DDPM:
    """
    Denoising Diffusion Probabilistic Model

    Forward process (diffusion):
        q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t) * x_{t-1}, β_t * I)

    Reverse process (denoising):
        p(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
    """

    def __init__(
        self,
        model,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device='cuda'
    ):
        self.model = model
        self.timesteps = timesteps
        self.device = device

        # β schedule (linear)
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)

        # α = 1 - β
        self.alphas = 1.0 - self.betas

        # α_bar = ∏ α_i
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Precompute values for q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Precompute values for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_0, t, noise=None):
        """
        Sample from q(x_t | x_0) - forward diffusion

        x_t = √α_bar_t * x_0 + √(1 - α_bar_t) * ε

        Args:
            x_0: (batch, C, H, W) - clean input
            t: (batch,) - time steps
            noise: (batch, C, H, W) - noise (optional)

        Returns:
            x_t: (batch, C, H, W) - noisy input at time t
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t.view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t.view(-1, 1, 1, 1)

        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

        return x_t

    def p_losses(self, x_0, t, noise=None):
        """
        Training loss: L = E[||ε - ε_θ(x_t, t)||²]

        Args:
            x_0: (batch, C, H, W) - clean input
            t: (batch,) - time steps
            noise: (batch, C, H, W) - noise (optional)

        Returns:
            loss: scalar
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Forward diffusion
        x_t = self.q_sample(x_0, t, noise)

        # Predict noise
        noise_pred = self.model(x_t, t)

        # MSE loss
        loss = F.mse_loss(noise, noise_pred)

        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        Sample from p(x_{t-1} | x_t) - reverse diffusion

        Args:
            x_t: (batch, C, H, W) - noisy input at time t
            t: int - current time step

        Returns:
            x_{t-1}: (batch, C, H, W) - denoised output
        """
        batch_size = x_t.shape[0]

        # Time tensor
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

        # Predict noise
        noise_pred = self.model(x_t, t_tensor)

        # Get coefficients
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        beta_t = self.betas[t]

        # Mean
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)

        mean = (x_t - (beta_t / sqrt_one_minus_alpha_cumprod_t) * noise_pred) / sqrt_alpha_t

        # Variance
        if t > 0:
            noise = torch.randn_like(x_t)
            variance = torch.sqrt(self.posterior_variance[t])
            x_t_minus_1 = mean + variance * noise
        else:
            x_t_minus_1 = mean

        return x_t_minus_1

    @torch.no_grad()
    def sample(self, shape):
        """
        Generate samples from noise

        Args:
            shape: (batch, C, H, W)

        Returns:
            x_0: (batch, C, H, W) - generated samples
        """
        batch_size = shape[0]
        device = self.device

        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        # Reverse diffusion
        for t in reversed(range(self.timesteps)):
            x_t = self.p_sample(x_t, t)

        return x_t


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("DDPM Implementation")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Model
    unet = SimpleUNet(
        in_channels=1,
        out_channels=1,
        time_emb_dim=128,
        hidden_dims=[64, 128, 256, 512]
    ).to(device)

    print(f"Model Parameters: {sum(p.numel() for p in unet.parameters()):,}")

    # DDPM
    ddpm = DDPM(
        model=unet,
        timesteps=1000,
        beta_start=1e-4,
        beta_end=0.02,
        device=device
    )

    # Training example
    print("\n" + "=" * 80)
    print("Training Example")
    print("=" * 80)

    batch_size = 4
    x_0 = torch.randn(batch_size, 1, 64, 64).to(device)  # Clean mel-spectrogram
    t = torch.randint(0, 1000, (batch_size,)).to(device)  # Random time steps

    loss = ddpm.p_losses(x_0, t)
    print(f"Training Loss: {loss.item():.4f}")

    # Sampling example
    print("\n" + "=" * 80)
    print("Sampling Example")
    print("=" * 80)

    samples = ddpm.sample(shape=(2, 1, 64, 64))
    print(f"Generated samples shape: {samples.shape}")

    print("\n" + "=" * 80)
    print("다음 단계:")
    print("1. 오디오 데이터셋으로 학습")
    print("2. DDIM으로 가속 (50 steps)")
    print("3. Conditional generation (클래스, 텍스트)")
    print("4. 음악 생성 품질 평가")
    print("=" * 80)
