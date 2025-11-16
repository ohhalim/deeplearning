"""
Month 4: VAE + WaveNet - Variational Autoencoder for Audio

목표:
1. VAE 개념 이해 및 구현
2. Latent space 탐색
3. 오디오 스펙트로그램 생성

참고 논문:
- "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
- "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import matplotlib.pyplot as plt


class AudioVAE(nn.Module):
    """
    VAE for Mel-Spectrogram Generation

    핵심 개념:
    1. Encoder: x → μ, σ (latent distribution parameters)
    2. Reparameterization: z = μ + σ * ε, where ε ~ N(0,1)
    3. Decoder: z → x_reconstructed

    Loss = Reconstruction Loss + KL Divergence
    """

    def __init__(
        self,
        input_shape=(128, 128),  # (n_mels, n_frames)
        latent_dim=256,
        hidden_dims=[32, 64, 128, 256, 512]
    ):
        super().__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim

        # ========== Encoder ==========
        encoder_layers = []
        in_channels = 1  # Mono spectrogram

        for h_dim in hidden_dims:
            encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2)
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Calculate flattened size after convolutions
        # After 5 conv layers with stride 2: 128 / (2^5) = 4
        self.encoder_out_size = hidden_dims[-1] * 4 * 4

        # Latent distribution parameters
        self.fc_mu = nn.Linear(self.encoder_out_size, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_out_size, latent_dim)

        # ========== Decoder ==========
        self.decoder_input = nn.Linear(latent_dim, self.encoder_out_size)

        hidden_dims.reverse()  # Reverse for decoder

        decoder_layers = []
        for i in range(len(hidden_dims) - 1):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i],
                        hidden_dims[i + 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2)
                )
            )

        self.decoder = nn.Sequential(*decoder_layers)

        # Final layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                1,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def encode(self, x):
        """
        Encode input to latent distribution parameters

        Args:
            x: (batch, 1, H, W) - mel-spectrogram

        Returns:
            mu, logvar: (batch, latent_dim)
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε

        Args:
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)

        Returns:
            z: (batch, latent_dim)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # ε ~ N(0, 1)

        z = mu + eps * std

        return z

    def decode(self, z):
        """
        Decode latent to reconstruction

        Args:
            z: (batch, latent_dim)

        Returns:
            x_recon: (batch, 1, H, W)
        """
        h = self.decoder_input(z)
        h = h.view(h.size(0), 512, 4, 4)  # Reshape

        h = self.decoder(h)
        x_recon = self.final_layer(h)

        return x_recon

    def forward(self, x):
        """
        Forward pass

        Args:
            x: (batch, 1, H, W)

        Returns:
            x_recon: (batch, 1, H, W)
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)

        return x_recon, mu, logvar

    def sample(self, num_samples, device='cuda'):
        """
        Sample from prior distribution

        Args:
            num_samples: int

        Returns:
            samples: (num_samples, 1, H, W)
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)

        return samples


def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    VAE Loss = Reconstruction Loss + β * KL Divergence

    Args:
        x_recon: (batch, 1, H, W) - reconstructed
        x: (batch, 1, H, W) - original
        mu: (batch, latent_dim)
        logvar: (batch, latent_dim)
        beta: KL weight (β-VAE)

    Returns:
        loss: scalar
        recon_loss: scalar
        kld_loss: scalar
    """
    # Reconstruction loss (MSE or BCE)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum') / x.size(0)

    # KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # Total loss
    loss = recon_loss + beta * kld_loss

    return loss, recon_loss, kld_loss


# ============================================================================
# Audio Utilities
# ============================================================================

def load_audio_to_mel(audio_path, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    """
    Load audio and convert to mel-spectrogram

    Args:
        audio_path: str
        sr: sample rate
        n_mels: number of mel bands
        n_fft: FFT window size
        hop_length: hop length

    Returns:
        mel_spec: (n_mels, n_frames)
    """
    # Load audio
    y, _ = librosa.load(audio_path, sr=sr)

    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Normalize to [0, 1]
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())

    return mel_spec_norm


def mel_to_audio(mel_spec, sr=22050, n_fft=2048, hop_length=512):
    """
    Convert mel-spectrogram back to audio (Griffin-Lim)

    Args:
        mel_spec: (n_mels, n_frames) in [0, 1]
        sr: sample rate
        n_fft: FFT window size
        hop_length: hop length

    Returns:
        y: audio waveform
    """
    # Denormalize (assuming normalized to [0, 1])
    # This is a simplified version - you may need to store original min/max

    # Convert back to power
    mel_spec_power = librosa.db_to_power(mel_spec * 80 - 80)

    # Inverse mel
    spec = librosa.feature.inverse.mel_to_stft(
        mel_spec_power,
        sr=sr,
        n_fft=n_fft
    )

    # Griffin-Lim reconstruction
    y = librosa.griffinlim(spec, hop_length=hop_length)

    return y


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Audio VAE Example")
    print("=" * 80)

    # Create model
    model = AudioVAE(
        input_shape=(128, 128),
        latent_dim=256
    )

    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Dummy input (random mel-spectrogram)
    batch_size = 8
    x = torch.rand(batch_size, 1, 128, 128)

    print(f"\nInput shape: {x.shape}")

    # Forward pass
    x_recon, mu, logvar = model(x)

    print(f"Reconstructed shape: {x_recon.shape}")
    print(f"Latent mu shape: {mu.shape}")
    print(f"Latent logvar shape: {logvar.shape}")

    # Loss
    loss, recon_loss, kld_loss = vae_loss(x_recon, x, mu, logvar, beta=1.0)

    print(f"\nTotal Loss: {loss.item():.4f}")
    print(f"Reconstruction Loss: {recon_loss.item():.4f}")
    print(f"KL Divergence: {kld_loss.item():.4f}")

    # Sampling
    print("\n" + "=" * 80)
    print("Sampling from prior")
    print("=" * 80)

    samples = model.sample(num_samples=4, device='cpu')
    print(f"Sampled shape: {samples.shape}")

    print("\n" + "=" * 80)
    print("다음 단계:")
    print("1. NSynth 데이터셋 다운로드")
    print("2. VAE 학습")
    print("3. Latent space interpolation")
    print("4. Audio reconstruction 품질 평가")
    print("=" * 80)
