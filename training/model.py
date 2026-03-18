"""
DeFeedback Pro — Lightweight GRU Denoiser/Dereverberator Model

Architecture inspired by DeepFilterNet but simplified:
- Input: magnitude spectrum frames [batch, num_frames, num_bins]
- Two GRU layers for temporal modeling
- Two output heads: noise_mask and reverb_mask, both in [0, 1]

One model per FFT size (input/output dims = fft_size//2 + 1).
"""

import torch
import torch.nn as nn


class DeFeedbackNet(nn.Module):
    """Lightweight spectral mask estimator with separate noise/reverb heads."""

    def __init__(self, num_bins: int, hidden_size: int = 256, num_layers: int = 2,
                 num_frames: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_bins = num_bins
        self.num_frames = num_frames
        self.hidden_size = hidden_size

        # Input projection: compress spectral bins
        self.input_norm = nn.LayerNorm(num_bins)
        self.input_proj = nn.Linear(num_bins, hidden_size)

        # Temporal modeling with GRU
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Noise mask head
        self.noise_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_bins),
            nn.Sigmoid()
        )

        # Reverb mask head
        self.reverb_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_bins),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Magnitude spectrum [batch, num_frames, num_bins]

        Returns:
            noise_mask: [batch, num_bins] — gain mask for noise suppression
            reverb_mask: [batch, num_bins] — gain mask for dereverberation
        """
        # Log-compress input for better dynamic range
        # Input is magnitude (non-negative), clamp for safety
        x = torch.log1p(x.clamp(min=0.0))

        # Normalize per-frame
        x = self.input_norm(x)

        # Project to hidden dimension
        x = self.input_proj(x)  # [batch, num_frames, hidden]

        # Temporal modeling
        gru_out, _ = self.gru(x)  # [batch, num_frames, hidden]

        # Use only the last frame's output
        last_frame = gru_out[:, -1, :]  # [batch, hidden]

        # Two separate mask heads
        noise_mask = self.noise_head(last_frame)    # [batch, num_bins]
        reverb_mask = self.reverb_head(last_frame)  # [batch, num_bins]

        return noise_mask, reverb_mask


def get_model_config(fft_size: int) -> dict:
    """Model size scales with FFT size for efficiency."""
    num_bins = fft_size // 2 + 1

    if fft_size <= 256:
        return dict(num_bins=num_bins, hidden_size=128, num_layers=2)
    elif fft_size <= 512:
        return dict(num_bins=num_bins, hidden_size=192, num_layers=2)
    else:
        return dict(num_bins=num_bins, hidden_size=256, num_layers=2)


def create_model(fft_size: int) -> DeFeedbackNet:
    config = get_model_config(fft_size)
    return DeFeedbackNet(**config)
