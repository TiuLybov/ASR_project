import torch
from torch import Tensor, nn


class AdditiveNoise(nn.Module):
    """Add Gaussian noise to the waveform."""

    def __init__(self, noise_scale=0.005):
        super().__init__()
        self.noise_scale = noise_scale

    def __call__(self, data: Tensor) -> Tensor:
        noise = torch.randn_like(data) * self.noise_scale
        return data + noise
