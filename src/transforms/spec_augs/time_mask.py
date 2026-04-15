import torch
from torch import Tensor, nn


class TimeMasking(nn.Module):
    """Mask random time steps in spectrogram."""

    def __init__(self, time_mask_param=100):
        super().__init__()
        self.time_mask_param = time_mask_param

    def __call__(self, data: Tensor) -> Tensor:
        # data: (n_freq, T) or (B, n_freq, T)
        cloned = data.clone()
        T = cloned.shape[-1]
        t = torch.randint(0, min(self.time_mask_param, T), (1,)).item()
        t0 = torch.randint(0, max(T - t, 1), (1,)).item()
        cloned[..., :, t0 : t0 + t] = 0
        return cloned
