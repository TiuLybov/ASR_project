import torch
from torch import Tensor, nn


class FrequencyMasking(nn.Module):
    """Mask random frequency bands in spectrogram."""

    def __init__(self, freq_mask_param=27):
        super().__init__()
        self.freq_mask_param = freq_mask_param

    def __call__(self, data: Tensor) -> Tensor:
        # data: (n_freq, T) or (B, n_freq, T)
        cloned = data.clone()
        n_freq = cloned.shape[-2]
        f = torch.randint(0, min(self.freq_mask_param, n_freq), (1,)).item()
        f0 = torch.randint(0, max(n_freq - f, 1), (1,)).item()
        cloned[..., f0 : f0 + f, :] = 0
        return cloned
