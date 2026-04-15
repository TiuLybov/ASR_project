import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SpeedPerturb(nn.Module):
    """
    Speed perturbation by resampling.
    Changes tempo without pitch correction.
    """

    def __init__(self, min_speed=0.9, max_speed=1.1):
        super().__init__()
        self.min_speed = min_speed
        self.max_speed = max_speed

    def __call__(self, data: Tensor) -> Tensor:
        speed = torch.empty(1).uniform_(self.min_speed, self.max_speed).item()
        if abs(speed - 1.0) < 1e-3:
            return data
        # data: (1, T) or (T,)
        orig_shape = data.shape
        if data.dim() == 1:
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.dim() == 2:
            data = data.unsqueeze(0)

        new_len = int(data.shape[-1] / speed)
        data = F.interpolate(data.float(), size=new_len, mode="linear", align_corners=False)

        if len(orig_shape) == 1:
            data = data.squeeze(0).squeeze(0)
        elif len(orig_shape) == 2:
            data = data.squeeze(0)
        return data
