import torch
from torch import Tensor, nn


class LASLoss(nn.Module):
    """
    Cross-entropy loss for LAS decoder output.
    Ignores padding positions (pad_id=0).
    """

    def __init__(self, pad_id=0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, logits, text_encoded, text_encoded_length, **batch) -> dict:
        """
        Args:
            logits: (B, S_pred, vocab_size) — decoder output
            text_encoded: (B, S_target) — ground truth token ids
            text_encoded_length: (B,) — lengths of targets
        Returns:
            dict with 'loss' key
        """
        B, S_pred, V = logits.shape
        S_target = text_encoded.shape[1]

        # align lengths — truncate or pad target to match prediction length
        min_len = min(S_pred, S_target)
        logits_trimmed = logits[:, :min_len, :]
        targets_trimmed = text_encoded[:, :min_len]

        loss = self.criterion(
            logits_trimmed.reshape(-1, V),
            targets_trimmed.reshape(-1),
        )

        return {"loss": loss}
