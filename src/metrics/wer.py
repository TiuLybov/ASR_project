from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer


class ArgmaxWERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, text: List[str], **kwargs
    ):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        for pred_ids, target_text in zip(predictions, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.decode(pred_ids)
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)


class BeamSearchWERMetric(BaseMetric):
    def __init__(self, text_encoder, beam_size=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.beam_size = beam_size

    def __call__(
        self, log_probs: Tensor, text: List[str], **kwargs
    ):
        wers = []
        for i in range(log_probs.shape[0]):
            target_text = self.text_encoder.normalize_text(text[i])
            pred_text = self.text_encoder.beam_search_decode(
                log_probs[i].cpu(), beam_size=self.beam_size
            )
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)
