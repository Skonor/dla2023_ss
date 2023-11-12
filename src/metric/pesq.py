from typing import List

import torch
from torch import Tensor

from src.base.base_metric import BaseMetric
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality



class PESQ(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode="wb")

    def __call__(self, s1: Tensor, target: Tensor, **kwargs):
        device = s1.get_device()
        self.pesq.to(device)
        common_len = min(s1.shape[-1], target.shape[-1])
        target = target[:, :common_len]
        s1 = s1[:, :common_len]
        try:
            return self.pesq(s1, target).mean()
        except:
            return torch.tensor(-0.5)
    