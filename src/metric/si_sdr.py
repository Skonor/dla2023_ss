from typing import List

import torch
from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as si_sdr

from src.base.base_metric import BaseMetric


class SI_SDR(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.si_sdr = si_sdr()

    def __call__(self, s1: Tensor, target: Tensor, **kwargs):
        device = s1.get_device()
        self.si_sdr.to(device)
        target = target[:, s1.shape[-1]]
        return self.si_sdr(s1, target)