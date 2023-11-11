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
        return self.si_sdr(s1, target)