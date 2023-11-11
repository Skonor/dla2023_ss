from typing import List

import torch
from torch import Tensor

from src.base.base_metric import BaseMetric



class PESQ(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, mix: Tensor, target: Tensor, **kwargs):
        pass
    