import torch.nn as nn
from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as si_sdr


class SpexPlusLossNoCe(nn.Module):
    def __init__(self, alpha = 0.1, beta=0.1, gamma=0.0):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.si_sdr = si_sdr()

    def forward(self, s1, s2, s3, target,
                **batch) -> Tensor:
        device = s1.get_device()
        self.si_sdr.to(device)
        common_len = min(s1.shape[-1], s2.shape[-1], s3.shape[-1], target.shape[-1])
        target = target[:, :common_len]
        s1 = s1[:, :common_len]
        s2 = s2[:, :common_len]
        s3 = s3[:, :common_len]
        return -(1 - self.alpha - self.beta) * 2 * self.si_sdr(s1, target) - self.alpha * 2 * self.si_sdr(s2, target) - self.beta * 2 * self.si_sdr(s3, target)

