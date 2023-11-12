import torch.nn as nn
from torch import Tensor
from torchmetrics.audio import ScaleInvariantSignalDistortionRatio as si_sdr


class SpexPlusLoss(nn.Module):
    def __init__(self, alpha = 0.1, beta=0.1, gamma=0.5):
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.si_sdr = si_sdr()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, s1, s2, s3, spk_pred, target, speaker,
                **batch) -> Tensor:
        device = s1.get_device()
        self.si_sdr.to(device)
        target = target[:, s1.shape[-1]]
        si_sdr_loss = -(1 - self.alpha - self.beta) * 2 * self.si_sdr(s1, target) - self.alpha * 2 * self.si_sdr(s2, target) - self.beta * 2 * self.si_sdr(s3, target)
        ce_loss = self.ce(spk_pred, speaker)
        return si_sdr_loss + self.gamma * ce_loss

