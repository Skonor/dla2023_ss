from torch import nn
from torch.nn import Sequential

from src.base import BaseModel


class BaselineModel(BaseModel):
    def __init__(self, num_speakers, fc_hidden=2, **batch):
        super().__init__(**batch)
        self.net = Sequential(
            # people say it can aproximate any function...
            nn.Linear(in_features=1, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=1)
        )

        self.classifier = nn.Linear(1, num_speakers)

    def forward(self, mix, ref, **batch):
        mix = mix.unsqueeze(-1)
        spk = self.classifier(ref[:, 0].unsqueeze(-1))

        return {"s1": self.net(mix).squeeze(-1),
                "s2": self.net(mix).squeeze(-1),
                "s3": self.net(mix).squeeze(-1),
                "spk_pred": spk}