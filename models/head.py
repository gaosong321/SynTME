import torch
import torch.nn as nn

class TMESynergyPredictor(torch.nn.Module):
    def __init__(self, out_channels=256):
        super(TMESynergyPredictor, self).__init__()
        self._synergy_mlp = nn.Sequential(
            nn.Linear(out_channels * 40, int(out_channels * 20)),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(int(out_channels * 20), out_channels * 10),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(out_channels * 10, 1),
        )

    def forward(self, latent_tensor):
        _prediction = self._synergy_mlp(latent_tensor)
        return _prediction

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)