import torch
import torch.nn as nn

class BasePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    def forward(self, **kwargs):
        raise NotImplementedError()

    def predict_action(self, obs) -> list:
        raise NotImplementedError()

    def compute_loss(self, batch):
        raise NotImplementedError()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype