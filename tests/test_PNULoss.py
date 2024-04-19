from src.utils import PNULoss, Multi_PNULoss
import torch
import torch.nn as nn
import pytest


class TestPNULoss:
    def test_output_size(self):
        t = torch.tensor([[1], [-1], [0], [1], [1], [-1]])
        y = torch.tensor([[0.8], [-0.7], [0.1], [0.6], [0.9], [-0.3]])

        loss_func = nn.Sigmoid()
        p_ratio = sum(t == 1).item() / len(t)
        eta = 0.2
        pnu_loss = PNULoss(loss_func, p_ratio, eta)
        assert pnu_loss.Risk_PNU(t, y).shape == torch.Size([])


class TestMultiPNULoss:
    t = torch.tensor([[0, 1], [1, 0], [0, 0], [0, 1], [0, 1], [1, 0]])
    y = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.1, 0.9], [0.2, 0.8], [0.8, 0.2]])

    loss_func = nn.Sigmoid()
    eta = 0.2

    def test_output_size(self):
        p_ratio = (self.t == 1).sum(dim=0) / len(self.t)
        pnu_loss = Multi_PNULoss(self.loss_func, p_ratio, self.eta)
        assert pnu_loss.Risk_PNU(self.t, self.y).shape == torch.Size([])

    def test_p_ratio_type(self):
        p_ratio = 0.5
        with pytest.raises(AssertionError):
            Multi_PNULoss(self.loss_func, p_ratio, self.eta)

    def test_p_ratio_shape(self):
        p_ratio = torch.tensor([0.5])
        with pytest.raises(AssertionError):
            pnu_loss = Multi_PNULoss(self.loss_func, p_ratio, self.eta)
            pnu_loss.Risk_PNU(self.t, self.y)
