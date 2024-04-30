import pytest
import torch
import torch.nn as nn

from src.pnu_loss import Multi_PNULoss, PNULoss


class TestPNULoss:
    t = torch.tensor([[1], [-1], [0], [1], [1], [-1]])
    y = torch.tensor([[0.8], [-0.7], [0.1], [0.6], [0.9], [-0.3]])

    p_ratio = torch.tensor([0.5, 0.33])
    eta = 0.2

    def test_output_size(self):
        pnu_loss = PNULoss(self.p_ratio, self.eta)
        assert pnu_loss(self.t, self.y).shape == torch.Size([])

    def test_p_ratio_shape(self):
        p_ratio = torch.tensor([0.5])
        with pytest.raises(AssertionError):
            pnu_loss = PNULoss(p_ratio, self.eta)
            pnu_loss(self.t, self.y)


class TestMultiPNULoss:
    t = torch.tensor([[0, 1], [1, 0], [0, 0], [0, 1], [0, 1], [1, 0]])
    y = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6], [0.1, 0.9], [0.2, 0.8], [0.8, 0.2]])
    eta = 0.2

    def test_output_size(self):
        p_ratio = (self.t == 1).sum(dim=0) / len(self.t)
        pnu_loss = Multi_PNULoss(p_ratio, self.eta)
        assert pnu_loss(self.t, self.y).shape == torch.Size([])

    def test_p_ratio_shape(self):
        p_ratio = torch.tensor([0.5])
        with pytest.raises(AssertionError):
            pnu_loss = Multi_PNULoss(p_ratio, self.eta)
            pnu_loss(self.t, self.y)
