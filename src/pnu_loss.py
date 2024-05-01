import torch
import torch.nn as nn
from torch import Tensor


class PNULoss(nn.Module):
    def __init__(self, p_ratio: Tensor, eta: float, loss_func: nn.Module = nn.Sigmoid()) -> None:
        super().__init__()
        """
        Args:
            p_ratio: 学習用データにおける正例の割合
            eta: -1~1の値をとるハイパーパラメータ
            loss_func: 損失として用いる関数 DefaultはSigmoid
        """
        self.p_ratio = p_ratio
        self.eta = eta
        self.loss_func = loss_func

    def t_P_index(self, t: Tensor) -> Tensor:
        return torch.maximum(t, torch.zeros_like(t))

    def t_N_index(self, t: Tensor) -> Tensor:
        return torch.maximum(-t, torch.zeros_like(t))

    def t_U_index(self, t: Tensor) -> Tensor:
        return torch.ones_like(t) - torch.abs(t)

    def Risk(self, index: Tensor, y: Tensor) -> Tensor:
        n = torch.max(torch.tensor([1, torch.sum(index).item()]))
        k = torch.sum(torch.mul(index, self.loss_func(-y)))
        return torch.div(k, n)

    def Risk_P_plus(self, t: Tensor, y: Tensor) -> Tensor:
        return self.Risk(self.t_P_index(t), y)

    def Risk_P_minus(self, t: Tensor, y: Tensor) -> Tensor:
        return self.Risk(self.t_P_index(t), -y)

    def Risk_N_plus(self, t: Tensor, y: Tensor) -> Tensor:
        return self.Risk(self.t_N_index(t), y)

    def Risk_N_minus(self, t: Tensor, y: Tensor) -> Tensor:
        return self.Risk(self.t_N_index(t), -y)

    def Risk_U_plus(self, t: Tensor, y: Tensor) -> Tensor:
        return self.Risk(self.t_U_index(t), y)

    def Risk_U_minus(self, t: Tensor, y: Tensor) -> Tensor:
        return self.Risk(self.t_U_index(t), -y)

    def Risk_PN(self, t: Tensor, y: Tensor) -> Tensor:
        assert self.p_ratio.shape == torch.Size([2]), f"p_ratio must be shape as {torch.Size([2])}"

        return self.p_ratio[0] * self.Risk_P_plus(t, y) + self.p_ratio[1] * self.Risk_N_minus(t, y)

    def Risk_PU(self, t: Tensor, y: Tensor) -> Tensor:
        return self.p_ratio[0] * (self.Risk_P_plus(t, y) - self.Risk_P_minus(t, y)) + self.Risk_U_minus(t, y)

    def Risk_NU(self, t: Tensor, y: Tensor) -> Tensor:
        return self.p_ratio[1] * (self.Risk_N_minus(t, y) - self.Risk_N_plus(t, y)) + self.Risk_U_plus(t, y)

    def forward(self, outputs: Tensor, targets: Tensor) -> Tensor:
        """
        Args:
            t: 正例なら1, 負例なら-1, ラベルなしなら0をとるターゲット
            y: モデルの出力
        """
        targets = targets.flatten()  # (_, 1) -> (_)
        outputs = outputs.flatten()  # (_, 1) -> (_)
        if self.eta >= 0:
            return (1 - self.eta) * self.Risk_PN(targets, outputs) + self.eta * self.Risk_PU(targets, outputs)
        else:
            return (1 + self.eta) * self.Risk_PN(targets, outputs) - self.eta * self.Risk_NU(targets, outputs)
