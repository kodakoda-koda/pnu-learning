import torch
import torch.nn as nn
from torch import Tensor


class PNULoss:
    def __init__(self, p_ratio: float, eta: float, loss_func: nn.Module = nn.Sigmoid) -> None:
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
        return self.p_ratio * self.Risk_P_plus(t, y) + (1 - self.p_ratio) * self.Risk_N_minus(t, y)

    def Risk_PU(self, t: Tensor, y: Tensor) -> Tensor:
        return self.p_ratio * (self.Risk_P_plus(t, y) - self.Risk_P_minus(t, y)) + self.Risk_U_minus(t, y)

    def Risk_NU(self, t: Tensor, y: Tensor) -> Tensor:
        return (1 - self.p_ratio) * (self.Risk_N_minus(t, y) - self.Risk_N_plus(t, y)) + self.Risk_U_plus(t, y)

    def Risk_PNU(self, t: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            t: 正例なら1, 負例なら-1, ラベルなしなら0をとるターゲット
            y: モデルの出力
        """
        t = t.flatten()  # (_, 1) -> (_)
        y = y.flatten()  # (_, 1) -> (_)
        if self.eta >= 0:
            return (1 - self.eta) * self.Risk_PN(t, y) + self.eta * self.Risk_PU(t, y)
        else:
            return (1 + self.eta) * self.Risk_PN(t, y) - self.eta * self.Risk_NU(t, y)


class Multi_PNULoss:
    def __init__(self, p_ratio: Tensor, eta: float, loss_func: nn.Module = nn.Sigmoid) -> None:
        """
        Args:
            p_ratio: 学習用データにおける各カテゴリの正例の割合
            eta: 0~1の値をとるハイパーパラメータ
        """
        self.p_ratio = p_ratio
        self.eta = eta
        self.loss_func = loss_func
        assert type(self.p_ratio) == torch.Tensor, "p_ratio must be Tensor"

    def Risk_P_plus(self, t: Tensor, y: Tensor) -> Tensor:
        y_plus = y.clone()
        y_plus[~t.to(torch.bool)] = 0
        n_plus = t.sum(dim=0)
        return self.loss_func(-y_plus).sum(dim=0) / n_plus

    def Risk_P_minus(self, t: Tensor, y: Tensor) -> Tensor:
        y_minus = 1 - y.clone()
        y_minus[~t.to(torch.bool)] = 0
        n_minus = t.sum(dim=0)
        return self.loss_func(-y_minus).sum(dim=0) / n_minus

    def Risk_U(self, t: Tensor, y: Tensor) -> Tensor:
        y_U = y.clone()
        y_U[t.sum(dim=1).to(torch.bool)] = 0
        n_U = len(t) - t.sum()
        return self.loss_func(-y_U).sum(dim=0) / n_U

    def Risk_PN(self, t: Tensor, y: Tensor) -> Tensor:
        assert self.p_ratio.shape == t.sum(dim=0).shape, f"p_ratio must be shape as {t.sum(dim=0).shape}"

        risk_PN = self.p_ratio * self.Risk_P_plus(t, y)
        return risk_PN.sum()

    def Risk_PU(self, t: Tensor, y: Tensor) -> Tensor:
        risk_PU = self.p_ratio * (self.Risk_P_plus(t, y) - self.Risk_P_minus(t, y)) + self.Risk_U(t, y)
        return risk_PU.sum()

    def Risk_PNU(self, t: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            t: 正例なら1, 負例なら0をとるターゲット # (_, C) C: カテゴリ数
            y: モデルの出力 # (_, C)
        """
        return (1 - self.eta) * self.Risk_PN(t, y) + self.eta * self.Risk_PU(t, y)
