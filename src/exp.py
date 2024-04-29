import numpy as np
import torch


class Exp:
    def __init__(self, train_loader, test_loader, model, loss_func, optimizer):
        """
        Args:
            *_loader: 訓練用または評価用のDataLoader
            model: モデル
            loss_func: 損失関数, PNULoss
        """
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.loss_func = loss_func
        self.optimizer = optimizer

        self.train_losses = []
        self.test_losses = []

    def train(self, epoch: int):
        self.model.train()
        losses = []
        for itr in self.train_loader:
            self.optimizer.zero_grad()

            input_ids = itr["input_ids"]
            attention_mask = itr["attention_mask"]
            labels = itr["labels"]

            outputs = self.model(input_ids, attention_mask)
            loss = self.loss_func(
                outputs,
                labels,
            )

            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

        self.train_losses.append(np.mean(losses))

        print(f"epoch: {epoch}")
        print(f"  train loss: {self.train_losses[-1]}")

    def test(self):
        self.model.eval()
        losses = []
        with torch.zero_grad:
            for itr in self.test_loader:
                input_ids = itr["input_ids"]
                attention_mask = itr["attention_mask"]
                labels = itr["labels"]

                outputs = self.model(input_ids, attention_mask)

                loss = self.loss_func(outputs, labels)
                losses.append(loss.item())

        self.test_losses.append(np.mean(losses))
        print(f"  test loss: {self.test_losses[-1]}")
        print("=" * 50)
