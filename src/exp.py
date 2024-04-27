import numpy as np
import torch
from torch import optim


class Exp:
    def __init__(self, train_loader, test_loader, model, loss_func):
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

        self.train_losses = []
        self.test_losses = []

    def train(self, n_epochs: int, optimizer: optim):
        self.model.train()
        for epoch in range(n_epochs):
            losses = []
            for itr in self.train_loader:
                optimizer.zero_grad()

                input_ids = itr["input_ids"]
                attention_mask = itr["attention_mask"]
                labels = itr["labels"]

                output = self.model(input_ids, attention_mask)
                loss = self.loss_func(labels, output)

                losses.append(loss.item())
                loss.backward()
                optimizer.step()

            self.train_losses.append(np.mean(losses))
            self.eval()

            print(f"epoch: {epoch}, train loss: {self.train_losses[-1]}, valid loss: {self.test_losses[-1]}")
            print("=" * 50)

    def eval(self):
        self.model.eval()
        losses = []
        with torch.zero_grad:
            for itr in self.test_loader:
                input_ids = itr["input_ids"]
                attention_mask = itr["attention_mask"]
                labels = itr["labels"]

                output = self.model(input_ids, attention_mask)

                loss = self.loss_func(labels, output)
                losses.append(loss)

        self.test_losses.append(np.mean(losses))
