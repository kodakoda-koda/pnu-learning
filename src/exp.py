from typing import Type

import numpy as np
import torch
from torch.utils.data import DataLoader


class Exp:
    def __init__(self, train_loader: DataLoader, test_loader: DataLoader, model) -> None:
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.train_losses = []
        self.test_losses = []
        self.model = model

    def train(self, n_epochs, optimizer):
        self.model.train()
        for epoch in range(n_epochs):
            losses = []
            for itr in self.train_loader:
                optimizer.zero_grad()

                input_ids = itr['input_ids']
                attention_mask = itr['attention_mask']
                labels = itr['labels']

                output = self.model(input_ids, attention_mask)
                loss = loss_func(labels, output)

                losses.append(loss.item())
                loss.backward()
                optimizer.step()

            self.train_losses.append(np.mean(losses))
            self.test()

    def test(self):
        self.model.eval()
        losses = []
        with torch.zero_grad:
            for itr in self.test_loader:
                input_ids = itr['input_ids']
                attention_mask = itr['attention_mask']
                labels = itr['labels']

                output = self.model(input_ids, attention_mask)

                loss = loss_func(labels, output)
                losses.append(loss)

        self.test_losses.append(np.mean(losses))
                
