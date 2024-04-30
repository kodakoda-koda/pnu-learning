import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm


class Exp:
    def __init__(self, train_loader, test_loader, model, loss_func, optimizer, n_classes, device):
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
        self.n_classes = n_classes
        self.device = device

        self.train_losses = []
        self.test_losses = []

    def train(self):
        self.model.train()
        losses = []
        for itr in tqdm(self.train_loader):
            self.optimizer.zero_grad()

            input_ids = itr["input_ids"].to(self.device)
            attention_mask = itr["attention_mask"].to(self.device)
            label = itr["labels"].to(self.device)

            output = self.model(input_ids, attention_mask).logits
            if self.n_classes == 2:
                label = label[:, 0]
                output = output[:, 0]

            loss = self.loss_func(output, label)
            losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

        self.train_losses.append(np.mean(losses))

        print(f"train loss: {self.train_losses[-1]}")

    def test(self):
        self.model.eval()
        outputs = []
        labels = []
        with torch.no_grad():
            for itr in tqdm(self.test_loader):
                input_ids = itr["input_ids"].to(self.device)
                attention_mask = itr["attention_mask"].to(self.device)
                label = itr["labels"].to(self.device)

                output = self.model(input_ids, attention_mask).logits
                if self.n_classes == 2:
                    label = label[:, 0]
                    output = output[:, 0]

                output = output.detach().cpu().numpy()
                label = label.detach().cpu().numpy()
                outputs.append(output)
                labels.append(label)

        outputs = np.concatenate(outputs)
        labels = np.concatenate(labels)
        outputs = self.round_(outputs)
        acc = accuracy_score(labels, outputs)
        print(f"test accuracy: {acc}")
        print("=" * 50)

    def round_(self, outputs):
        outputs[outputs >= 0.0] = 1
        outputs[outputs < 0.0] = -1
        return outputs
