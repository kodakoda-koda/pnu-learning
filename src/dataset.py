import random

import pandas as pd
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, dataset: dict) -> None:
        super().__init__()
        """
        Args:
            dataset: 辞書型のデータセット
        """
        self.dataset = dataset

    def __getitem__(self, index):
        input_ids = self.dataset["input_ids"][index]
        attention_mask = self.dataset["attention_mask"][index]
        labels = self.dataset["labels"][index]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    def __len__(self):
        return len(self.dataset["input_ids"])


def data_preprocess(train_df, test_df, tokenizer, unlabel_rate):
    # データの結合
    df = pd.concat([train_df, test_df], axis=0)
    df = df.dropna(subset=["Title", "Content"])
    df = pd.concat([df, pd.get_dummies(df["Label 1"], dtype=int)], axis=1)
    df = df.drop(["Target Organization", "Label 2", "Label 3", "Label 4", "Label 5"], axis=1)
    df["input"] = df["Title"].str.cat(df["Content"], sep=".\n")

    # データのトークン化
    xy_ids = tokenizer.batch_encode_plus(
        list(df["input"]), truncation=True, max_length=512, padding="max_length", return_tensors="pt"
    )

    # ラベル数の多いものを抽出
    labels = torch.tensor(df.drop(["Title", "Content", "Label 1", "input"], axis=1).values)
    top_indices = torch.argsort(labels.sum(dim=0), descending=True)[:2]
    indices = [i for i, _ in enumerate(labels.argmax(dim=1)) if _ in top_indices]
    xy_ids["labels"] = labels

    for k in xy_ids.keys():
        xy_ids[k] = xy_ids[k][indices]

    xy_ids["labels"] = xy_ids["labels"][:, top_indices]
    xy_ids["labels"][xy_ids["labels"] == 0] = -1

    # データセットの分割
    train_xy_ids = {k: v[: int(len(xy_ids["labels"]) * 0.9)] for k, v in xy_ids.items()}
    test_xy_ids = {k: v[int(len(xy_ids["labels"]) * 0.9) :] for k, v in xy_ids.items()}

    dataset = {"train": train_xy_ids, "test": test_xy_ids}

    # アンラベル化
    unlabel_indices = [
        False for _ in range(len(train_xy_ids["labels"]) - int(len(train_xy_ids["labels"]) * unlabel_rate))
    ] + [True for _ in range(int(len(train_xy_ids["labels"]) * unlabel_rate))]
    random.shuffle(unlabel_indices)

    dataset["train"]["labels"][unlabel_indices] = torch.zeros([sum(unlabel_indices), 2], dtype=int)

    # 正例率
    p_ratio = (dataset["train"]["labels"] == 1).sum(dim=0) / len(dataset["train"]["labels"])

    return dataset, p_ratio
