import argparse

import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, set_seed

from dataset import MyDataset, data_preprocess
from exp import Exp
from model import CustomModelForSequenceClassification
from pnu_loss import Multi_PNULoss, PNULoss


def main():
    set_seed(seed=42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="./data/")
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--use_multi_loss", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--unlabel_rate", type=float, default=0.1)
    parser.add_argument("--eta", type=float, default=0.2)
    args = parser.parse_args()

    train_df = pd.read_csv(args.data_path + "train.csv", index_col=0)
    test_df = pd.read_csv(args.data_path + "test.csv", index_col=0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    dataset, p_ratio = data_preprocess(train_df, test_df, tokenizer, args.n_classes, args.unlabel_rate)
    train_dataset = MyDataset(dataset["train"])
    test_dataset = MyDataset(dataset["test"])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = CustomModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=args.n_classes,
        output_attentions=False,
    ).to(device)

    if args.use_multi_loss:
        loss_func = Multi_PNULoss(p_ratio=p_ratio, eta=args.eta)
    else:
        loss_func = PNULoss(p_ratio=p_ratio, eta=args.eta)
    optimizer = Adam(model.parameters(), lr=args.lr)

    exp = Exp(
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        n_classes=args.n_classes,
        device=device,
    )

    for epoch in range(args.n_epochs):
        print(f"epoch: {epoch}")
        exp.train()
        exp.test()


if __name__ == "__main__":
    main()
