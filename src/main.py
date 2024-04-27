import argparse

from torch.optim import Adam
from transformers import AutoModel, set_seed

from dataset import Get_DataLoader
from exp import Exp
from pnu_loss import Multi_PNULoss, PNULoss


def main():
    set_seed(seed=42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--use_multi_loss", action="store_true")
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    get_loader = Get_DataLoader(n_classe=args.n_classes, batch_size=args.batch_size, model_name=args.model_name)

    train_loader = get_loader.get_loader("train")
    test_loader = get_loader.get_loader("test")

    model = AutoModel.from_pretrained(args.model_name)
    loss_func = Multi_PNULoss if args.use_multi_loss else PNULoss
    optimizer = Adam(model.parameters(), lr=args.lr)

    exp = Exp(train_loader=train_loader, test_loader=test_loader, model=model, loss_func=loss_func)

    exp.train(
        n_epochs=args.n_epochs,
        optimizer=optimizer,
    )
