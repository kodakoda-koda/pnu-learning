import pytest
import torch

from src.dataset import Get_DataLoader


class TestDataset:
    get_dataloader = Get_DataLoader(n_classes=2, batch_size=2, model_name="bert-base-uncased")
    multi_get_dataloader = Get_DataLoader(n_classes=3, batch_size=2, model_name="bert-base-uncased")
    train_loader = get_dataloader.get_loader("train")
    multi_train_loader = multi_get_dataloader.get_loader("train")

    def test_labels_shape(self):
        get_dataloader = Get_DataLoader(n_classes=2, batch_size=2, model_name="bert-base-uncased")
        multi_get_dataloader = Get_DataLoader(n_classes=3, batch_size=2, model_name="bert-base-uncased")
        train_loader = get_dataloader.get_loader("train")
        multi_train_loader = multi_get_dataloader.get_loader("train")

        batch = train_loader.__iter__().__next__()
        multi_batch = multi_train_loader.__iter__().__next__()

        assert batch["labels"].shape == torch.Size([2])
        assert multi_batch["labels"].shape == torch.Size([2, 3])
