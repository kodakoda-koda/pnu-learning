from typing import Tuple

from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer


class Get_DataLoader:
    def __init__(self, n_classes: int, batch_size: int, model_name: str) -> None:
        """
        Args:
            n_classes: 検証するカテゴリ数
            batch_size: バッチサイズ
            model_name: モデル名 Tokenizerに使用
        """
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.model_name = model_name

        self.__read_dataset__()

    def get_loader(self, d_set: str) -> DataLoader:
        loader = DataLoader(self.dataset[d_set], batch_size=self.batch_size, shuffle=True)
        return loader

    def __read_dataset__(self) -> None:
        dataset = load_dataset("knowledgator/events_classification_biotech")
        self.classes = set(i[0] for i in dataset["all_labels"])
        self.class2id = {class_: id for id, class_ in enumerate(self.classes)}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        tokenized_dataset = dataset.map(self.__preprocess__)
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])

        tokenized_dataset["train"] = self.__subset__(tokenized_dataset, "train")
        tokenized_dataset["test"] = self.__subset__(tokenized_dataset, "test")

        self.dataset = tokenized_dataset

    def __preprocess__(self, example: dict) -> dict:
        text = f"{example['title']}.\n{example['content']}"
        labels = [0.0 for _ in range(len(self.classes))]
        label_id = self.class2id[example["all_labels"][0]]
        labels[label_id] = 1.0

        example = self.tokenizer(text, truncation=True, max_length=512, padding="max_length")
        example["labels"] = labels
        return example

    def __subset__(self, dataset: Dataset, d_set: str) -> Subset:
        n_labels = dataset[d_set]["labels"].sum(dim=0)
        top_indices = []
        for _ in range(self.n_classes):
            i = n_labels.argmax().item()
            n_labels[i] = 0.0
            top_indices.append(i)
        indices = [i for i, _ in enumerate(dataset[d_set]["labels"].argmax(dim=1)) if _ in top_indices]
        return Subset(dataset[d_set], indices=indices)
