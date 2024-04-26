from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer


class Get_DataLoader:
    def __init__(self, n_classes: int, batch_size: int, model_name: str) -> None:
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.model_name = model_name

        self.__read_dataset__()

    def __read_dataset__(self) -> None:
        dataset = load_dataset("knowledgator/events_classification_biotech")
        self.classes = set(i[0] for i in dataset["all_labels"])
        self.class2id = {class_: id for id, class_ in enumerate(self.classes)}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        tokenized_dataset = dataset.map(self.preprocess_function)
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])

        n_labels = tokenized_dataset["train"]["labels"].sum(dim=0)  # ここをもっとスマートに
        top_indices = []
        for _ in range(self.n_classes):
            i = n_labels.argmax().item()
            n_labels[i] = 0.0
            top_indices.append(i)
        indices = [i for i, _ in enumerate(tokenized_dataset["train"]["labels"].argmax(dim=1)) if _ in top_indices]
        tokenized_dataset["train"] = Subset(tokenized_dataset["train"], indices=indices)  # ちょっと間違ってる

        n_labels = tokenized_dataset["test"]["labels"].sum(dim=0)  # 冗長
        top_indices = []
        for _ in range(self.n_classes):
            i = n_labels.argmax().item()
            n_labels[i] = 0.0
            top_indices.append(i)
        indices = [i for i, _ in enumerate(tokenized_dataset["test"]["labels"].argmax(dim=1)) if _ in top_indices]
        tokenized_dataset["test"] = Subset(tokenized_dataset["test"], indices=indices)

        self.dataset = tokenized_dataset

    def preprocess_function(self, example):  # 型を確認
        text = f"{example['title']}.\n{example['content']}"
        labels = [0.0 for _ in range(len(self.classes))]
        label_id = self.class2id[example["all_labels"][0]]
        labels[label_id] = 1.0

        example = self.tokenizer(text, truncation=True, max_length=512, padding="max_length")
        example["labels"] = labels
        return example

    def get_loader(self):
        train_loader = DataLoader(self.dataset["train"], batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.dataset["test"], batch_size=self.batch_size, shuffle=True)
        return train_loader, test_loader
