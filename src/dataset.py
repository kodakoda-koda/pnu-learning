from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class Get_DataLoader:
    def __init__(self, n_classes: int, batch_size: int, model_name: str) -> None:
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.model_name = model_name

        self.__read_dataset__()

    def __read_dataset__(self) -> None: # クラス数を制御できるように
        dataset = load_dataset("knowledgator/events_classification_biotech")
        self.classes = [i[0] for i in dataset["all_labels"]]
        self.class2id = {class_: id for id, class_ in enumerate(self.classes)}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        tokenized_dataset = dataset.map(self.preprocess_function)
        # classを多い方からn_classesだけとる
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])
        # 99%をunlabel化
        # unlabelも含むlabelとそうでないlabelも両方作る
        self.dataset = tokenized_dataset

    def preprocess_function(self, example): # 型を確認
        text = f"{example['title']}.\n{example['content']}"
        labels = [0.0 for _ in range(len(self.classes))]
        label_id = self.class2id[example["all_labels"][0]]
        labels[label_id] = 1.0

        example = self.tokenizer(text, truncation=True, max_length=512, padding="max_length")
        example["labels"] = labels
        return example

    def get_loader(self):
        train_loader = DataLoader(self.dataset['train'], batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(self.dataset['test'], batch_size=self.batch_size, shuffle=True)
        return train_loader, test_loader
