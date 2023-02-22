import datasets
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


class DontPatronizeMePCL(Dataset):
    """Base dataloader for the Jigsaw Toxic Comment Classification Challenges."""

    def __init__(self, train_csv_file, val_csv_file, test_csv_file, train=True):
        print(train_csv_file)
        print(val_csv_file)
        print(test_csv_file)
        if train:
            self.data = self.load_data(train_csv_file)
        else:
            self.data = self.load_val(val_csv_file)

        self.train = train

    def __getitem__(self, index):
        meta = {}
        entry = self.data[index]
        text_id = entry["id"]
        text = entry["comment_text"]

        target_dict = {label: value for label, value in entry.items() if label in self.classes}

        meta["multi_target"] = torch.tensor(list(target_dict.values()), dtype=torch.int32)
        meta["text_id"] = text_id

        return text, meta

    def __len__(self):
        return len(self.data)

    def load_data(self, csv_file):
        train_set_pd = pd.read_csv(csv_file)
        train_set = datasets.Dataset.from_pandas(train_set_pd)
        return train_set

    def load_val(self, val_csv_file, add_labels=False):
        val_set = self.load_data(val_csv_file)
        val_set = datasets.Dataset.from_pandas(val_set)
        return val_set
