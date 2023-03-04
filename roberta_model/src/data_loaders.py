import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


class DontPatronizeMePCL(Dataset):
    def __init__(self, train_csv_file, val_csv_file, test_csv_file, loss_weight=0.75,
                 classes=["toxic"], train=True):
        print("Loading data...")
        if train:
            self.data = self.load_data(train_csv_file)
        else:
            self.data = self.load_val(val_csv_file)

        print("Data loaded:")
        print(self.data)

        self.train = train
        self.classes = classes
        self.loss_weight = loss_weight

    def __getitem__(self, index):
        print("__getitem__ called")
        meta = {}
        entry = self.data[index]
        text_id = entry["par_id"]
        text = entry["text"]

        target_dict = {label: value for label,
                       value in entry.items() if label in self.classes}

        meta["multi_target"] = torch.tensor(
            list(target_dict.values()), dtype=torch.int32)
        meta["text_id"] = text_id

        return text, meta

    def __len__(self):
        return len(self.data)

    def load_data(self, csv_file):
        train_set_pd = pd.read_csv(csv_file)
        return train_set_pd
        # train_set = datasets.Dataset.from_pandas(train_set_pd)
        # return train_set

    def load_val(self, val_csv_file, add_labels=False):
        val_set = self.load_data(val_csv_file)
        return val_set
        # val_set = datasets.Dataset.from_pandas(val_set)
        # return val_set
