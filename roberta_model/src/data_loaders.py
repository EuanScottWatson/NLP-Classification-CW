import datasets
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset


class DontPatronizeMePCL(Dataset):
    def __init__(self, train_csv_file, val_csv_file, test_csv_file, classes, loss_weight=0.75, mode="TRAIN"):
        print(f"Loading data: mode={mode}")
        if mode == "TRAIN":
            self.data = self.load_data(train_csv_file)
        elif mode == "VALIDATION":
            self.data = self.load_val(val_csv_file)
        elif mode == "TEST":
            self.data = self.load_test(test_csv_file)
        else:
            raise "Enter a correct usage mode: TRAIN, VALIDATION or TEST"

        self.train = (mode == "TRAIN")
        self.classes = classes
        self.loss_weight = loss_weight

    def __getitem__(self, index):
        meta = {}
        entry = self.data[index]
        text_id = entry["par_id"]
        text = entry["text"]
        target_dict = {label: entry[label] for label in self.classes}
        meta["target"] = torch.tensor(list(target_dict.values()), dtype=torch.int32)
        meta["text_id"] = text_id

        return text, meta

    def __len__(self):
        return len(self.data)

    def load_data(self, csv_file):
        train_set_pd = pd.read_csv(csv_file)
        train_set = datasets.Dataset.from_pandas(train_set_pd)
        return train_set

    def load_val(self, val_csv_file):
        val_set = self.load_data(val_csv_file)
        return val_set
    
    def load_test(self, test_csv_file):
        test_set = self.load_data(test_csv_file)
        return test_set
