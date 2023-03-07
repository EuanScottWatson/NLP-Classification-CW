import argparse
import json
import pandas as pd
import sys

sys.path.insert(1, '/vol/bitbucket/es1519/NLPClassification_01/roberta_model/src')
sys.path.insert(1, '/vol/bitbucket/es1519/NLPClassification_01/roberta_model')

import os
import numpy as np
import src.data_loaders as module_data
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm
from train import PatronisingClassifier
from transformers import logging
logging.set_verbosity_error()


def test_classifier(config, dataset, checkpoint_path, device="cuda:0"):
    model = PatronisingClassifier(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # Sets to evaluation mode (disable dropout + batch normalisation)
    model.to(device)

    def get_instance(module, name, config, *args, **kwargs):
        return getattr(module, config[name]["type"])(*args, **config[name]["args"], **kwargs)

    config["dataset"]["args"]["test_csv_file"] = dataset

    print(f"Dataset: {dataset}")

    test_dataset = get_instance(module_data, "dataset", config, mode="TEST", inference=True)

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=10,
        shuffle=False,
    )

    preds = []
    for *items, meta in tqdm(test_data_loader):
        with torch.no_grad():
            out = model.forward(*items)
            sm = torch.sigmoid(out).cpu().detach().numpy()
        preds.extend((sm >= 0.5).astype(int))

    preds = np.stack(preds)

    return {
        "predictions": preds.tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-ckpt",
        "--checkpoint",
        type=str,
        help="path to a saved checkpoint",
    )
    parser.add_argument(
        "-d",
        "--device",
        default="cuda:0",
        type=str,
        help="device name e.g., 'cpu' or 'cuda' (default cuda:0)",
    )
    parser.add_argument(
        "-t",
        "--test_csv",
        default=None,
        type=str,
        help="path to test dataset",
    )
    args = parser.parse_args()
    config = json.load(open(args.config))

    if args.device is not None:
        config["gpus"] = args.device

    results = test_classifier(config, args.test_csv,
                                args.checkpoint, args.device)
    test_set_name = args.test_csv.split("/")[-1:][0]

    with open(args.checkpoint[:-4] + f"results_{test_set_name}.json", "w") as f:
        json.dump(results, f)

# LR=1e-5	X	X	X	70010	0.48531	0.30363	0.601	0.599	0.602	0.931	7
