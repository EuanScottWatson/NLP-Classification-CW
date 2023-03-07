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

def test_single_input(config, input, checkpoint_path, device="cuda:0"):
    model = PatronisingClassifier(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # Sets to evaluation mode (disable dropout + batch normalisation)
    model.to(device)
    with torch.no_grad():
        out = model.forward(input)
        sm = torch.sigmoid(out).cpu().detach().numpy()

    res_df = pd.DataFrame(sm, index=[input] if isinstance(
        input, str) else input, columns=["Patronising"]).round(5)
    print(res_df)


def test_folder_of_checkpoints(folder_path, config, test_csv, device):
    print(f"Testing checkpoints found in {folder_path}")
    checkpoint_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".ckpt"):
                checkpoint_path = os.path.join(root, file)
                checkpoint_paths.append(checkpoint_path)
    checkpoint_paths = sorted(checkpoint_paths)
    print(f"{len(checkpoint_paths)} checkpoints found")
    print("Testing...")

    checkpoint_results = {}

    for checkpoint_path in checkpoint_paths:
        print(f"Evaluating: {checkpoint_path}")
        _, file_name = os.path.split(checkpoint_path)

        results = test_classifier(config, test_csv,
                                  checkpoint_path, device)

        checkpoint_results[file_name] = results
    
    print(checkpoint_results)
    with open(folder_path + "_folder_results.json", "w") as f:
        json.dump(checkpoint_results, f)


def test_classifier(config, dataset, checkpoint_path, device="cuda:0", log=False):
    model = PatronisingClassifier(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # Sets to evaluation mode (disable dropout + batch normalisation)
    model.to(device)

    def get_instance(module, name, config, *args, **kwargs):
        return getattr(module, config[name]["type"])(*args, **config[name]["args"], **kwargs)

    config["dataset"]["args"]["test_csv_file"] = dataset

    print(f"Dataset: {dataset}")

    test_dataset = get_instance(module_data, "dataset", config, mode="TEST")

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=10,
        shuffle=False,
    )

    preds = []
    targets = []
    ids = []
    for *items, meta in tqdm(test_data_loader):
        ids += meta.get("text_id", None)
        targets += meta["target"]
        with torch.no_grad():
            out = model.forward(*items)
            sm = torch.sigmoid(out).cpu().detach().numpy()
        preds.extend((sm >= 0.5).astype(int))
    
    preds = np.stack(preds)
    targets = np.stack(targets)
    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds)
    rec = recall_score(targets, preds)
    f1 = f1_score(targets, preds)
    ids = [id.item() if id else None for id in ids]

    conf_matrix = confusion_matrix(targets, preds)
    print(conf_matrix)

    tn, fp, fn, tp = conf_matrix.ravel()
    print(f"tn={tn}, fp={fp}, fn={fn}, tp={tp}")

    print(f"Number of predicted 0s: {len([p for p in preds if p == 0])}")
    print(f"Number of actual 0s: {len([t for t in targets if t == 0])}")
    print(f"Number of predicted 1s: {len([p for p in preds if p == 1])}")
    print(f"Number of actual 1s: {len([t for t in targets if t == 1])}")
    
    print(f"Precision: {prec}")
    print(f"Accuracy: {acc}")
    print(f"Recall: {rec}")
    print(f"F1 score: {f1}")

    data_points = []
    for (id, target, prediction) in zip(ids, targets, preds):
        data_points.append({
            "id": int(id),
            "target": int(target[0]),
            "prediction": int(prediction[0])
        })

    print(data_points)

    if log:
        return {
            "data_points": data_points,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
        }

    return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
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
    parser.add_argument(
        "-i",
        "--input",
        default=None,
        type=str,
        help="Single input test"
    )
    parser.add_argument(
        "--folder",
        default=None,
        type=str,
        help="Folder of converted checkpoints"
    )

    args = parser.parse_args()
    config = json.load(open(args.config))

    if args.device is not None:
        config["gpus"] = args.device

    if args.folder:
        test_folder_of_checkpoints(args.folder, config, args.test_csv, args.device)
    elif args.input:
        test_single_input(config, args.input, args.checkpoint, args.device)
    else:
        results = test_classifier(config, args.test_csv,
                                  args.checkpoint, args.device, log=True)
        test_set_name = args.test_csv.split("/")[-1:][0]

        with open(args.checkpoint[:-4] + f"results_{test_set_name}.json", "w") as f:
            json.dump(results, f)
