import argparse
import json

import pytorch_lightning as pl
import src.data_loaders as module_data
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from src.utils import get_model_and_tokenizer
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader


class PatronisingClassifier(pl.LightningModule):
    """
    Args:
        config ([dict]): takes in args from a predefined config
                              file containing hyperparameters.
    """

    def __init__(self, config):
        super().__init__()
        print(f"Batch size: {config['batch_size']}")
        self.save_hyperparameters()
        self.num_classes = config["arch"]["args"]["num_classes"]
        self.model_args = config["arch"]["args"]
        self.model, self.tokenizer = get_model_and_tokenizer(**self.model_args)
        self.bias_loss = False

        self.loss_weight = config["loss_weight"]

        self.config = config

        self.fc = nn.Linear(self.model.config.hidden_size, self.num_classes)

    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors="pt", truncation=True, padding=True).to(self.model.device)
        outputs = self.model(**inputs)[0]
        x = self.fc(outputs)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = nn.BCEWithLogitsLoss()(output.view(-1), y.float())
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = nn.BCEWithLogitsLoss()(output.view(-1), y.float())
        acc = self.binary_accuracy(output.view(-1), y.float())  # using binary accuracy instead
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return {"loss": loss, "acc": acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = nn.BCEWithLogitsLoss()(output.view(-1), y.float())
        acc = self.binary_accuracy(output.view(-1), y.float())  # using binary accuracy instead
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), **self.config["optimizer"]["args"])


def cli_main():
    pl.seed_everything(1234)

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: None)",
    )
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="number of workers used in the data loader (default: 10)",
    )
    parser.add_argument("-e", "--n_epochs", default=100, type=int, help="if given, override the num")

    args = parser.parse_args()
    print("Opening config...")
    config = json.load(open(args.config))

    if args.device is not None:
        config["device"] = args.device

    # data
    def get_instance(module, name, config, *args, **kwargs):
        return getattr(module, config[name]["type"])(*args, **config[name]["args"], **kwargs)

    print("Fetching datasets")
    dataset = get_instance(module_data, "dataset", config)
    val_dataset = get_instance(module_data, "dataset", config, train=False)

    print("Datasets fetched")

    data_loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    valid_data_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        shuffle=False, # Deterministic
    )

    print("Dataset loaded")

    # model
    model = PatronisingClassifier(config)

    print("Model created")

    # training
    print("Training started...")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=100,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    print("Checkpoint created")
    trainer = pl.Trainer(
        accelerator='cpu', 
        devices=2,
        gpus=args.device,
        max_epochs=args.n_epochs,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=args.resume,
        default_root_dir="/vol/bitbucket/es1519/detecting-hidden-purpose-in-nlp-models/detoxify/saved/" + config["name"],
        deterministic=True,
    )
    trainer.fit(model, data_loader, valid_data_loader)


if __name__ == "__main__":
    cli_main()
