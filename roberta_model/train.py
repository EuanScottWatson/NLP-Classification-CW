import argparse
import json

import pytorch_lightning as pl
import src.data_loaders as module_data
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from src.utils import get_model_and_tokenizer
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import transformers
from transformers.optimization import (
    Adafactor,
    AdafactorSchedule,
)


optimizer_table = {
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "AdaFactor": Adafactor,
}


lr_scheduler_table = {
    "ConstantWithWarmup": transformers.get_constant_schedule_with_warmup,
    "CosineWithWarmup": transformers.get_cosine_schedule_with_warmup,
    "LinearWithWarmup": transformers.get_linear_schedule_with_warmup,
    "CosineWithHardRestarts": transformers.get_cosine_with_hard_restarts_schedule_with_warmup,
    "AdafactorScedule": AdafactorSchedule,
}


class PatronisingClassifier(pl.LightningModule):
    """
    Args:
        config ([dict]): takes in args from a predefined config
                              file containing hyperparameters.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = config["arch"]["args"]["num_classes"]
        self.model_args = config["arch"]["args"]
        self.model, self.tokenizer = get_model_and_tokenizer(
            self.model_args["model_type"],
            self.model_args["model_name"],
            self.model_args["tokenizer_name"],
            self.model_args["num_classes"],
        )

        self.bias_loss = False
        self.loss_weight = config["loss_weight"]
        self.config = config
        self.scheduler_present = False

    def forward(self, x):
        inputs = self.tokenizer(
            list(x), return_tensors="pt", truncation=True, padding=True
        ).to(self.model.device)
        outputs = self.model(**inputs)[0]
        return outputs

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y["target"]
        output = self.forward(x)
        loss = nn.BCEWithLogitsLoss()(output, y.float())
        self.log("train_loss", loss)

        if self.scheduler_present:
            sch = self.lr_schedulers()
            sch.step()

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y["target"]
        output = self.forward(x)
        loss = nn.BCEWithLogitsLoss()(output, y.float())
        # using binary accuracy instead
        acc = self.binary_accuracy(output, y.float())
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return {"loss": loss, "acc": acc}

    def test_step(self, batch, batch_idx):
        x, y = batch
        output = self.forward(x)
        loss = nn.BCEWithLogitsLoss()(output, y.float())
        # using binary accuracy instead
        acc = self.binary_accuracy(output, y.float())
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        return {"loss": loss, "acc": acc}

    def binary_accuracy(self, output, target):
        """Custom binary_accuracy function.

        Args:
            output ([torch.tensor]): model predictions
            meta ([dict]): meta dict of tensors including targets and weights

        Returns:
            [torch.tensor]: model accuracy
        """
        with torch.no_grad():
            pred = torch.sigmoid(output) >= 0.5
            correct = torch.sum(pred.to(output.device) == target)
            if len(target) != 0:
                correct = correct.item() / len(target)
            else:
                correct = 0

        return torch.tensor(correct)

    def configure_optimizers(self):
        optimizer_type = self.config["optimizer"]["type"]
        optimizer_fn = optimizer_table.get(optimizer_type)

        optimizer = optimizer_fn(self.parameters(), **self.config["optimizer"]["args"])

        if self.config["optimizer"].get("lr_scheduler") is not None:
            self.scheduler_present = True
            lr_scheduler_type = self.config["optimizer"]["lr_scheduler"]["type"]
            print(f"Using {lr_scheduler_type} scheduler")
            lr_scheduler_fn = lr_scheduler_table.get(lr_scheduler_type)
            lr_scheduler = lr_scheduler_fn(
                optimizer, **self.config["optimizer"]["lr_scheduler"]["args"]
            )
            return [optimizer], [lr_scheduler]

        return optimizer


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
    parser.add_argument(
        "-e", "--n_epochs", default=100, type=int, help="if given, override the num"
    )

    args = parser.parse_args()
    print(f"Opening config {args.config}...")
    config = json.load(open(args.config))

    if args.device is not None:
        config["device"] = args.device

    # data
    def get_instance(module, name, config, *args, **kwargs):
        return getattr(module, config[name]["type"])(
            *args, **config[name]["args"], **kwargs
        )

    print("Fetching datasets")
    train_dataset = get_instance(module_data, "dataset", config)
    val_dataset = get_instance(module_data, "dataset", config, mode="VALIDATION")

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        shuffle=False,  # Deterministic
    )

    print(f"Batch size: {config['batch_size']}")
    print("Dataset loaded")
    print(f"\tTrain size: {len(train_data_loader)}")
    print(f"\tValidation size: {len(val_data_loader)}")

    # model
    model = PatronisingClassifier(config)

    print("Model created")

    # training
    checkpoint_callback = ModelCheckpoint(
        save_top_k=100,
        verbose=True,
        monitor="val_loss",
        mode="min",
    )

    callbacks = [checkpoint_callback]
    if config["arch"]["args"]["early_stop"]:
        print("Implementing Early Stop")
        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=3, verbose=False, mode="min"
        )
        callbacks.append(early_stop_callback)

    print("Training started...")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=int(config.get("n_gpu", 1)),
        gpus=args.device,
        max_epochs=args.n_epochs,
        accumulate_grad_batches=config["accumulate_grad_batches"],
        callbacks=callbacks,
        resume_from_checkpoint=args.resume,
        default_root_dir="/vol/bitbucket/es1519/NLPClassification_01/roberta_model/saved/" + config["name"],
        deterministic=True,
    )
    trainer.fit(model, train_data_loader, val_data_loader)


if __name__ == "__main__":
    cli_main()
