import argparse
import hashlib
from collections import OrderedDict

import torch


def main():
    """Converts saved checkpoint to the expected format for detoxify."""
    print(f"Loading checkpoint {ARGS.checkpoint}")
    checkpoint = torch.load(ARGS.checkpoint, map_location=ARGS.device)

    new_state_dict = {
        "state_dict": OrderedDict(),
        "config": checkpoint["hyper_parameters"]["config"],
    }
    for k, v in checkpoint["state_dict"].items():
        new_state_dict["state_dict"][k] = v

    print(f"Saving to {ARGS.save_to}")
    torch.save(new_state_dict, ARGS.save_to)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="path to model checkpoint",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        help="path to save the model to",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device to load the checkpoint on",
    )
    ARGS = parser.parse_args()
    main()
