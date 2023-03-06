import json
import matplotlib.pyplot as plt
import argparse
from collections import OrderedDict

def plot(json_path, save_path):

    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    formatted_data = {}
    for epoch_str, scores in data.items():
        epoch_string = epoch_str.split("-step")[0].split("=")[1]
        formatted_data[int(epoch_string)] = scores

    ordered_data = OrderedDict(sorted(formatted_data.items()))

    # Extract the scores for each metric
    acc_scores = [epoch_data['accuracy'] for epoch_data in ordered_data.values()]
    prec_scores = [epoch_data['precision'] for epoch_data in ordered_data.values()]
    rec_scores = [epoch_data['recall'] for epoch_data in ordered_data.values()]
    f1_scores = [epoch_data['f1_score'] for epoch_data in ordered_data.values()]

    # Plot the scores over time
    epochs = range(len(data))
    plt.plot(epochs, acc_scores, label='Accuracy')
    plt.plot(epochs, prec_scores, label='Precision')
    plt.plot(epochs, rec_scores, label='Recall')
    plt.plot(epochs, f1_scores, label='F1-score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title("Evaluation Metrics per Epoch")
    plt.legend()
    plt.savefig(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "--json",
        default=None,
        type=str,
        help="File that contains JSON data",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        help="Where to save the plot",
    )
    args = parser.parse_args()

    plot(args.json, args.save_to)