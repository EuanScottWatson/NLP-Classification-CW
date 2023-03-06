import pandas as pd
from sklearn.utils import resample
import argparse


def upsample(input_data, output_file):
    # Read in the CSV file
    df = pd.read_csv(input_data)

    # Count the number of entries with a label of 0 and 1
    count_0 = len(df[df["label"] == 0])
    count_1 = len(df[df["label"] == 1])

    # Oversample the entries with a label of 1 to balance the dataset
    df_1 = df[df["label"] == 1]
    df_1_oversampled = resample(
        df_1, replace=True, n_samples=count_0, random_state=42)

    # Combine the original entries with the oversampled entries
    df_balanced = pd.concat([df[df["label"] == 0], df_1_oversampled])

    # Save the balanced dataset to a new CSV file
    df_balanced.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=None,
        type=str,
        help="File containing all the data",
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="Where to save the data",
    )
    args = parser.parse_args()

    upsample(args.input, args.output)
