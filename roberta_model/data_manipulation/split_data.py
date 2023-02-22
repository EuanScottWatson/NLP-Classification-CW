import random
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
import math

# Set the random seed for reproducibility
random.seed(123)

def split_data(all_data_path, train_path, validation_path, test_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Read in the CSV file as a Pandas DataFrame
    df = pd.read_csv(all_data_path)

    assert math.isclose((train_ratio + val_ratio + test_ratio), 1), "Data ratios must add up to 1"

    test_ratio_total = 1 - train_ratio
    test_ratio_only = test_ratio / test_ratio_total

    # Split the data into train, validation, and test sets
    train, test = train_test_split(df, test_size=test_ratio_total)
    val, test = train_test_split(test, test_size=test_ratio_only)

    # Write the train, validation, and test sets to separate CSV files
    train.to_csv(train_path, index=False)
    val.to_csv(validation_path, index=False)
    test.to_csv(test_path, index=False)

    print("{: >10}: {: >5} samples".format("Train", len(train)))
    print("{: >10}: {: >5} samples".format("Validation", len(val)))
    print("{: >10}: {: >5} samples".format("Test", len(test)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all",
        default=None,
        type=str,
        help="File containing all the data",
    )
    parser.add_argument(
        "--train",
        default=None,
        type=str,
        help="Where to save the training data",
    )
    parser.add_argument(
        "--val",
        default=None,
        type=str,
        help="Where to save the validation data",
    )
    parser.add_argument(
        "--test",
        default=None,
        type=str,
        help="Where to save the testing data",
    )
    args = parser.parse_args()

    split_data(args.all, args.train, args.val, args.test)