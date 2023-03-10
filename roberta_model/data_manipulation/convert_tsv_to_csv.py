import csv
import argparse

def convert_tsv_to_csv(tsv_file_path, csv_file_path, skip=True):
    tsv_file = open(tsv_file_path)
    csv_file = open(csv_file_path, 'w')

    tsv_reader = csv.reader(tsv_file, delimiter='\t')
    csv_writer = csv.writer(csv_file)

    skip_range = 4 if skip else 1
    # Skip the first 4 lines of the TSV file as this is the disclaimer
    for i in range(skip_range):
        next(tsv_reader)

    # Write the column headers to the CSV file
    column_headers = ["par_id", "art_id", "keyword", "country_code", "text", "label"]
    csv_writer.writerow(column_headers)

    for row in tsv_reader:
        if not row[4]:
            continue
        label = int(row[-1])
        if label in [0, 1, 2]:
            label = 0
        elif label in [3, 4]:
            label = 1
        else:
            raise ValueError(f"Invalid label value: {label}")
        row[-1] = str(label)
        if skip:
            csv_writer.writerow(row)
        else:
            csv_writer.writerow(row[:5] + [row[-1]])

    tsv_file.close()
    csv_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tsv",
        default=None,
        type=str,
        help="File containing TSV data",
    )
    parser.add_argument(
        "--csv",
        default=None,
        type=str,
        help="File containing CSV data",
    )
    parser.add_argument(
        "--skip",
        action='store_false',
    )
    args = parser.parse_args()

    convert_tsv_to_csv(args.tsv, args.csv, args.skip)