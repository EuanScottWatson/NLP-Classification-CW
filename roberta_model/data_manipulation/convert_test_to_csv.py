import csv
import argparse


def convert_tsv_to_csv(tsv_file_path, csv_file_path):
    tsv_file = open(tsv_file_path)
    csv_file = open(csv_file_path, 'w')

    tsv_reader = csv.reader(tsv_file, delimiter='\t')
    csv_writer = csv.writer(csv_file)

    column_headers = ["art_id", "keyword", "country_code", "text"]
    csv_writer.writerow(column_headers)

    for row in tsv_reader:
        csv_writer.writerow(row[1:])

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
    args = parser.parse_args()

    convert_tsv_to_csv(args.tsv, args.csv)
