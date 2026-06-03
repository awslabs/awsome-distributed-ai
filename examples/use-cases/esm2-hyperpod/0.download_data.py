import argparse
import boto3
import csv
import datasets
import logging
import os
import pyfastx
import random
import requests
import tempfile
import tqdm
from urllib.parse import urlparse

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def parse_args():
    """Parse the arguments."""
    logging.info("Parsing arguments")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max_records_per_partition",
        type=int,
        default=500000,
        help="Max number of sequence records per csv partition",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getcwd(),
        help="Output dir for processed files",
    )
    parser.add_argument(
        "--save_arrow",
        type=bool,
        default=False,
        help="Save Apache Arrow files to output dir?",
    )
    parser.add_argument(
        "--save_csv",
        type=bool,
        default=True,
        help="Save csv files to output dir?",
    )
    parser.add_argument(
        "--save_fasta",
        type=bool,
        default=False,
        help="Save FASTA file to output dir?",
    )
    parser.add_argument(
        "--save_parquet",
        type=bool,
        default=False,
        help="Save Apache Parquet files to output dir?",
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=True,
        help="Shuffle the records in each csv partition?",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz",
        help="Path to input .fasta or .fasta.gz file, e.g. s3://myfasta.fa, http://myfasta.fasta.gz, ~/myfasta.fasta, etc",
    )

    args, _ = parser.parse_known_args()
    return args


def main(args):
    """Transform fasta file into dataset"""

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tmp_dir = tempfile.TemporaryDirectory(dir=os.getcwd())

    logging.info("Downloading FASTA")
    fasta_dir = (
        os.path.join(args.output_dir, "fasta")
        if args.save_fasta
        else os.path.join(tmp_dir.name, "fasta")
    )
    fasta_path = download(args.source, fasta_dir)

    logging.info("Generating csv files")
    csv_dir = (
        os.path.join(args.output_dir, "csv")
        if args.save_csv
        else os.path.join(tmp_dir.name, "csv")
    )
    csv_path = fasta_to_csv(
        fasta_path, csv_dir, args.max_records_per_partition
    )

    if args.save_arrow or args.save_parquet:
        logging.info("Loading csv files into dataset")
        ds = datasets.load_dataset(
            "csv",
            data_dir=csv_path,
            num_proc=os.cpu_count(),
            cache_dir=os.path.join(tmp_dir.name, "dataset_cache"),
        )

        logging.info("Saving dataset in Arrow format")
        if args.save_arrow:
            ds.save_to_disk(os.path.join(args.output_dir, "arrow"))

        logging.info("Saving dataset in Parquet format")
        if args.save_parquet:
            for split in ds.keys():
                ds[split].to_parquet(
                    f"{os.path.join(args.output_dir, 'parquet')}/data.parquet"
                )

    tmp_dir.cleanup()
    logging.info("Save complete")
    return args.output_dir


def download(source: str, filename: str) -> str:
    logging.info(f"Downloading {source} to {filename}")
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if source.startswith("s3"):
        s3 = boto3.client("s3")
        parsed = urlparse(source, allow_fragments=False)
        bucket = parsed.netloc
        key = parsed.path[1:]
        total = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
        tqdm_params = {
            "desc": source,
            "total": total,
            "miniters": 1,
            "unit": "B",
            "unit_scale": True,
            "unit_divisor": 1024,
        }
        with tqdm.tqdm(**tqdm_params) as pb:
            s3.download_file(
                parsed.netloc,
                parsed.path[1:],
                filename,
                Callback=lambda bytes_transferred: pb.update(bytes_transferred),
            )
    elif source.startswith("http"):
        with open(filename, "wb") as f:
            with requests.get(source, stream=True) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))

                tqdm_params = {
                    "desc": source,
                    "total": total,
                    "miniters": 1,
                    "unit": "B",
                    "unit_scale": True,
                    "unit_divisor": 1024,
                }
                with tqdm.tqdm(**tqdm_params) as pb:
                    for chunk in r.iter_content(chunk_size=8192):
                        pb.update(len(chunk))
                        f.write(chunk)
    elif os.path.isfile(source):
        logging.info(f"{source} already exists")
    else:
        raise ValueError(f"Invalid source: {source}")
    return filename


def fasta_to_csv(
    fasta: str,
    output_dir: str = "csv",
    max_records_per_partition=2000000,
    shuffle=False,
) -> list:
    """Split a .fasta or .fasta.gz file into multiple .csv files."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print("Reading FASTA file")
    fasta_list = []
    fasta_idx = 0

    for i, seq in tqdm.tqdm(
        enumerate(pyfastx.Fasta(fasta, build_index=False, uppercase=True))
    ):
        fasta_list.append(seq)

        if (i + 1) % max_records_per_partition == 0:
            if shuffle:
                random.shuffle(fasta_list)
            fasta_idx = int(i / max_records_per_partition)
            _write_seq_record_to_csv(fasta_list, output_dir, fasta_idx)
            fasta_list = []
    else:
        _write_seq_record_to_csv(fasta_list, output_dir, fasta_idx + 1)
    return output_dir


def _write_seq_record_to_csv(content_list, output_dir, index):
    output_path = os.path.join(output_dir, f"x{str(index).rjust(3, '0')}.csv")
    logging.info(f"Writing {len(content_list)} records to {output_path}")

    with open(output_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(("id", "text"))
        writer.writerows(content_list)
    return None


if __name__ == "__main__":
    args = parse_args()
    main(args)