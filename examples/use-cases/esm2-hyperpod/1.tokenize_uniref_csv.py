import argparse
import datasets
from itertools import chain
import logging
import os
import transformers
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
        "--input_dir",
        type=str,
        default="/fsx/ubuntu/csv",
        help="Input dir for protein sequence csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/fsx/ubuntu/processed",
        help="Output dir for processed files",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=10000000,
        help="The number of samples used for a test set",
    )
    parser.add_argument(
        "--validation_size",
        type=int,
        default=50000,
        help="The number of samples used for a validation set",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=50000,
        help="The number of samples used for a test set",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        type=bool,
        default=True,
        help="Whether to pad all samples to `max_seq_length`. If False, will pad the samples dynamically when batching to the maximum length in the batch.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=8,
        help="The number of workers to use for the preprocessing.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="facebook/esm2_t30_150M_UR50D",
        help="Pretrained tokenizer name to use.",
    )
    parser.add_argument(
        "--line_by_line",
        type=bool,
        default=True,
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )

    args, _ = parser.parse_known_args()
    return args


def main(args):

    logging.info(f"Loading csv files from {args.input_dir}")
    extension = "csv"
    data_files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith(extension)
    ]

    raw_data = datasets.load_dataset(
        "csv",
        data_files=data_files,
        num_proc=args.preprocessing_num_workers,
    )

    raw_data = raw_data.remove_columns("id")

    logging.info(raw_data)

    logging.info("Splitting dataset")
    train_testvalid = raw_data["train"].train_test_split(
        train_size=args.train_size, test_size=args.validation_size + args.test_size
    )
    test_valid = train_testvalid["test"].train_test_split(
        train_size=args.validation_size, test_size=args.test_size
    )
    raw_data = datasets.DatasetDict(
        {
            "train": train_testvalid["train"],
            "validation": test_valid["train"],
            "test": test_valid["test"],
        }
    )
    del train_testvalid
    del test_valid

    raw_data.flatten_indices()

    logging.info("Saving splits to csv")

    for dir in ["train", "val", "test"]:
        path = os.path.join(args.output_dir, "csv/" + dir)
        if not os.path.exists(path):
            os.makedirs(path)

    raw_data["train"].to_csv(os.path.join(args.output_dir, "csv/train", "x000.csv"))
    raw_data["validation"].to_csv(os.path.join(args.output_dir, "csv/val", "x001.csv"))
    raw_data["test"].to_csv(os.path.join(args.output_dir, "csv/test", "x002.csv"))

    column_names = list(raw_data["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_name)

    if args.line_by_line == True:
        logging.info("Processing line by line")

        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line
                for line in examples[text_column_name]
                if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=args.max_seq_length,
                return_special_tokens_mask=True,
            )

        tokenized_datasets = raw_data.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            desc="Running tokenizer on dataset line_by_line",
        )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting them in smaller parts.
        # We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling (see below) is more
        # efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(
                examples[text_column_name], return_special_tokens_mask=True
            )

        tokenized_datasets = raw_data.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on every text in dataset",
        )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {
                k: list(chain(*examples[k])) for k in examples.keys()
            }
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < max_seq_length  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // args.max_seq_length) * args.max_seq_length
            # Split by chunks of max_len.
            result = {
                k: [
                    t[i : i + args.max_seq_length]
                    for i in range(0, total_length, args.max_seq_length)
                ]
                for k, t in concatenated_examples.items()
            }
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            desc=f"Grouping texts in chunks of {args.max_seq_length}",
        )
    arrow_output_path = os.path.join(args.output_dir, "arrow")
    tokenized_datasets.save_to_disk(arrow_output_path)

    return arrow_output_path


if __name__ == "__main__":
    args = parse_args()
    main(args)