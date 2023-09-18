#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from evaluation.scoring.core import gpt_predictions_to_labels_from_file
from evaluation.scoring.metrics import Metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json",
        "-j",
        required=True,
        help="JSON structure v1",
    )
    parser.add_argument(
        "--ref-id",
        "-r",
        help="predictions will be scored with respect to this reference",
    )
    parser.add_argument(
        "--pred-id",
        "-p",
        default="[ALL]",
        help="labels from this prediction source will be scored; use '[ALL]' "
        + "or leave this argument unspecified to score all labels (except from ref-id) ",
    )
    parser.add_argument(
        "--save-path",
        "-s",
        help="location and file name of JSON output",
    )
    args = parser.parse_args()

    # load data
    with open(args.json) as file:
        data = json.load(file)

    # compute scores
    data_with_scores, _ = Metrics(data, args.pred_id, args.ref_id).calculate(
        [
            "custom_precision",
            "custom_recall",
            "custom_f1_score",
        ]
    )

    # save computed scores to file
    with open(args.save_path, "w") as json_file:
        json.dump(data_with_scores, json_file)
