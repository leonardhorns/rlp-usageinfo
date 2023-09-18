#!/usr/bin/env python3
import copy
from datetime import datetime
import json
from typing import Union
from pathlib import Path
import sys, os

from helpers.review import Review


REVIEW_FIELDS_OTHER_THAN_LABELS = Review.review_attributes.union(
    {"review_id"}
).difference({"labels"})


def v0_to_v1(
    data_v0: dict,
    label_id: str,
    source: str = "labellingTool",
) -> dict:
    """Upgrades a JSON file in "v0" format (pre March 2023) to our more consistent v1 format

    Args:
    `label_id`: unique id for where the labels comes from, like "golden_v2" or "davinci_freddy_v1"
    `source` can be either "labellingTool" or "openAI" and will be used as key for metadata
    """

    data_v1 = {"version": 1, "reviews": []}

    for review_v0 in data_v0["reviews"]:
        review_v1 = {}
        review_metadata = {}

        for key in review_v0:
            if key in REVIEW_FIELDS_OTHER_THAN_LABELS:
                review_v1[key] = review_v0[key]
            elif key == "label":
                if source == "labellingTool":
                    usage_options = review_v0[key]["customUsageOptions"] + [
                        " ".join(annotation["tokens"])
                        for annotation in review_v0[key]["annotations"]
                    ]
                    for info in review_v0[key]:
                        if (
                            info != "replacementClasses"
                        ):  # this is a relict of the past...
                            review_metadata[info] = review_v0[key][info]
                elif source == "openAI":
                    usage_options = review_v0[key]["usageOptions"]  # I guess?
                else:
                    raise ValueError(
                        f"expected parameter 'source' to be labellingTool or openAI but got {source}"
                    )
                review_v1["labels"] = {
                    label_id: {"usageOptions": usage_options, "metadata": {}}
                }
            else:
                review_metadata[key] = review_v0[key]
        if "label" not in review_v0:
            review_v1["labels"] = {}
        else:
            review_v1["labels"][label_id]["metadata"][source] = review_metadata
        data_v1["reviews"].append(review_v1)

    return data_v1


def v1_to_v2(data_v1: dict) -> dict:
    """Upgrades a review dictionairy in v1 format to the improved v2 format"""

    timestamp = datetime.now().astimezone().isoformat()  # using ISO 8601

    data_v2 = {"version": 2, "reviews": {}}

    for review_v1 in data_v1["reviews"]:
        if "inspectionTime" in review_v1:
            del review_v1["inspectionTime"]
        if "workerId" in review_v1:
            del review_v1["workerId"]
        if "workerInspectionTime" in review_v1:
            del (review_v1["workerInspectionTime"],)

        review_id = review_v1["review_id"]

        review_v2 = copy.deepcopy(review_v1)
        del review_v2["review_id"]  # the review id will be the key

        for label_id in review_v2["labels"]:
            scores = review_v2["labels"][label_id]["metadata"].pop("scores", {})
            review_v2["labels"][label_id]["scores"] = scores
            datasets = review_v2["labels"][label_id]["metadata"].pop("datasets", [])
            review_v2["labels"][label_id]["datasets"] = datasets
            review_v2["labels"][label_id]["createdAt"] = timestamp
            review_v2["labels"][label_id].pop("source", None)

        data_v2["reviews"][review_id] = review_v2
    return data_v2


def v2_to_v3(data_v2: dict) -> dict:
    """Upgrades a review dictionairy in v2 format to the improved v3 format"""

    data_v3 = {"version": 3, "reviews": data_v2["reviews"]}

    for _, review in data_v3["reviews"].items():
        for label_id in review["labels"]:
            review["labels"][label_id]["datasets"] = {
                dataset: "train" for dataset in review["labels"][label_id]["datasets"]
            }
            review["labels"][label_id].pop("source", None)

    return data_v3


def v3_to_v4(data_v3: dict) -> dict:
    data_v4 = {"version": 4, "reviews": data_v3["reviews"]}

    for _, review in data_v3["reviews"].items():
        for label_id in review["labels"]:
            review["labels"][label_id]["augmentations"] = []

    return data_v4


def v4_to_v5(data_v4: dict) -> dict:
    data_v5 = {"version": 5, "reviews": data_v4["reviews"]}

    for review in data_v4["reviews"].values():
        for label in review["labels"].values():
            augmentations = label["augmentations"]
            label["augmentations"] = {}
            if augmentations:
                label["augmentations"]["legacy"] = augmentations

            label["datasets"] = list(label["datasets"].keys())

    return data_v5


AUTOMATIC_REVIEW_SET_UPGRADE_FUNCTIONS = {
    1: v1_to_v2,
    2: v2_to_v3,
    3: v3_to_v4,
    4: v4_to_v5,
}


def upgrade_json_version(
    old_json_path: Union[str, Path], new_json_path: Union[str, Path], label_id=None
) -> None:
    with open(old_json_path, "r") as file:
        data = json.load(file)

    if "version" not in data:
        # make version -1 to version 0
        if type(data) == list:
            data = {"reviews": data}

        data = v0_to_v1(
            data,
            input("Enter label_id for the label of v0 json structure: "),
            input("Enter label source (either 'labellingTool' or 'openAI')"),
        )
    data = upgrade_to_latest_version(data)

    with open(new_json_path, "w") as file:
        json.dump(data, file)


def upgrade_to_latest_version(data: dict) -> dict:
    while data["version"] in AUTOMATIC_REVIEW_SET_UPGRADE_FUNCTIONS:
        data = AUTOMATIC_REVIEW_SET_UPGRADE_FUNCTIONS[data["version"]](data)

    return data


if __name__ == "__main__":
    assert len(sys.argv) in [
        2,
        3,
    ], "wrong number of arguments supplied\nusage: python upgrade_json_files.py old_json_file target_file"
    assert os.path.exists(sys.argv[1]), f"file {sys.argv[1]} doesn't exist, aborting..."

    new_json_path = sys.argv[1]
    if len(sys.argv) == 3:
        assert not os.path.exists(
            sys.argv[2]
        ), f"file {sys.argv[2]} does already exist, aborting..."
        new_json_path = sys.argv[2]

    upgrade_json_version(old_json_path=sys.argv[1], new_json_path=new_json_path)
