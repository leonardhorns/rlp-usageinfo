import pandas as pd
import json
from pathlib import Path
from typing import Union


def extract_labelled_reviews_from_json(
    path: Union[Path, str],
    extraction_func: callable,
    label_coloumn_name: str = "label",
) -> pd.DataFrame:
    with open(path, "r") as file:
        data = json.load(file)
        df = pd.DataFrame(data["reviews"])
        df[label_coloumn_name] = df["label"].apply(extraction_func)
        if label_coloumn_name != "label":
            df.drop("label", axis=1, inplace=True)
    return df


def extract_reviews_with_usage_options_from_json(
    path: Union[Path, str], use_predicted_usage_options=False
) -> pd.DataFrame:
    extract_usage_options_list = None
    if use_predicted_usage_options:
        extract_usage_options_list = lambda row: [
            x["label"] for x in row["predictedUsageOptions"]
        ]
    else:
        extract_usage_options_list = lambda x: x["customUsageOptions"] + [
            " ".join(annotation["tokens"]) for annotation in x["annotations"]
        ]
    return extract_labelled_reviews_from_json(
        path, extract_usage_options_list, "usage_options"
    )
