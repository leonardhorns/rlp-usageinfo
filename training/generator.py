import torch
from typing import List, Optional
from typing import Union

from training import utils
from helpers.review_set import ReviewSet
from training.utils import get_config
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_GENERATION_CONFIG = "diverse_beam_search"

GENERATION_CONFIGS = [
    file_name.replace(".yml", "")
    for file_name in os.listdir(
        os.path.dirname(os.path.realpath(__file__)) + "/generation_configs/"
    )
]


class Generator:
    def __init__(
        self,
        artifact_name,
        generation_config: str = DEFAULT_GENERATION_CONFIG,
        checkpoint: Optional[Union[int, str]] = None,
        prompt_id="original",
    ) -> None:
        global device

        if artifact_name in utils.model_tuples.keys():
            self.prompt_id = prompt_id
            self.model_artifact = artifact_name

        else:
            self.prompt_id = utils.load_config_from_artifact_name(artifact_name)[
                "prompt_id"
            ]
            self.model_artifact = {"name": artifact_name, "checkpoint": checkpoint}

        (
            self.model,
            self.tokenizer,
            self.max_length,
            self.model_name,
        ) = utils.initialize_model_tuple(self.model_artifact)

        self.model.to(device)
        self.model.eval()
        self.generation_config = get_config(
            f"{os.path.dirname(os.path.realpath(__file__))}/generation_configs/{generation_config}.yml"
        )

    def format_usage_options(self, text_completion: str) -> List[str]:
        if text_completion.lower() == "no usage options":
            return []
        return [
            usage_option.strip()
            for usage_option in text_completion.split("; ")
            if usage_option.strip()
        ]

    def generate_usage_options(self, batch) -> None:
        # batch is Iterable containing [List of model inputs, List of labels, List of review_ids]
        review_ids = list(batch["review_id"])
        input_ids = batch["input"]["input_ids"].to(device)
        attention_mask = batch["input"]["attention_mask"].to(device)
        model_inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **self.generation_config,
            )

        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions = [
            self.format_usage_options(usage_options) for usage_options in predictions
        ]

        return zip(review_ids, model_inputs, predictions)

    def generate_label(
        self, reviews: ReviewSet, label_id: str = None, verbose: bool = False
    ) -> None:
        if not label_id and not verbose:
            raise ValueError(
                "Specify either a label_id to save the labels or set verbose to True. (Or both)"
            )

        dataloader, _ = reviews.get_dataloader(
            batch_size=32,
            num_workers=0,
            tokenizer=self.tokenizer,
            model_max_length=self.max_length,
            for_training=False,
            prompt_id=self.prompt_id,
        )

        label_metadata = {
            "generator": {
                "model_name": self.model_name,
                "generation_config": self.generation_config,
                "prompt_id": self.prompt_id,
            }
        }
        if self.model_name != self.model_artifact:
            label_metadata["generator"].update(
                {
                    "artifact_name": self.model_artifact["name"],
                    "checkpoint": self.model_artifact["checkpoint"]
                    if self.model_artifact["checkpoint"] is not None
                    else "last",
                }
            )

        if verbose:
            print(f"Generating label {label_id}...")
            print(f"Label Metadata: {label_metadata}", end="\n\n")

        for batch in dataloader:
            usage_options_batch = self.generate_usage_options(batch)
            for review_id, model_input, usage_options in usage_options_batch:
                if label_id is not None:
                    reviews[review_id].add_label(
                        label_id=label_id,
                        usage_options=usage_options,
                        metadata=label_metadata,
                    )
                if verbose:
                    print(f"Review {review_id}")
                    print(f"Model input:\n{model_input}")
                    print(f"Usage options:\n\t{usage_options}", end="\n\n")
