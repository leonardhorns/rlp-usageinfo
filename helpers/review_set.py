import asyncio
import functools
import json
import itertools
import random
from copy import copy, deepcopy
from functools import partial
from pathlib import Path
from statistics import mean, quantiles, variance
from typing import Callable, ItemsView, Iterable, Iterator, Optional, Union

from numpy import mean, var
import pandas as pd
import helpers.label_selection as ls
from evaluation.scoring import DEFAULT_METRICS
from evaluation.scoring.evaluation_cache import EvaluationCache
from helpers.review import Review
from helpers.worker import Worker
from evaluation.scoring.h_tests import h_test, permutation_harmonic


class ReviewSet:
    """A ReviewSet object holds a set of reviews and makes them easily accessible.

    Data can be loaded from and saved to JSON files in the appropriate format.
    """

    latest_version = 5

    def __init__(
        self, version: str, reviews: dict, save_path: Optional[str] = None
    ) -> "ReviewSet":
        """load data and make sure it is structured according to our latest JSON format"""
        self.version = version
        self.reviews = reviews

        self.validate()

        self.save_path = save_path

    def __len__(self) -> int:
        return len(self.reviews)

    def __eq__(self, other: "ReviewSet") -> bool:
        if self.version != other.version:
            return False

        for review in self:
            if review not in other:
                return False

        for review in other:
            if review not in self:
                return False

        return True

    def __contains__(self, obj: Union[str, Review]) -> bool:
        if isinstance(obj, Review):
            obj = obj.review_id
        return obj in self.reviews

    def __iter__(self) -> Iterator[Review]:
        yield from self.reviews.values()

    def __or__(self, other: "ReviewSet") -> "ReviewSet":
        return self.merge(other, allow_new_reviews=True, inplace=False)

    def __ior__(self, other: "ReviewSet") -> "ReviewSet":
        self.merge(other, allow_new_reviews=True, inplace=True)
        return self

    def __sub__(self, other: "ReviewSet") -> "ReviewSet":
        return self.set_minus(other)

    def __copy__(self):
        return self.from_reviews(*self)

    def __deepcopy__(self, memo):
        return self.from_reviews(*deepcopy(list(self), memo))

    def __str__(self) -> str:
        reviews = "{\n" + ",\n".join([str(review) for review in self]) + "}"
        return f"ReviewSet version {self.version}, reviews: {reviews}"

    def __getitem__(self, review_id: str) -> Review:
        if isinstance(review_id, slice):
            return [self.reviews[review_id] for review_id in list(self.reviews)][
                review_id
            ]
        else:
            return self.reviews[review_id]

    def __delitem__(self, review_id: str):
        del self.reviews[review_id]

    def __setitem__(self, review_id: str, review: Review):
        self.reviews[review_id] = review

    @classmethod
    def from_dict(cls, data: dict, save_path: Optional[str] = None) -> "ReviewSet":
        version = data.get("version", 0)
        if version < cls.latest_version:
            if version < 1:
                exit(
                    1,
                    "Automatic upgrade from version 0 is not supported.\nPlease resort to the manual upgrade process.",
                )
            print(
                f"Auto-upgrading your json review set to version {cls.latest_version} for usage (current version: {data.get('version')})...\nThis will not override your file unless you save this reviewset!"
            )
            from helpers.upgrade_json_files import upgrade_to_latest_version

            data = upgrade_to_latest_version(data)

        reviews = {
            review_id: Review(review_id=review_id, data=review_data)
            for review_id, review_data in data.get("reviews", {}).items()
        }

        return cls(data.get("version"), reviews, save_path)

    @classmethod
    def from_reviews(
        cls, *reviews: Review, save_path: Optional[str] = None
    ) -> "ReviewSet":
        return cls(
            cls.latest_version,
            {review.review_id: review for review in reviews},
            save_path,
        )

    @classmethod
    def from_files(
        cls, *source_paths: Union[str, Path], save_path: Optional[str] = None
    ) -> "ReviewSet":
        if len(source_paths) == 0:
            raise ValueError("Expected at least one source path argument")

        def get_review_set(path: Union[str, Path]):
            with open(path) as file:
                return cls.from_dict(json.load(file))

        review_sets = []

        for path in source_paths:
            absolute_path = Path(path).expanduser().resolve()
            if absolute_path.is_dir():
                for file in absolute_path.glob("*.json"):
                    review_sets.append(get_review_set(str(file)))
            elif absolute_path.is_file():
                review_sets.append(get_review_set(str(absolute_path)))

        review_set = functools.reduce(
            lambda review_set_1, review_set_2: review_set_1 | review_set_2, review_sets
        )

        review_set.save_path = (
            str(Path(source_paths[0]).expanduser().resolve())
            if len(source_paths) == 1
            else save_path
        )

        return review_set

    def items(self) -> ItemsView[str, Review]:
        return self.reviews.items()

    def add(self, review: Review, add_new=True) -> None:
        if review in self:
            self[review.review_id] |= review
        elif add_new:
            self.reviews[review.review_id] = review

    def get_review(self, review_id: str) -> Review:
        return self.reviews[review_id]

    def count_common_reviews(self, other: "ReviewSet") -> int:
        common_review_counter = 0
        for review in other:
            if review in self:
                common_review_counter += 1
        return common_review_counter

    def count_new_reviews(self, other: "ReviewSet") -> int:
        return len(other) - self.count_common_reviews(other)

    def set_minus(self, other: "ReviewSet") -> "ReviewSet":
        return ReviewSet.from_reviews(*(set(self) - set(other)))

    def get_all_label_ids(self) -> set:
        label_ids = set()
        for review in self:
            label_ids |= review.get_label_ids()
        return label_ids

    def remove_label(self, label_id: str, inplace=True) -> Optional["ReviewSet"]:
        review_set = self if inplace else deepcopy(self)
        for review in review_set:
            review.remove_label(label_id, inplace=True)

        if not inplace:
            return review_set

    def get_usage_options(self, label_id: str) -> list:
        usage_options = list(
            itertools.chain(*[review.get_usage_options(label_id) for review in self])
        )
        if not usage_options:
            raise ValueError(f"Label {label_id} not found in any review")
        return usage_options

    def score(
        self,
        label_id: str,
        reference_label_id: str,
        metric_ids: Iterable[str] = DEFAULT_METRICS,
    ):
        if len(metric_ids) > 0:
            for review in self.reviews_with_labels({label_id, reference_label_id}):
                review.score(label_id, reference_label_id, metric_ids)

        EvaluationCache.get().save_to_disk()  # save newly calculated scores to disk

    def get_harmonic_scores(
        self,
        label_id: Union[str, ls.LabelSelectionStrategyInterface],
        *reference_label_candidates: Union[str, ls.LabelSelectionStrategyInterface],
        metric_ids: Union[set, list] = DEFAULT_METRICS,
    ):
        scores = [
            review.get_scores(
                label_id, *reference_label_candidates, metric_ids=metric_ids
            )
            for review in self
        ]

        scores = list(filter(lambda x: x is not None, scores))

        score_dict = {
            metric_id: {"positives": [], "negatives": [], "true_positives": []}
            for metric_id in metric_ids
        }

        for score in scores:
            for metric in metric_ids:
                # score[metric] is a tuple (score, prediction_is_positive_usage, reference_is_positive_usage)
                if score[metric][2] == True:
                    score_dict[metric]["positives"].append(score[metric][0])
                    if score[metric][1] == True:
                        score_dict[metric]["true_positives"].append(score[metric][0])
                else:
                    score_dict[metric]["negatives"].append(score[metric][0])

        harmonic_scores = {}

        for metric in metric_ids:
            mean_positives = (
                mean(score_dict[metric]["positives"])
                if len(score_dict[metric]["positives"]) > 0
                else 1
            )
            mean_true_positives = (
                mean(score_dict[metric]["true_positives"])
                if len(score_dict[metric]["true_positives"]) > 0
                else 1
            )
            mean_negatives = (
                mean(score_dict[metric]["negatives"])
                if len(score_dict[metric]["negatives"]) > 0
                else 1
            )
            harmonic_scores[metric] = {
                "harmonic": 2
                * (mean_positives * mean_negatives)
                / (mean_positives + mean_negatives),
                "mean_positives": mean_positives,
                "mean_true_positives": mean_true_positives,
                "mean_negatives": mean_negatives,
            }

        return harmonic_scores

    def get_agg_scores(
        self,
        label_id: Union[str, ls.LabelSelectionStrategyInterface],
        *reference_label_candidates: Union[str, ls.LabelSelectionStrategyInterface],
        metric_ids: Union[set, list] = DEFAULT_METRICS,
    ) -> dict[dict[str, float]]:

        aggregations = {
            "mean": mean,
            "variance": variance,
            "quantiles (n=4)": quantiles,
        }

        scores = [
            review.get_scores(
                label_id, *reference_label_candidates, metric_ids=metric_ids
            )
            for review in self
        ]
        scores = list(filter(lambda x: x is not None, scores))
        if len(scores) < 2:
            raise ValueError(
                "At least two reviews are required to calculate aggregated scores"
            )
        agg_scores = {"num_reviews": len(scores)}
        for metric_id in metric_ids:
            agg_scores[metric_id] = {
                agg_name: agg_func([score[metric_id][0] for score in scores])
                for agg_name, agg_func in aggregations.items()
            }

        return agg_scores
    
    
    def _get_scored_reviews_dataframe(
        self,
        label_id: Union[str, ls.LabelSelectionStrategyInterface],
        *reference_label_ids: Union[str, ls.LabelSelectionStrategyInterface],
        metric_id: str = "custom_weighted_mean_f1",
    ) -> pd.DataFrame:
        reviews_list = []
        for review in self:
            review_scores = review.get_scores(
                label_id, *reference_label_ids, metric_ids=[metric_id]
            )
            if not review_scores:
                continue

            reference_labels = []
            no_usage_options_ok, usage_options_ok = False, False
            for reference_label_id in reference_label_ids:
                if issubclass(type(reference_label_id), ls.AbstractLabelSelectionStrategy):
                    reference_label = review.get_label_from_strategy(reference_label_id)
                else:
                    reference_label = review.get_label_for_id(reference_label_id)
                if reference_label:
                    reference_labels.append(reference_label["usageOptions"])
                    if len(reference_label["usageOptions"]) > 0:
                        usage_options_ok = True
                    else:
                        no_usage_options_ok = True

            if issubclass(type(label_id), ls.AbstractLabelSelectionStrategy):
                label_id = label_id.retrieve_label_id(review)
            prediction_has_usage_options = (
                len(review.get_label_for_id(label_id)["usageOptions"]) > 0
            )

            if prediction_has_usage_options:
                usage_class = "TP" if usage_options_ok else "FP"
            else:
                usage_class = "TN" if no_usage_options_ok else "FN"

            reviews_list.append(
                {
                    "review_id": review.review_id,
                    "review": f'Product title: {review.data["product_title"]}\nHeadline: {review.data["review_headline"]}\n{review.data["review_body"]}',
                    "review_body": review.data["review_body"],
                    "usage_class": usage_class,
                    "predicted_usage_options": "; ".join(
                        review.get_label_for_id(label_id)["usageOptions"]
                    ),
                    "reference_usage_options": "\n".join(
                        [
                            "; ".join(reference_labels[i])
                            for i in range(len(reference_labels))
                        ]
                    ),
                    "star_rating": review.data["star_rating"],
                    "product_category": review.data["product_category"],
                    metric_id: review_scores[metric_id],
                }
            )

        df = pd.DataFrame.from_records(reviews_list)
        # Correct star rating because some are saved as int vs some as str
        df["star_rating"] = df["star_rating"].apply(lambda x: int(x))
        return df

    def get_classification_scores(
        self,
        label_id: Union[str, ls.LabelSelectionStrategyInterface],
        *reference_label_candidates: Union[str, ls.LabelSelectionStrategyInterface],
    ) -> dict[dict[str, float]]:
        reviews_df = self._get_scored_reviews_dataframe(
            label_id, *reference_label_candidates
        )

        TP = len(reviews_df[reviews_df["usage_class"] == "TP"])
        FP = len(reviews_df[reviews_df["usage_class"] == "FP"])
        TN = len(reviews_df[reviews_df["usage_class"] == "TN"])
        FN = len(reviews_df[reviews_df["usage_class"] == "FN"])

        classification_score = {
            "accuracy": (TP + TN) / (TP + TN + FN + FP),
            "recall": TP / (TP + FN) if (TP + FN) > 0 else 0.0,
            "precision": TP / (TP + FP) if (TP + FP) > 0 else 0.0,
            "sensitivity": TN / (TN + FP) if (TN + FP) > 0 else 0.0,
            "specificity": TN / (TN + FP) if (TN + FP) > 0 else 0.0,
            "f1": (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0,
        }

        classification_score["count"] = {"TP": TP, "FP": FP, "TN": TN, "FN": FN}

        classification_score["balanced_accuracy"] = (
            classification_score["sensitivity"] + classification_score["specificity"]
        ) / 2

        return classification_score

    def test(
        self,
        label_id_1: Union[str, ls.LabelSelectionStrategyInterface],
        label_id_2: Union[str, ls.LabelSelectionStrategyInterface],
        *reference_label_candidates: Union[str, ls.LabelSelectionStrategyInterface],
        tests: list[str] = ["ttest"],
        alternatives: list[str] = ["greater"],
        metric_ids: Iterable[str] = DEFAULT_METRICS,
        confidence_level: float = 0.95,  # only used for bootstrap
    ):
        import numpy as np

        tests = (
            ["ttest", "wilcoxon", "bootstrap", "permutation"]
            if "all" in tests
            else tests
        )

        # iterate over all reviews and score label_id_1 against reference_label_candidates and label_id_2 against reference_label_candidates
        scores = {
            metric_id: {label_id_1: [], label_id_2: []} for metric_id in metric_ids
        }

        for review in self:
            score_1 = review.get_scores(
                label_id_1, *reference_label_candidates, metric_ids=metric_ids
            )
            score_2 = review.get_scores(
                label_id_2, *reference_label_candidates, metric_ids=metric_ids
            )

            if score_1 is None or score_2 is None:
                continue

            for metric_id in metric_ids:
                scores[metric_id][label_id_1].append(score_1[metric_id])
                scores[metric_id][label_id_2].append(score_2[metric_id])

        test_results = {}

        for test in tests:
            for metric_id in metric_ids:
                for alternative in alternatives:
                    if test == "permutation_harmonic":
                        test_results[
                            (test, metric_id, alternative)
                        ] = permutation_harmonic(
                            scores[metric_id][label_id_1],
                            scores[metric_id][label_id_2],
                            alternative,
                        )
                    else:
                        test_results[(test, metric_id, alternative)] = h_test(
                            test,
                            np.array(
                                [
                                    score_tuple[0]
                                    for score_tuple in scores[metric_id][label_id_1]
                                ]
                            ),
                            np.array(
                                [
                                    score_tuple[0]
                                    for score_tuple in scores[metric_id][label_id_2]
                                ]
                            ),
                            alternative=alternative,
                            confidence_level=confidence_level,
                        )

        EvaluationCache.get().save_to_disk()

        return test_results

    def reviews_with_labels(self, label_ids: set[str]) -> list[Review]:
        """Returns a review set containing only reviews with the given labels"""
        relevant_reviews = [
            review for review in self if label_ids <= review.get_label_ids()
        ]
        return self.from_reviews(*relevant_reviews)

    def validate(self) -> None:
        if self.version != self.latest_version:
            raise ValueError(
                f"only the latest format (v{self.latest_version})"
                "of our JSON format is supported"
            )

        for review in self:
            review.validate()

    def merge(
        self,
        review_set: "ReviewSet",
        allow_new_reviews: bool = False,
        inplace=False,
    ) -> Optional["ReviewSet"]:
        """Merges foreign ReviewSet into this ReviewSet

        Args:
            review_set (ReviewSet): foreign ReviewSet to merge into this one
            allow_new_reviews (bool, optional): if set to True, all unseen reviews from `review_set` will be added. Defaults to False.
            inplace (bool, optional): if set to True, overwrites object data; otherwise, creates a new ReviewSet object. Defaults to False.
        """

        assert (
            self.version == review_set.version == self.latest_version
        ), f"expected ReviewSets in latest format (v{self.latest_version})"

        merged_review_set = self if inplace else copy(self)

        for review in review_set:
            merged_review_set.add(review, add_new=allow_new_reviews)

        if not inplace:
            return merged_review_set

    def get_data(self) -> dict:
        """get data in correct format of the latest version"""
        result = {
            "version": self.version,
            "reviews": {review_id: review.data for review_id, review in self.items()},
        }
        return result

    def drop_review(
        self, obj: Union[str, Review], inplace=True
    ) -> Optional["ReviewSet"]:
        if isinstance(obj, Review):
            obj = obj.review_id

        if not inplace:
            reviews = deepcopy(self.reviews)
            reviews.pop(obj, None)
            return ReviewSet.from_dict({"version": self.version, "reviews": reviews})

        self.reviews.pop(obj, None)

    def filter(
        self, filter_function: Callable[[Review], bool], inplace=True, invert=False
    ) -> Optional["ReviewSet"]:
        reviews = self if inplace else copy(self)
        for review in copy(reviews):
            # if invert is True, we want to drop all reviews that match the filter function. Otherwise, we want to drop all reviews that do not match the filter function.
            if invert == bool(filter_function(review)):
                reviews.drop_review(review)

        if not inplace:
            return reviews

    def filter_with_label_strategy(
        self,
        selection_strategy: ls.LabelSelectionStrategyInterface,
        inplace=True,
        invert=False,
    ) -> Optional["ReviewSet"]:
        return self.filter(
            lambda review: review.get_label_from_strategy(selection_strategy),
            inplace=inplace,
            invert=invert,
        )

    def _split_by_label_has_usage_options(
        self,
        selection_strategy: ls.LabelSelectionStrategyInterface,
    ) -> tuple["ReviewSet", "ReviewSet"]:
        review_has_usage_options = lambda review: review.label_has_usage_options(
            selection_strategy
        )
        try:
            usage_option_reviews = self.filter(
                review_has_usage_options,
                inplace=False,
            )
            no_usage_option_reviews = self.filter(
                review_has_usage_options,
                inplace=False,
                invert=True,
            )
        except ValueError as e:
            raise ValueError(
                f"Not all reviews have labels for {selection_strategy}\n\t-> {e}"
            )
        return usage_option_reviews, no_usage_option_reviews

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(self.get_data()["reviews"], orient="index")

    def get_dataloader(
        self,
        tokenizer,
        model_max_length: int,
        for_training: bool,
        selection_strategy: ls.LabelSelectionStrategyInterface = None,
        multiple_usage_options_strategy: Optional[str] = None,
        prompt_id: str = "avetis_v1",
        drop_out: float = 0.0,
        stratified_drop_out: bool = False,
        seed: int = None,
        rng: random.Random = random,
        **dataloader_args: dict,
    ) -> tuple[any, dict]:
        def are_all_valid_data_points(data_points: list[dict]) -> bool:
            for datapoint in data_points:
                # If None is in the values of a datapoint, the tokenized input or label was too long for the model
                # If selection_strategy is specified the output should not be 0 which is used as the default value
                if None in datapoint.values() or (
                    selection_strategy is not None and 0 in datapoint.values()
                ):
                    return False
            return True

        from torch.utils.data import DataLoader

        if seed is not None:
            rng.seed(seed)

        # Unpack all normal reviews and their augmentations into single ReviewSet
        all_reviews = copy(self)
        for review in copy(all_reviews):
            review.tokenized_datapoints = list(
                review.get_tokenized_datapoints(
                    selection_strategy=selection_strategy,
                    tokenizer=tokenizer,
                    max_length=model_max_length,
                    for_training=for_training,
                    multiple_usage_options_strategy=multiple_usage_options_strategy,
                    prompt_id=prompt_id,
                )
            )
        valid_reviews = all_reviews.filter(
            lambda review: are_all_valid_data_points(review.tokenized_datapoints),
            inplace=False,
        )

        if stratified_drop_out and not selection_strategy:
            print(
                "Warning: Stratified drop out is only possible when a label is specified.\nResorting to simple random drop out."
            )
        dropped_reviews, remaining_reviews = (
            valid_reviews.stratified_split(drop_out, selection_strategy, rng=rng)
            if stratified_drop_out and selection_strategy
            else valid_reviews.split(drop_out, rng=rng)
        )

        tokenized_datapoints = [
            data_point
            for review in remaining_reviews
            for data_point in review.tokenized_datapoints
        ]
        rng.shuffle(tokenized_datapoints)
        return (
            DataLoader(tokenized_datapoints, **dataloader_args),
            {
                "selection_strategy": selection_strategy,
                "num_reviews": len(self),
                "num_augmented_reviews": len(all_reviews) - len(self),
                "num_invalid_reviews": len(all_reviews)
                - len(
                    valid_reviews
                ),  # invalid due to tokenized length or selection strategy
                "num_dropped_reviews": len(
                    dropped_reviews
                ),  # dropped due to random drop out
                "num_remaining_reviews": len(remaining_reviews),
                "num_datapoints": len(tokenized_datapoints),
            },
        )

    def reset_scores(self):
        for review in self:
            review.reset_scores()

    def split(
        self,
        fraction: float,
        seed: int = None,
        rng: random.Random = random,
    ) -> tuple["ReviewSet", "ReviewSet"]:
        if fraction < 0 or fraction > 1:
            raise ValueError("Fraction must be between 0 and 1")
        if fraction == 0:
            return ReviewSet.from_reviews(), ReviewSet.from_reviews(*self)
        if fraction == 1:
            return ReviewSet.from_reviews(*self), ReviewSet.from_reviews()

        if seed is not None:
            rng.seed(seed)

        reviews = sorted(list(self), key=lambda review: review.review_id)
        rng.shuffle(reviews)

        split_index = min(len(reviews) - 1, max(1, int(len(reviews) * fraction)))
        return (
            ReviewSet.from_reviews(*reviews[:split_index]),
            ReviewSet.from_reviews(*reviews[split_index:]),
        )

    def stratified_split(
        self,
        fraction: float,
        label_selection_strategy: ls.LabelSelectionStrategyInterface,
        seed: Optional[int] = None,
        rng: random.Random = random,
    ) -> tuple["ReviewSet", "ReviewSet"]:
        """Split the ReviewSet while keeping ratio of label classes.

        Args:
            fraction (float): Fraction of reviews to be in the first set.
            label_selection_strategy (ls.LabelSelectionStrategyInterface): Label selection strategy that specifies which label to use for stratification.
            seed (int, optional): Seed for random number generator. Defaults to None.
        """
        if seed is not None:
            rng.seed(seed)

        (
            usage_option_reviews,
            no_usage_option_reviews,
        ) = self._split_by_label_has_usage_options(label_selection_strategy)

        set1_usage, set2_usage = usage_option_reviews.split(fraction, rng=rng)
        set1_no_usage, set2_no_usage = no_usage_option_reviews.split(fraction, rng=rng)

        return set1_usage | set1_no_usage, set2_usage | set2_no_usage

    def create_dataset(
        self,
        dataset_name: str,
        label_selection_strategy: ls.LabelSelectionStrategyInterface,
    ) -> tuple["ReviewSet", dict]:
        reviews = deepcopy(
            self.filter_with_label_strategy(label_selection_strategy, inplace=False)
        )
        dataset_length = len(reviews)
        if dataset_length == 0:
            raise ValueError("There is no review that has any of the specified labels.")

        label_id_counts = {}
        for review in reviews:
            dataset_label_id = review.get_label_id_from_strategy(
                label_selection_strategy
            )
            dataset_label = review.get_label_for_id(dataset_label_id)

            for label_id, label in copy(review["labels"]).items():
                if label is not dataset_label:
                    del review["labels"][label_id]

            # when creating a dataset the datasets field in a review will only contain the dataset that is currently being created, same for test
            dataset_label["datasets"] = [dataset_name]
            if dataset_label_id not in label_id_counts:
                label_id_counts[dataset_label_id] = 1
            else:
                label_id_counts[dataset_label_id] += 1

        num_reviews_with_usage = len(
            reviews.filter(
                lambda review: review.label_has_usage_options(label_selection_strategy),
                inplace=False,
            )
        )

        return reviews, {
            "num_reviews": dataset_length,
            "num_reviews_for_labels": label_id_counts,
            "original_usage_split": round(num_reviews_with_usage / dataset_length, 3),
        }

    def score_labels_pairwise(
        self, label_ids: list[str] = None, metric_ids: list[str] = DEFAULT_METRICS
    ):
        if label_ids is None:
            label_ids = self.get_all_label_ids()

        for label_id in label_ids:
            for label_id2 in label_ids:
                if label_id != label_id2:
                    self.score(label_id, label_id2, metric_ids=metric_ids)

    def compute_label_variance(
        self,
        label_ids_to_compare: Union[str, list[str]] = "all",
        variance_type: str = "reviews",
        metric_ids: list[str] = DEFAULT_METRICS,
    ):
        """Computes the variance of the pairwise scores of the labels in the review set."""
        result = {metric_id: {} for metric_id in metric_ids}

        if label_ids_to_compare == "all":
            label_ids_to_compare = self.get_all_label_ids()
        else:
            assert set(label_ids_to_compare).issubset(set(self.get_all_label_ids()))

        self.score_labels_pairwise(
            label_ids=label_ids_to_compare, metric_ids=metric_ids
        )

        if variance_type == "reviews":
            res = {metric_id: [] for metric_id in metric_ids}

            for review in self:
                pairwise_scores_per_review = {}
                for label_id, label in review.get_labels().items():
                    if label_id not in label_ids_to_compare:
                        continue

                    for ref_id, score_dict in label["scores"].items():
                        if ref_id not in label_ids_to_compare or label_id == ref_id:
                            continue

                        key = tuple(sorted([label_id, ref_id]))
                        if key not in pairwise_scores_per_review:
                            pairwise_scores_per_review[key] = {
                                metric_id: score_dict[metric_id]
                                for metric_id in metric_ids
                            }

                for metric_id in metric_ids:
                    if len(pairwise_scores_per_review) == 0:
                        continue

                    metric_scores = [
                        scores[metric_id]
                        for scores in pairwise_scores_per_review.values()
                    ]
                    res[metric_id].append(
                        (review.review_id, mean(metric_scores), var(metric_scores))
                    )

            for metric_id in metric_ids:
                result[metric_id]["expectation"] = mean([x[1] for x in res[metric_id]])
                result[metric_id]["variance"] = mean([x[2] for x in res[metric_id]])

            return result

        elif variance_type == "labels":
            pairwise_scores = {}
            for review in self:
                for label_id, label in review.get_labels().items():
                    if label_id not in label_ids_to_compare:
                        continue

                    for ref_id, score_dict in label["scores"].items():
                        if ref_id not in label_ids_to_compare:
                            continue

                        key = tuple(sorted([label_id, ref_id]))
                        if key not in pairwise_scores:
                            pairwise_scores[key] = {
                                metric_id: [] for metric_id in metric_ids
                            }

                        for metric_id in metric_ids:
                            pairwise_scores[key][metric_id].append(
                                score_dict[metric_id]
                            )

            for metric_id in metric_ids:
                result[metric_id]["expectation"] = mean(
                    [mean(x[metric_id]) for x in pairwise_scores.values()]
                )
                result[metric_id]["variance"] = mean(
                    [var(x[metric_id]) for x in pairwise_scores.values()]
                )

            return result

    def merge_labels(self, *label_ids: str, new_label_id: str) -> None:
        assert new_label_id not in self.get_all_label_ids()

        strategy = ls.LabelIDSelectionStrategy(*label_ids)
        for review in self:
            label = review.get_label_from_strategy(strategy)
            review.add_label(
                label_id=new_label_id,
                usage_options=label["usageOptions"],
                datasets=label["datasets"],
                metadata=label["metadata"],
            )

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        if path:
            self.save_path = path
        assert (
            self.save_path is not None
        ), "ReviewSet has no `save_path`; please supply a path when calling `save`"

        with open(self.save_path, "w") as file:
            json.dump(self.get_data(), file)
