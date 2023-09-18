import math
from statistics import mean
from typing import Union

from evaluation.scoring import DEFAULT_METRICS
from evaluation.scoring.core import get_most_similar, get_similarity
from evaluation.scoring.custom_metrics import (
    custom_f1_score,
    custom_f1_score_ak,
    custom_precision,
    custom_precision_ak,
    custom_recall,
    custom_recall_ak,
    word_movers_similarity,
)
from evaluation.scoring.standard_metrics import bleu_score, rouge_score, sacrebleu_score

# NOTE: Do not change these values. They are directly linked with specific metric names
# If you want to use other parameters, add new metric configurations in CUSTOM_METRIC_FUNCTIONS
KWARGS_MPNET_V1 = {
    "use_lowercase": True,
    "comparator": "all-mpnet-base-v2",
    "similarity_metric": "cosine_relu",
}

# This allows for using asyncio in ReviewSet.score()
CUSTOM_METRIC_FUNCTIONS = {
    "custom_min_precision": (custom_precision, {"agg": min, **KWARGS_MPNET_V1}),
    "custom_mean_precision": (custom_precision, {"agg": mean, **KWARGS_MPNET_V1}),
    "custom_min_recall": (custom_recall, {"agg": min, **KWARGS_MPNET_V1}),
    "custom_mean_recall": (custom_recall, {"agg": mean, **KWARGS_MPNET_V1}),
    "custom_min_f1": (custom_f1_score, {"agg": min, **KWARGS_MPNET_V1}),
    "custom_mean_f1": (custom_f1_score, {"agg": mean, **KWARGS_MPNET_V1}),
    "custom_weighted_mean_precision": (custom_precision_ak, KWARGS_MPNET_V1),
    "custom_weighted_mean_recall": (custom_recall_ak, KWARGS_MPNET_V1),
    "word_movers_similarity": (word_movers_similarity, KWARGS_MPNET_V1),
    "custom_weighted_mean_f1": (custom_f1_score_ak, KWARGS_MPNET_V1),
    "custom_weighted_mean_f1_stem": (
        custom_f1_score_ak,
        {"modification": "stem", **KWARGS_MPNET_V1},
    ),
    "custom_weighted_mean_f1_lemmatize": (
        custom_f1_score_ak,
        {"modification": "lemmatize", **KWARGS_MPNET_V1},
    ),
}
STANDARD_METRIC_FUNCTIONS = {
    "bleu": (bleu_score, {}),
    "sacrebleu": (sacrebleu_score, {}),
    "rouge1": (rouge_score, {"score_type": "rouge1"}),
    "rouge2": (rouge_score, {"score_type": "rouge2"}),
    "rougeL": (rouge_score, {"score_type": "rougeL"}),
    "rougeLsum": (rouge_score, {"score_type": "rougeLsum"}),
    "rouge1": (rouge_score, {"score_type": "rouge1"}),
}
DEFAULT_NLP_THRESHOLD = 0.7


class NoUseCaseOptions:
    pass


class SingleReviewMetrics:
    def __init__(self, predictions: list, references: list) -> None:
        self.predictions = predictions
        self.references = references

    @classmethod
    def from_labels(
        cls, labels: dict[str, dict], prediction_label_id: str, reference_label_id: str
    ):
        return cls(
            predictions=labels[prediction_label_id]["usageOptions"],
            references=labels[reference_label_id]["usageOptions"],
        )

    def calculate(
        self, metric_ids=DEFAULT_METRICS, include_pos_neg_info=False
    ) -> Union[dict[str, float], dict[str, tuple[float, str]]]:
        scores = {}

        for metric_id in metric_ids:
            if metric_id in CUSTOM_METRIC_FUNCTIONS:
                fn, kwargs = CUSTOM_METRIC_FUNCTIONS[metric_id]
                metric_result = fn(
                    predictions=self.predictions, references=self.references, **kwargs
                )
            elif metric_id in STANDARD_METRIC_FUNCTIONS:
                fn, kwargs = STANDARD_METRIC_FUNCTIONS[metric_id]
                metric_result = fn(
                    predictions=self.predictions, references=self.references, **kwargs
                )
            else:
                try:
                    metric_result = getattr(self, metric_id)()
                except ZeroDivisionError:
                    metric_result = math.nan

            if include_pos_neg_info:
                scores[metric_id] = (
                    metric_result,
                    True if len(self.predictions) > 0 else False,
                    True if len(self.references) > 0 else False,
                )
            else:
                scores[metric_id] = metric_result

        return scores

    def custom_classification_score(self):
        matches = {reference: [] for reference in self.references}
        non_matching_predictions = []

        for prediction in self.predictions:
            is_prediction_matched = False
            for reference in self.references:
                similarity = get_similarity(prediction, reference, **KWARGS_MPNET_V1)
                if similarity >= DEFAULT_NLP_THRESHOLD:
                    matches[reference].append(prediction)
                    is_prediction_matched = True
            if is_prediction_matched == False:
                non_matching_predictions.append(prediction)

        results = dict.fromkeys(["TP", "FP", "TN", "FN"], 0)

        results["TP"] = sum(
            [
                1 if len(matched_predictions) > 0 else 0
                for matched_predictions in matches.values()
            ]
        )
        results["FP"] = len(non_matching_predictions)
        results["TN"] = int(len(self.references) == len(self.predictions) == 0)
        results["FN"] = sum(
            [
                0 if len(matched_predictions) > 0 else 1
                for matched_predictions in matches.values()
            ]
        )

        return results

    def custom_classification_score_with_negative_class(self):
        matches = {reference: [] for reference in self.references}
        non_matching_predictions = []

        for prediction in self.predictions:
            is_prediction_matched = False
            for reference in self.references:
                similarity = get_similarity(prediction, reference, **KWARGS_MPNET_V1)
                if similarity >= DEFAULT_NLP_THRESHOLD:
                    matches[reference].append(prediction)
                    is_prediction_matched = True
            if is_prediction_matched == False:
                non_matching_predictions.append(prediction)

        results = dict.fromkeys(["TP", "FP", "TN", "FN"], 0)

        results["TN"] = math.inf
        if len(self.references) == 0:
            results["TP"] = int(len(self.references) == len(self.predictions) == 0)
            results["FP"] = len(self.predictions)
            results["FN"] = int(len(self.predictions) > 0)
            return results
        else:
            results["TP"] = sum(
                [
                    1 if len(matched_predictions) > 0 else 0
                    for matched_predictions in matches.values()
                ]
            )
            results["FP"] = len(non_matching_predictions)
            results["FN"] = sum(
                [
                    0 if len(matched_predictions) > 0 else 1
                    for matched_predictions in matches.values()
                ]
            )

            return results

    # not in use
    def custom_symmetric_similarity_classification_score_with_negative_class(self):
        best_matching_predictions = dict.fromkeys(self.references, 0)
        best_matching_references = dict.fromkeys(self.predictions, 0)

        for prediction in self.predictions:
            similarity, best_matched_reference = get_most_similar(
                prediction, self.references, **KWARGS_MPNET_V1
            )
            best_matching_references[prediction] = (similarity, best_matched_reference)

        for reference in self.references:
            similarity, best_matching_prediction = get_most_similar(
                reference, self.predictions, **KWARGS_MPNET_V1
            )
            best_matching_predictions[reference] = (
                similarity,
                best_matching_prediction,
            )

        if len(self.references) == 0:
            if len(self.predictions) == 0:
                best_matching_predictions[NoUseCaseOptions] = (1.0, NoUseCaseOptions)
            else:
                best_matching_predictions[NoUseCaseOptions] = (0.0, None)

        if len(self.predictions) == 0:
            if len(self.references) == 0:
                best_matching_references[NoUseCaseOptions] = (1.0, NoUseCaseOptions)
            else:
                best_matching_references[NoUseCaseOptions] = (0.0, None)

        return best_matching_predictions, best_matching_references

    def custom_similarity_classification_score_with_negative_class(self):
        best_matching_predictions = dict.fromkeys(self.references, 0)
        best_matching_references = dict.fromkeys(self.predictions, 0)

        for prediction in self.predictions:
            similarity, best_matched_reference = get_most_similar(
                prediction, self.references, **KWARGS_MPNET_V1
            )
            best_matching_references[prediction] = (similarity, best_matched_reference)
        for reference in self.references:
            similarity, best_matching_prediction = get_most_similar(
                reference, self.predictions, **KWARGS_MPNET_V1
            )
            best_matching_predictions[reference] = (
                similarity,
                best_matching_prediction,
            )

        if len(self.references) == 0:
            if len(self.predictions) == 0:
                best_matching_predictions[NoUseCaseOptions] = (1.0, NoUseCaseOptions)
            else:
                best_matching_predictions[NoUseCaseOptions] = (0.0, None)

        if len(self.predictions) == 0:
            if len(self.references) == 0:
                best_matching_references[NoUseCaseOptions] = (1.0, NoUseCaseOptions)
            else:
                best_matching_references[NoUseCaseOptions] = (0.0, None)

        results = dict.fromkeys(["TP", "FP", "TN", "FN"], 0)

        results["TN"] = math.inf

        results["TP"] = sum(
            [similarity for similarity, _ in best_matching_predictions.values()]
        )

        results["FN"] = sum(
            [1 - similarity for similarity, _ in best_matching_predictions.values()]
        )

        results["FP"] = sum(
            [1 - similarity for similarity, _ in best_matching_references.values()]
        )

        return results
