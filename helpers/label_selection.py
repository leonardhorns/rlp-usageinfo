from __future__ import annotations
from abc import ABC
import fnmatch
from typing import Optional, Union
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from helpers.review import Review


class LabelSelectionStrategyInterface(ABC):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "retrieve_label")
            and callable(subclass.retrieve_label)
            and hasattr(subclass, "retrieve_label_id")
            and callable(subclass.retrieve_label_id)
            and hasattr(subclass, "retrieve_labels")
            and callable(subclass.retrieve_labels)
            and hasattr(subclass, "retrieve_label_ids")
            and callable(subclass.retrieve_label_ids)
        )

    def retrieve_label(self, review: Review) -> Optional[dict]:
        raise NotImplementedError

    def retrieve_label_id(self, review: Review) -> Optional[str]:
        raise NotImplementedError

    def retrieve_labels(self, review: Review) -> list[dict]:
        raise NotImplementedError

    def retrieve_label_ids(self, review: Review) -> list[str]:
        raise NotImplementedError


class AbstractLabelSelectionStrategy:
    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def retrieve_label(self, review: Review) -> Optional[dict]:
        try:
            return review.get_labels()[self.retrieve_label_id(review)]
        except KeyError:
            return None

    def retrieve_label_id(self, review: Review) -> Optional[str]:
        label_ids = self.retrieve_label_ids(review)
        if len(label_ids) > 0:
            return label_ids[0]
        return None

    def retrieve_labels(self, review: Review) -> list[dict]:
        label_ids = self.retrieve_label_ids(review)
        labels = review.get_labels()
        return [labels[label_id] for label_id in label_ids]

    def retrieve_label_ids(self, review: Review) -> list[str]:
        raise NotImplementedError


class LabelIDSelectionStrategy(AbstractLabelSelectionStrategy):
    def __init__(self, *label_ids: str):
        self.label_ids = label_ids

    def retrieve_label_ids(self, review: Review) -> list[str]:
        review_label_ids = review.get_label_ids()
        label_ids = []
        for label_id in self.label_ids:
            matches = sorted(fnmatch.filter(review_label_ids, label_id))
            if matches:
                label_ids += matches
        return label_ids


class DatasetSelectionStrategy(AbstractLabelSelectionStrategy):
    def __init__(self, *datasets: str):
        self.datasets = datasets

    def retrieve_label_ids(self, review: Review) -> list[str]:
        label_ids = []
        for dataset in self.datasets:
            for label_id, label in review.get_labels().items():
                datasets = label["datasets"]
                if dataset in datasets:
                    label_ids.append(label_id)

        return label_ids
