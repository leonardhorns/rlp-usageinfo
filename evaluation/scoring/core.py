from pathlib import Path
from typing import Optional, Union

import evaluate
import pandas as pd

from scipy.stats import beta

from evaluation.scoring.evaluation_cache import EvaluationCache
from helpers.extract_reviews import extract_reviews_with_usage_options_from_json

# models for string similarity will only be loaded when needed
spacy_eval = bleu_eval = sacrebleu_eval = rouge_eval = None
st_eval = {}


def extract_review_with_id(df: pd.DataFrame, review_id: str) -> Optional[pd.Series]:
    review = df[df.review_id == review_id]
    if review.empty:
        return None
    return review.iloc[0]


def human_predictions_to_labels(
    predictions_path: Union[Path, str],
    ground_truth_path: Union[Path, str],
    origin: Optional[str],
):
    vendor_data = extract_reviews_with_usage_options_from_json(predictions_path)
    golden_data = extract_reviews_with_usage_options_from_json(ground_truth_path)

    labels = []
    for _, predicted_review in vendor_data.iterrows():
        golden_review = extract_review_with_id(golden_data, predicted_review.review_id)
        if golden_review is not None:
            labels.append(
                {
                    "review_id": predicted_review.review_id,
                    "references": golden_review.usage_options,
                    "predictions": predicted_review.usage_options,
                    "origin": origin if origin else predicted_review.workerId,
                }
            )

    return labels


def get_embedding(usage_option: str, comparator: str = "all-mpnet-base-v2") -> list:
    global st_eval, spacy_eval, bleu_eval, sacrebleu_eval, rouge_eval

    cache = EvaluationCache.get()
    key = (comparator, usage_option)

    if key in cache:
        return cache[key]

    if (
        comparator == "all-mpnet-base-v2"
        or comparator == "sentence-t5-xxl"
        or comparator == "gtr-t5-xxl"
    ):
        if comparator not in st_eval:
            from sentence_transformers import SentenceTransformer

            st_eval[comparator] = SentenceTransformer(comparator)
        embedding = st_eval[comparator].encode(usage_option)
    elif comparator == "spacy":
        if spacy_eval is None:
            import spacy

            spacy_eval = spacy.load("en_core_web_lg")
        embedding = spacy_eval(usage_option).vector
    else:
        raise ValueError(f"embeddings for metric {comparator} doesn't exist")

    cache[key] = embedding
    return embedding


def get_similarity(
    label_1: str,
    label_2: str,
    comparator: str = "all-mpnet-base-v2",
    use_lowercase: bool = True,
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
    similarity_metric: str = "cosine_relu",  # alternative: "euclidean", "cosine"
) -> float:
    global st_eval, spacy_eval, bleu_eval, sacrebleu_eval, rouge_eval

    if use_lowercase:
        label_1 = label_1.lower()
        label_2 = label_2.lower()

    if modification == "stem":
        import nltk

        nltk.download("punkt", quiet=True)
        ps = nltk.stem.PorterStemmer()
        label_1 = " ".join(ps.stem(word) for word in label_1.split())
        label_2 = " ".join(ps.stem(word) for word in label_2.split())
    elif modification == "lemmatize":
        import nltk

        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        wnl = nltk.stem.WordNetLemmatizer()
        label_1 = " ".join(wnl.lemmatize(word) for word in label_1.split())
        label_2 = " ".join(wnl.lemmatize(word) for word in label_2.split())

    elif (
        comparator == "all-mpnet-base-v2"
        or comparator == "spacy"
        or comparator == "sentence-t5-xxl"
        or comparator == "gtr-t5-xxl"
    ):
        prediction_tokens = get_embedding(label_1, comparator)
        reference_tokens = get_embedding(label_2, comparator)
        from sentence_transformers import util

        if similarity_metric == "euclidean":
            from numpy import linalg

            # normalize to unit vectors
            prediction_tokens = prediction_tokens / linalg.norm(prediction_tokens)
            reference_tokens = reference_tokens / linalg.norm(reference_tokens)
            similarity = 1 - (linalg.norm(prediction_tokens - reference_tokens) / 2.0)
        elif similarity_metric == "cosine":
            similarity = util.cos_sim(prediction_tokens, reference_tokens)[0][0].item()
        elif similarity_metric == "cosine_relu":
            similarity = min(
                1,
                max(0, util.cos_sim(prediction_tokens, reference_tokens)[0][0].item()),
            )
            similarity = beta.cdf(similarity, 1.3492828476735637, 1.6475489724420649)
            similarity = beta.cdf(similarity, 14.715846019280558, 3.3380276739903016)
        else:
            raise ValueError(f"similarity metric {similarity_metric} not supported")
        return similarity
    elif comparator == "bleu":
        if bleu_eval is None:
            bleu_eval = evaluate.load("bleu")
        pr, re = [label_1], [[label_2]]
        return bleu_eval.compute(predictions=pr, references=re)["bleu"]

    elif comparator == "sacrebleu":
        if sacrebleu_eval is None:
            sacrebleu_eval = evaluate.load("sacrebleu")
        res = sacrebleu_eval.compute(predictions=[label_1], references=[[label_2]])
        return res["score"] * 0.01
    else:
        if rouge_eval is None:
            rouge_eval = evaluate.load("rouge")
        pr, re = [label_1], [[label_2]]
        rogue_metrics = rouge_eval.compute(predictions=pr, references=re)
        # currently available: rouge1, rouge2, rougeL, rougeLsum
        if comparator in rogue_metrics.keys():
            return rogue_metrics[comparator]
        else:
            raise ValueError(f"comparator {comparator} is not supported")


def get_most_similar(
    label: str,
    options: list[str],
    comparator: str = "all-mpnet-base-v2",
    use_lowercase: bool = True,
    modification: Optional[str] = None,  # options: "stem" or "lemmatize"
    similarity_metric: str = "cosine_relu",
    threshold_word_sim: float = 0,
) -> tuple[float, str]:
    """For a single `label`, find the most similar match from `options`.

    Returns tuple (best similarity score, option with best similiarity score)."""
    assert 0 <= threshold_word_sim <= 1

    result = (0, None)
    for option in options:
        similarity = get_similarity(
            label_1=option,
            label_2=label,
            comparator=comparator,
            use_lowercase=use_lowercase,
            modification=modification,
            similarity_metric=similarity_metric,
        )
        if similarity >= max(result[0], threshold_word_sim):
            result = (similarity, option)

    return result
