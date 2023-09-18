from scipy import stats
import numpy as np


def ttest(
    scores: np.array,
    alternative: str = "two-sided",
):
    return stats.ttest_1samp(scores, 0, alternative=alternative)


def wilcoxon(
    scores: np.array,
    alternative: str = "two-sided",
):
    return stats.wilcoxon(scores, alternative=alternative)


def bootstrap(
    scores: np.array,
    alternative: str = "two-sided",
    confidence_level: float = 0.95,
):
    assert 0 < confidence_level < 1

    res = stats.bootstrap((scores,), np.mean, confidence_level=confidence_level)

    if res.confidence_interval[0] > 0:
        if alternative == "two-sided" or alternative == "greater":
            return (
                res,
                f"H0 rejected for signficance level {1 - confidence_level}",
            )
    if res.confidence_interval[1] < 0:
        if alternative == "two-sided" or alternative == "less":
            return (
                res,
                f"H0 rejected for signficance level {1 - confidence_level}",
            )
    else:
        return (
            res,
            f"H0 NOT rejected for signficance level {1 - confidence_level}",
        )


def permutation(
    scores_1: np.array,
    scores_2: np.array,
    alternative: str = "two-sided",
):
    def statistic(x, y, axis):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    try:
        return stats.permutation_test(
            (scores_1, scores_2),
            statistic,
            alternative=alternative,
            vectorized=True,
            permutation_type="samples",
        )
    except ValueError:
        print("Permutation test failed, returning None")
        return None


def _statistic_harmonic(x: np.ndarray, y: np.ndarray):
    x = x.transpose()
    y = y.transpose()

    positives_x = x[x[:, 2] == 1, 0]
    negatives_x = x[x[:, 2] == 0, 0]
    positives_y = y[y[:, 2] == 1, 0]
    negatives_y = y[y[:, 2] == 0, 0]

    positive_score_x = np.mean(positives_x) if len(positives_x) > 0 else 1
    negative_score_x = np.mean(negatives_x) if len(negatives_x) > 0 else 1
    positive_score_y = np.mean(positives_y) if len(positives_y) > 0 else 1
    negative_score_y = np.mean(negatives_y) if len(negatives_y) > 0 else 1

    harmonic_mean_x = (
        2 * positive_score_x * negative_score_x / (positive_score_x + negative_score_x)
    )
    harmonic_mean_y = (
        2 * positive_score_y * negative_score_y / (positive_score_y + negative_score_y)
    )

    return harmonic_mean_x - harmonic_mean_y


def statistic_harmonic(x: np.ndarray, y: np.ndarray, axis):
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    return (
        np.array([_statistic_harmonic(x[i], y[i]) for i in range(len(x))])
        if len(x.shape) > 2
        else _statistic_harmonic(x, y)
    )


def permutation_harmonic(
    scores_1,
    scores_2,
    alternative: str = "two-sided",
):
    scores_1 = np.array(
        [np.array([score[0], int(score[1]), int(score[2])]) for score in scores_1]
    )
    scores_2 = np.array(
        [np.array([score[0], int(score[1]), int(score[2])]) for score in scores_2]
    )
    return stats.permutation_test(
        (scores_1, scores_2),
        statistic_harmonic,
        alternative=alternative,
        permutation_type="samples",
        vectorized=True,
        axis=0,
        n_resamples=10000,
    )


def h_test(
    test_type: str = "ttest",
    scores_1: np.array = None,
    scores_2: np.array = None,
    alternative: str = "two-sided",
    confidence_level: float = None,
):
    """
    Perform a hypothesis test on the given scores.
    Is for two paired samples meaning they have equal length and the indexes match to the subsamples.
    """
    scores = scores_1 - scores_2

    if test_type in ["ttest", "wilcoxon"]:
        return eval(f"{test_type}(scores, alternative)")
    elif test_type == "bootstrap":
        return bootstrap(scores, alternative, confidence_level)
    elif test_type == "permutation":
        return permutation(scores_1, scores_2, alternative)
    else:
        raise ValueError("Unknown test type")
