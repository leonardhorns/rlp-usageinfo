import evaluate

# models for string similarity will only be loaded when needed
bleu_eval = sacrebleu_eval = rouge_eval = None


def bleu_score(predictions: list[str], references: list[str]) -> float:
    global bleu_eval

    if bleu_eval is None:
        bleu_eval = evaluate.load("bleu")

    return bleu_eval.compute(
        predictions=predictions,
        references=[references for i in range(len(predictions))],
    )["bleu"]


def sacrebleu_score(predictions: list[str], references: list[str]) -> float:
    global sacrebleu_eval

    if sacrebleu_eval is None:
        sacrebleu_eval = evaluate.load("sacrebleu")

    return (
        sacrebleu_eval.compute(
            predictions=predictions,
            references=[references for i in range(len(predictions))],
        )["score"]
        * 0.01
    )


def rouge_score(
    predictions: list[str], references: list[str], score_type="rouge1"
) -> float:
    global rouge_eval

    if rouge_eval is None:
        rouge_eval = evaluate.load("rouge")

    scores = rouge_eval.compute(
        predictions=predictions,
        references=[references for i in range(len(predictions))],
    )
    if score_type in scores.keys():
        return scores[score_type]
    raise ValueError(f"Invalid score type: {score_type}")
