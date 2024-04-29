from typing import Iterable

import numpy as np


def rouge_like_metric(pred: dict[str, list[str]], target: dict[str, list[str]], penalty: bool = False) -> float:
    pred_pairs: set[tuple[str, str]] = set()
    target_pairs: set[tuple[str, str]] = set()

    for dict_, pair in zip((pred, target), (pred_pairs, target_pairs)):
        for noun, adjectives in dict_.items():
            for adjective in adjectives:
                pair.add((noun.lower(), adjective.lower()))

    extracted = pred_pairs & target_pairs
    if penalty:
        incorrectly_extracted = pred_pairs - target_pairs
        metric = (len(extracted) - len(incorrectly_extracted)) / len(target_pairs)
    else:
        metric = len(extracted) / len(target_pairs)

    return metric


def mean_rouge_like_metric(
    pred_col: Iterable[dict[str, list[str]]], target_col: Iterable[dict[str, list[str]]], penalty: bool = False
) -> float:
    metrics = np.array(
        [rouge_like_metric(pred=pred, target=target, penalty=penalty) for pred, target in zip(pred_col, target_col)]
    )
    return metrics.mean()
