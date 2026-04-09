"""Baseline solution for binary security patch recognition.

Interface required by the grader:
  - predict(samples: list[dict]) -> list[float]
  - optional train(samples: list[dict], labels: list[int]) -> None
"""

from __future__ import annotations

import math
import re
from collections import Counter

TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_+\-]+")

_token_weights: dict[str, float] = {}
_bias: float = 0.0

POSITIVE_HINTS = {
    "bounds_check",
    "length_check",
    "sanitize",
    "overflow",
    "uaf",
    "null_check",
    "auth_check",
    "permission_check",
    "strncpy",
    "memmove",
    "validate",
    "clamp",
}

NEGATIVE_HINTS = {
    "refactor",
    "rename",
    "comment",
    "format",
    "logging",
    "trace",
    "cleanup",
    "perf_tune",
}


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOKEN_PATTERN.findall(text)]


def _extract_tokens(sample: dict) -> list[str]:
    before = str(sample.get("before", ""))
    after = str(sample.get("after", ""))
    context = str(sample.get("context", ""))
    before_tokens = _tokenize(before)
    after_tokens = _tokenize(after)
    before_set = set(before_tokens)
    after_set = set(after_tokens)
    added = [t for t in after_tokens if t not in before_set]
    removed = [t for t in before_tokens if t not in after_set]
    return (
        [f"after:{t}" for t in after_tokens]
        + [f"add:{t}" for t in added]
        + [f"del:{t}" for t in removed]
        + [f"ctx:{t}" for t in _tokenize(context)]
    )


def train(samples: list[dict], labels: list[int]) -> None:
    """Fit simple token log-odds weights from dev split."""
    global _token_weights, _bias
    pos_counts = Counter()
    neg_counts = Counter()
    pos_n = 0
    neg_n = 0
    for sample, label in zip(samples, labels):
        feats = set(_extract_tokens(sample))
        if int(label) == 1:
            pos_n += 1
            pos_counts.update(feats)
        else:
            neg_n += 1
            neg_counts.update(feats)

    vocab = set(pos_counts) | set(neg_counts)
    _token_weights = {}
    alpha = 1.0
    for token in vocab:
        p = (pos_counts[token] + alpha) / (pos_n + 2 * alpha)
        n = (neg_counts[token] + alpha) / (neg_n + 2 * alpha)
        _token_weights[token] = math.log(p / n)
    _bias = 0.0


def _heuristic_boost(sample: dict) -> float:
    before = str(sample.get("before", "")).lower()
    after = str(sample.get("after", "")).lower()
    text = f"{before} {after}"
    boost = 0.0
    for k in POSITIVE_HINTS:
        if k in text:
            boost += 0.3
    for k in NEGATIVE_HINTS:
        if k in text:
            boost -= 0.25
    return boost


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def predict(samples: list[dict]) -> list[float]:
    """Return probability scores in [0, 1]."""
    predictions: list[float] = []
    for sample in samples:
        tokens = set(_extract_tokens(sample))
        score = _bias + sum(_token_weights.get(t, 0.0) for t in tokens)
        score += _heuristic_boost(sample)
        prob = _sigmoid(score)
        predictions.append(max(0.0, min(1.0, prob)))
    return predictions
