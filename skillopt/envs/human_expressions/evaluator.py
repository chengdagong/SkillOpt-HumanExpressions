"""Deterministic text-similarity evaluator for rewrite-to-original tasks."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Any


DEFAULT_HARD_THRESHOLD = 0.86
DEFAULT_WEIGHTS = {
    "char_ngram_fscore": 0.50,
    "normalized_edit_similarity": 0.30,
    "rouge_l_fscore": 0.15,
    "structure_similarity": 0.05,
}


def normalize_text(text: Any) -> str:
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    replacements = {
        "“": "\"",
        "”": "\"",
        "‘": "'",
        "’": "'",
        "……": "...",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    lines = [" ".join(line.strip().split()) for line in value.split("\n")]
    collapsed: list[str] = []
    blank_seen = False
    for line in lines:
        if not line:
            if not blank_seen:
                collapsed.append("")
            blank_seen = True
            continue
        collapsed.append(line)
        blank_seen = False
    return "\n".join(collapsed).strip()


def _char_ngrams(text: str, n_min: int = 1, n_max: int = 4) -> Counter[str]:
    compact = re.sub(r"\s+", "", text)
    counts: Counter[str] = Counter()
    for n in range(n_min, n_max + 1):
        if len(compact) < n:
            continue
        counts.update(compact[i : i + n] for i in range(len(compact) - n + 1))
    return counts


def _fscore(overlap: int, predicted_total: int, reference_total: int, beta: float = 1.0) -> float:
    if predicted_total <= 0 and reference_total <= 0:
        return 1.0
    if predicted_total <= 0 or reference_total <= 0 or overlap <= 0:
        return 0.0
    precision = overlap / predicted_total
    recall = overlap / reference_total
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / ((beta_sq * precision) + recall)


def char_ngram_fscore(prediction: str, reference: str, n_min: int = 1, n_max: int = 4) -> float:
    pred_counts = _char_ngrams(prediction, n_min=n_min, n_max=n_max)
    ref_counts = _char_ngrams(reference, n_min=n_min, n_max=n_max)
    overlap = sum((pred_counts & ref_counts).values())
    return _fscore(overlap, sum(pred_counts.values()), sum(ref_counts.values()))


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (char_a != char_b),
                )
            )
        previous = current
    return previous[-1]


def normalized_edit_similarity(prediction: str, reference: str) -> float:
    if not prediction and not reference:
        return 1.0
    denominator = max(len(prediction), len(reference), 1)
    return max(0.0, 1.0 - (levenshtein_distance(prediction, reference) / denominator))


def _lcs_length(a: str, b: str) -> int:
    if not a or not b:
        return 0
    if len(a) > len(b):
        a, b = b, a
    previous = [0] * (len(a) + 1)
    for char_b in b:
        current = [0]
        for i, char_a in enumerate(a, start=1):
            if char_a == char_b:
                current.append(previous[i - 1] + 1)
            else:
                current.append(max(previous[i], current[i - 1]))
        previous = current
    return previous[-1]


def rouge_l_fscore(prediction: str, reference: str) -> float:
    pred = re.sub(r"\s+", "", prediction)
    ref = re.sub(r"\s+", "", reference)
    return _fscore(_lcs_length(pred, ref), len(pred), len(ref))


def _paragraph_lengths(text: str) -> list[int]:
    paragraphs = [re.sub(r"\s+", "", part) for part in text.split("\n\n")]
    return [len(part) for part in paragraphs if part]


def structure_similarity(prediction: str, reference: str) -> float:
    pred_lengths = _paragraph_lengths(prediction)
    ref_lengths = _paragraph_lengths(reference)
    paragraph_score = 1.0 - (
        abs(len(pred_lengths) - len(ref_lengths)) / max(len(pred_lengths), len(ref_lengths), 1)
    )

    pred_len = sum(pred_lengths)
    ref_len = sum(ref_lengths)
    length_score = 1.0 - (abs(pred_len - ref_len) / max(pred_len, ref_len, 1))

    punct = "，。！？；：,.!?;:"
    pred_punct = sum(prediction.count(ch) for ch in punct)
    ref_punct = sum(reference.count(ch) for ch in punct)
    punct_score = 1.0 - (abs(pred_punct - ref_punct) / max(pred_punct, ref_punct, 1))
    return max(0.0, min(1.0, (paragraph_score + length_score + punct_score) / 3))


def evaluate_rewrite(
    prediction: str,
    reference: str,
    *,
    hard_threshold: float = DEFAULT_HARD_THRESHOLD,
) -> dict:
    predicted_norm = normalize_text(prediction)
    reference_norm = normalize_text(reference)
    metrics = {
        "char_ngram_fscore": char_ngram_fscore(predicted_norm, reference_norm),
        "normalized_edit_similarity": normalized_edit_similarity(predicted_norm, reference_norm),
        "rouge_l_fscore": rouge_l_fscore(predicted_norm, reference_norm),
        "structure_similarity": structure_similarity(predicted_norm, reference_norm),
    }
    soft = sum(metrics[name] * weight for name, weight in DEFAULT_WEIGHTS.items())
    return {
        "hard": int(soft >= hard_threshold),
        "soft": soft,
        "metrics": metrics,
        "predicted_text": predicted_norm,
        "reference_text": reference_norm,
        "hard_threshold": hard_threshold,
    }

