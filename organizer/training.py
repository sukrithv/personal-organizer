"""
training.py — build the trained classifier from data you already have (Phase 3).

The cold-start trick: your ALREADY-ORGANIZED folders are a labeled dataset — a
file in Documents/Taxes/2024 is an example whose label is that folder. Walk them
to bootstrap a model from your own conventions, then keep adding the corrections
you log with `organizer correct`.

Model: TF-IDF (word 1–2 grams UNION char 3–5 grams — filenames carry signal in
substrings) -> LogisticRegression. A few MB, trains in seconds, gives calibrated
confidences (predict_proba) that drive the confidence gate. Needs the `ml` extra:
    pip install -e ".[ml]"
"""

from __future__ import annotations

import os
import pickle
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path

from .config import IGNORE_SUFFIXES
from .core import FileContext
from .extractors import extract
from .features import featurize

DEFAULT_MODEL_PATH = Path.home() / ".file-organizer" / "model.pkl"


def bootstrap_dataset(organized_root, label_depth: int = 2,
                      preview_chars: int = 800) -> Iterator[tuple[str, str]]:
    """Yield (features, label) pairs from an already-organized tree.

    label_depth controls granularity: 1 -> 'Documents', 2 -> 'Documents/Taxes'.
    Loose files at the root have no folder, so they're skipped.
    """
    root = Path(organized_root)
    for dp, _dirs, files in os.walk(root):
        for name in files:
            p = Path(dp) / name
            if name.startswith(".") or p.suffix.lower() in IGNORE_SUFFIXES:
                continue
            rel = p.relative_to(root)
            folder_parts = rel.parts[:-1]
            if not folder_parts:
                continue
            label = "/".join(folder_parts[:label_depth])
            try:
                ctx = extract(FileContext.from_path(p), preview_chars)
            except OSError:
                continue
            yield featurize(ctx), label


def train(dataset: Iterable[tuple[str, str]], model_out=DEFAULT_MODEL_PATH,
          min_per_class: int = 2, test_size: float = 0.2, log=print) -> dict:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import FeatureUnion, Pipeline

    X: list[str] = []
    y: list[str] = []
    for feats, label in dataset:
        X.append(feats)
        y.append(label)
    if not X:
        raise ValueError("No training examples found.")

    # Drop classes too rare to learn anything from.
    counts = Counter(y)
    kept = [(x, lab) for x, lab in zip(X, y) if counts[lab] >= min_per_class]
    dropped = len(X) - len(kept)
    if len(kept) < 4 or len({lab for _, lab in kept}) < 2:
        raise ValueError(
            f"Not enough data to train: {len(kept)} usable examples across "
            f"{len({lab for _, lab in kept})} folder(s) with >= {min_per_class} files each. "
            "Point `train` at a folder you've already organized, lower "
            "--min-per-class, or use --label-depth 1.")
    X, y = map(list, zip(*kept))

    vectorizer = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)),
    ])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    pipe = Pipeline([("features", vectorizer), ("clf", clf)])

    acc = None
    # Stratified holdout only if every class has >= 2 examples and set is big enough.
    can_split = min(Counter(y).values()) >= 2 and len(y) >= 10
    if can_split:
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=0)
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        acc = accuracy_score(yte, pred)
        log(classification_report(yte, pred, zero_division=0))
        pipe.fit(X, y)   # refit on everything for the shipped model
    else:
        log("(dataset small — trained on all examples, no held-out score)")
        pipe.fit(X, y)

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(model_out, "wb") as f:
        pickle.dump({"pipeline": pipe, "labels": sorted(set(y))}, f)

    return {"accuracy": acc, "examples": len(y),
            "classes": len(set(y)), "dropped": dropped, "path": str(model_out)}


def load_model(path=DEFAULT_MODEL_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


def retrain_with_corrections(organized_root, correction_log,
                             model_out=DEFAULT_MODEL_PATH, label_depth: int = 2,
                             log=print) -> dict:
    data = list(bootstrap_dataset(organized_root, label_depth))
    data += list(correction_log.export())
    return train(data, model_out, log=log)
