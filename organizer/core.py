"""
core.py — the vocabulary of the whole system.

FileContext   what we know about a file
Decision      a classifier's answer: where, how sure, why, from whom
Classifier    the one-method interface every decision-maker implements
run_chain     tries classifiers in order, takes the first confident answer
Policy        turns confidence into an outcome (used from Phase 3 on)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path


@dataclass
class FileContext:
    path: Path
    name: str
    ext: str
    size: int
    modified: datetime
    created: datetime
    text_preview: str | None = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_path(cls, path: Path) -> FileContext:
        path = Path(path)
        st = path.stat()
        return cls(
            path=path,
            name=path.name,
            ext=path.suffix.lower().lstrip("."),
            size=st.st_size,
            modified=datetime.fromtimestamp(st.st_mtime),
            created=datetime.fromtimestamp(st.st_ctime),
        )

    def date(self, source: str = "modified") -> datetime:
        return self.created if source == "created" else self.modified


@dataclass
class Decision:
    folder: Path        # RELATIVE destination folder
    confidence: float   # 0.0 - 1.0
    reason: str
    source: str         # "rule" | "trained" | "llm" | "type"


class Classifier(ABC):
    name: str = "base"

    @abstractmethod
    def classify(self, ctx: FileContext) -> Decision | None:
        ...


def run_chain(
    ctx: FileContext,
    classifiers: list[Classifier],
    min_confidence: float = 0.0,
) -> Decision | None:
    """Return the first Decision at/above the bar; else the best below-bar
    guess; else None. In Phase 1 the bar is 0.0, so ordering decides precedence
    (rules -> llm -> type)."""
    best: Decision | None = None
    for clf in classifiers:
        d = clf.classify(ctx)
        if d is None:
            continue
        if d.confidence >= min_confidence:
            return d
        if best is None or d.confidence > best.confidence:
            best = d
    return best


class Outcome(Enum):
    AUTO = "auto"
    REVIEW = "review"
    QUARANTINE = "quarantine"


@dataclass
class Policy:
    """Confidence -> Outcome. Defined now, wired into moving in Phase 3."""
    auto_above: float = 0.80
    review_above: float = 0.50

    def decide(self, decision: Decision | None) -> Outcome:
        if decision is None:
            return Outcome.QUARANTINE
        if decision.confidence >= self.auto_above:
            return Outcome.AUTO
        if decision.confidence >= self.review_above:
            return Outcome.REVIEW
        return Outcome.QUARANTINE
