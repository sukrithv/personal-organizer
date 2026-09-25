"""
dedupe.py — exact- and near-duplicate detection.

Exact: a streamed SHA-256 (cached by size+mtime so we don't rehash unchanged
files) and an index of hashes we've seen.

Near: byte-identical hashing misses re-saves, re-exports, and lightly-edited
copies (a re-typed header, a changed footer, a different export tool). For
those, NearDuplicateIndex takes the first N tokens of each document's text
preview, turns them into a term-frequency vector, and compares that vector
to every vector seen so far by cosine similarity. A score at/above the
configured threshold means "close enough to be the same document" and the
new file is routed to the same place as the original.

DuplicateClassifier wraps both: exact match first (cheap, certain), then
near-match (if a NearDuplicateIndex was supplied), and routes either kind to
a holding folder instead of scattering copies around.

Safe by default: duplicates are MOVED TOGETHER, never deleted. You decide.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from collections import Counter
from pathlib import Path

from .core import Classifier, Decision, FileContext

_CACHE = {}  # (path, size, mtime) -> hexdigest
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    path = Path(path)
    st = path.stat()
    key = (str(path), st.st_size, int(st.st_mtime))
    if key in _CACHE:
        return _CACHE[key]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    _CACHE[key] = h.hexdigest()
    return _CACHE[key]


class DuplicateIndex:
    def __init__(self):
        self.seen = {}  # hexdigest -> first path that had it

    def scan(self, root: Path) -> None:
        """Pre-populate from an existing tree so new files that duplicate
        already-filed ones are caught."""
        for dp, _, files in os.walk(root):
            for name in files:
                p = Path(dp) / name
                try:
                    self.seen.setdefault(file_sha256(p), p)
                except OSError:
                    continue

    def check_and_add(self, path: Path) -> Path | None:
        """Return the original path if this is a duplicate, else None (and record)."""
        try:
            digest = file_sha256(path)
        except OSError:
            return None
        original = self.seen.get(digest)
        if original is not None and Path(original) != Path(path):
            return original
        self.seen.setdefault(digest, path)
        return None


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens, punctuation and whitespace dropped."""
    return _TOKEN_RE.findall(text.lower())


def leading_term_vector(text: str, n_tokens: int = 100) -> Counter:
    """Term-frequency vector over just the first n_tokens tokens of text.

    Only the leading tokens are used (not the whole preview) so two documents
    that start the same way — a shared letterhead, a boilerplate opening
    paragraph, a repeated report template — score as similar even if they
    later diverge, which is exactly the "same document, edited" case we want
    to catch.
    """
    return Counter(tokenize(text)[:n_tokens])


def cosine_similarity(a: Counter, b: Counter) -> float:
    """Cosine similarity between two term-frequency vectors, in [0.0, 1.0]."""
    if not a or not b:
        return 0.0
    common = a.keys() & b.keys()
    dot = sum(a[k] * b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class NearDuplicateIndex:
    """Flags files whose leading text is a close match for something already
    seen, even when the bytes differ.

    Comparison is against every vector seen so far (O(n) per file), which is
    fine at personal-file-organizer scale; it isn't meant for indexing
    hundreds of thousands of documents.
    """

    def __init__(self, n_tokens: int = 100, threshold: float = 0.9,
                min_tokens: int = 20):
        self.n_tokens = n_tokens
        self.threshold = threshold
        self.min_tokens = min_tokens
        self.seen: list[tuple[Path, Counter]] = []  # (first path, its vector)

    def scan(self, root: Path, preview_chars: int | None = None) -> None:
        """Pre-populate from an existing tree so new files that near-duplicate
        already-filed ones are caught. Mirrors DuplicateIndex.scan."""
        from .extractors import extract

        chars = preview_chars or self.n_tokens * 10  # rough chars-per-token headroom
        for dp, _, files in os.walk(root):
            for name in files:
                p = Path(dp) / name
                try:
                    ctx = extract(FileContext.from_path(p), chars)
                except OSError:
                    continue
                self.check_and_add(p, ctx.text_preview)

    def check_and_add(self, path: Path,
                      text_preview: str | None) -> tuple[Path, float] | None:
        """Return (original_path, similarity) if text_preview's leading tokens
        are a near-duplicate of something already seen, else None (and record
        this file's vector for future comparisons)."""
        if not text_preview:
            return None
        vec = leading_term_vector(text_preview, self.n_tokens)
        if sum(vec.values()) < self.min_tokens:
            return None  # too little text to compare reliably
        for seen_path, seen_vec in self.seen:
            if Path(seen_path) == Path(path):
                continue
            sim = cosine_similarity(vec, seen_vec)
            if sim >= self.threshold:
                return seen_path, sim
        self.seen.append((path, vec))
        return None


class DuplicateClassifier(Classifier):
    name = "dupe"

    def __init__(self, index: DuplicateIndex, folder: str = "_Duplicates",
                near_index: NearDuplicateIndex | None = None):
        self.index = index
        self.folder = folder
        self.near_index = near_index

    def classify(self, ctx: FileContext) -> Decision | None:
        original = self.index.check_and_add(ctx.path)
        if original is not None:
            return Decision(Path(self.folder), 1.0,
                            f"duplicate of {Path(original).name}", "dupe")
        if self.near_index is not None:
            hit = self.near_index.check_and_add(ctx.path, ctx.text_preview)
            if hit is not None:
                near_original, sim = hit
                return Decision(
                    Path(self.folder), sim,
                    f"near-duplicate of {Path(near_original).name} ({sim:.0%} similar)",
                    "dupe")
        return None
