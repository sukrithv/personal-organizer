"""
dedupe.py — exact-duplicate detection.

A streamed SHA-256 (cached by size+mtime so we don't rehash unchanged files),
an index of hashes we've seen, and a DuplicateClassifier that routes an exact
duplicate to a holding folder instead of scattering copies around.

Safe by default: duplicates are MOVED TOGETHER, never deleted. You decide.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .core import Classifier, Decision, FileContext

_CACHE = {}  # (path, size, mtime) -> hexdigest


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


class DuplicateClassifier(Classifier):
    name = "dupe"

    def __init__(self, index: DuplicateIndex, folder: str = "_Duplicates"):
        self.index = index
        self.folder = folder

    def classify(self, ctx: FileContext) -> Decision | None:
        original = self.index.check_and_add(ctx.path)
        if original is None:
            return None
        return Decision(Path(self.folder), 1.0,
                        f"duplicate of {Path(original).name}", "dupe")
