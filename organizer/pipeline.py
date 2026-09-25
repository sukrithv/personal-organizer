"""
pipeline.py — assemble the chain and turn a folder into a list of moves.

An optional DuplicateClassifier sits at the front of the chain, and Move
carries source + confidence so the SQLite log can record them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .classifiers import (
    LLMClassifier,
    RuleClassifier,
    TrainedClassifier,
    TypeClassifier,
)
from .config import IGNORE_SUFFIXES
from .core import Classifier, Decision, FileContext, run_chain
from .dedupe import DuplicateClassifier, DuplicateIndex, NearDuplicateIndex
from .extractors import extract
from .training import DEFAULT_MODEL_PATH


def build_chain(cfg, use_llm: bool, llm_ok: bool,
                dupe_index: DuplicateIndex | None = None,
                near_dupe_index: NearDuplicateIndex | None = None) -> list[Classifier]:
    chain: list[Classifier] = []
    if dupe_index is not None:
        folder = cfg.get("dedupe", {}).get("folder", "_Duplicates")
        chain.append(DuplicateClassifier(dupe_index, folder, near_dupe_index))
    chain.append(RuleClassifier(cfg))
    mp = cfg.get("model_path") or DEFAULT_MODEL_PATH
    if Path(mp).exists():
        chain.append(TrainedClassifier(mp, cfg.get("trained_threshold", 0.55)))
    if use_llm and llm_ok:
        chain.append(LLMClassifier(cfg))
    chain.append(TypeClassifier(cfg))       # always-answers fallback
    return chain


def classify_one(ctx: FileContext, chain, cfg) -> Decision | None:
    extract(ctx, cfg["ollama"].get("preview_chars", 1500))
    return run_chain(ctx, chain, min_confidence=0.0)


def final_placement(decision: Decision, cfg) -> tuple:
    """Apply the review policy: send a low-confidence guess to the review folder
    instead of a confident-looking wrong location. Rules/dupes/corrections are
    trusted and never diverted. Returns (folder, reason)."""
    rb = cfg.get("review_below", 0.0)
    if rb and decision.confidence < rb and decision.source not in ("rule", "dupe", "correction"):
        return Path(cfg.get("review_folder", "_Review")), decision.reason + " (low confidence)"
    return decision.folder, decision.reason


@dataclass
class Move:
    src: Path
    dst: Path
    reason: str
    source: str = ""
    confidence: float = 0.0


def dest_root_for(root: Path, cfg) -> Path:
    if cfg.get("destination"):
        return Path(cfg["destination"]).expanduser().resolve()
    return root


def iter_files(root: Path, recursive: bool, include_hidden: bool):
    if recursive:
        walker = (Path(dp) / f for dp, _, fs in os.walk(root) for f in fs)
    else:
        walker = (p for p in root.iterdir() if p.is_file())
    for p in walker:
        rel = p.relative_to(root)
        if not include_hidden and any(part.startswith(".") for part in rel.parts):
            continue
        yield p


def unique_destination(dst: Path, claimed: set) -> Path:
    if dst not in claimed and not dst.exists():
        claimed.add(dst)
        return dst
    stem, suffix = dst.stem, dst.suffix
    i = 1
    while True:
        cand = dst.with_name(f"{stem} ({i}){suffix}")
        if cand not in claimed and not cand.exists():
            claimed.add(cand)
            return cand
        i += 1


def build_plan(root: Path, cfg, chain, recursive=False,
               include_hidden=False) -> tuple[list[Move], Path]:
    dest_root = dest_root_for(root, cfg)
    claimed, moves = set(), []
    for f in sorted(iter_files(root, recursive, include_hidden)):
        rf = f.resolve()
        if f.suffix.lower() in IGNORE_SUFFIXES:
            continue
        ctx = FileContext.from_path(f)
        decision = classify_one(ctx, chain, cfg)
        if decision is None:
            continue
        folder, reason = final_placement(decision, cfg)
        target = dest_root / folder / f.name
        if target.resolve() == rf:
            continue
        target = unique_destination(target, claimed)
        moves.append(Move(rf, target, reason,
                          decision.source, decision.confidence))
    return moves, dest_root
