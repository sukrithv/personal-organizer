"""
features.py — turn a FileContext into the text a model learns from / predicts on.

The CorrectionLog needs to store features at correction time, and the trained
classifier must use this same function at train and predict time, so it lives
in one place.
"""

from __future__ import annotations

from .core import FileContext


def featurize(ctx: FileContext, preview_chars: int = 800) -> str:
    parts = [ctx.name, f"ext_{ctx.ext or 'none'}"]
    for key in ("camera", "artist", "album", "genre"):
        if ctx.metadata.get(key):
            parts.append(f"{key}_{ctx.metadata[key]}")
    if ctx.text_preview:
        parts.append(ctx.text_preview[:preview_chars])
    return " ".join(parts)
