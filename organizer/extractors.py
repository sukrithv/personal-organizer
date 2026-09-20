"""
extractors.py — turn raw bytes into signal for the classifiers.

Phase 2: real content extraction. Each extractor lazily imports its library so
the package still works with nothing installed (a missing lib just means that
file gets no preview). Install what you need via the pyproject extras:
    pip install -e ".[pdf,images,audio]"

Extractors fill:
    ctx.text_preview   text the LLM / trained model can read
    ctx.metadata       structured hints, e.g. {"taken": datetime, "pages": 3}
"""

from __future__ import annotations

from datetime import datetime

from .core import FileContext

_REGISTRY = {}
PREVIEW_CHARS = 1500

TEXT_EXTS = ("txt", "md", "csv", "tsv", "log", "json", "yaml", "yml", "ini",
             "cfg", "conf", "py", "js", "ts", "jsx", "tsx", "java", "c", "cpp",
             "h", "go", "rs", "rb", "php", "html", "css", "sh", "sql", "tex")


def register(*exts):
    def deco(fn):
        for e in exts:
            _REGISTRY[e.lower()] = fn
        return fn
    return deco


def extract(ctx: FileContext, preview_chars: int = PREVIEW_CHARS) -> FileContext:
    fn = _REGISTRY.get(ctx.ext)
    if fn is None:
        return ctx
    try:
        fn(ctx, preview_chars)
    except Exception:
        pass  # best-effort: a failed extract just means less signal
    return ctx


# --------------------------------------------------------------------------- #
# Plaintext
# --------------------------------------------------------------------------- #

@register(*TEXT_EXTS)
def extract_text(ctx: FileContext, preview_chars: int) -> None:
    with open(ctx.path, errors="replace") as f:
        ctx.text_preview = f.read(preview_chars)


# --------------------------------------------------------------------------- #
# PDF  (pdfminer.six)
# --------------------------------------------------------------------------- #

@register("pdf")
def extract_pdf(ctx: FileContext, preview_chars: int) -> None:
    from pdfminer.high_level import extract_text as pdf_text  # lazy

    text = pdf_text(str(ctx.path), maxpages=2) or ""
    text = " ".join(text.split())          # collapse whitespace
    if text:
        ctx.text_preview = text[:preview_chars]
    else:
        ctx.metadata["scanned"] = True     # no text layer -> likely a scan (OCR later)


# --------------------------------------------------------------------------- #
# DOCX  (stdlib only — it's a zip of XML)
# --------------------------------------------------------------------------- #

@register("docx")
def extract_docx(ctx: FileContext, preview_chars: int) -> None:
    import re
    import zipfile

    with zipfile.ZipFile(ctx.path) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    # drop tags, keep text; paragraph breaks -> spaces
    text = re.sub(r"<[^>]+>", " ", xml)
    text = " ".join(text.split())
    if text:
        ctx.text_preview = text[:preview_chars]


# --------------------------------------------------------------------------- #
# Images  (Pillow) — EXIF "date taken" beats file mtime for photos
# --------------------------------------------------------------------------- #

@register("jpg", "jpeg", "png", "heic", "tiff", "tif", "webp")
def extract_image_exif(ctx: FileContext, preview_chars: int) -> None:
    from PIL import ExifTags, Image  # lazy

    with Image.open(ctx.path) as im:
        ctx.metadata["dimensions"] = f"{im.width}x{im.height}"
        exif = im.getexif()
        if not exif:
            return
        tag = {v: k for k, v in ExifTags.TAGS.items()}
        dt = exif.get(tag.get("DateTimeOriginal")) or exif.get(tag.get("DateTime"))
        if dt:
            try:
                ctx.metadata["taken"] = datetime.strptime(str(dt), "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass
        model = exif.get(tag.get("Model"))
        if model:
            ctx.metadata["camera"] = str(model).strip()


# --------------------------------------------------------------------------- #
# Audio  (mutagen)
# --------------------------------------------------------------------------- #

@register("mp3", "flac", "m4a", "ogg", "wav")
def extract_audio_tags(ctx: FileContext, preview_chars: int) -> None:
    import mutagen  # lazy

    m = mutagen.File(ctx.path, easy=True)
    if not m:
        return
    for key in ("artist", "album", "date", "genre"):
        val = m.get(key)
        if val:
            ctx.metadata[key] = val[0] if isinstance(val, list) else val
    bits = [f"{k}: {v}" for k, v in ctx.metadata.items()
            if k in ("artist", "album", "genre")]
    if bits:
        ctx.text_preview = " | ".join(bits)
