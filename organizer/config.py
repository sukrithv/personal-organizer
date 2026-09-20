"""
config.py — defaults + loading. Ported from the single-file tool.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_CATEGORIES = {
    "Images": ["jpg", "jpeg", "png", "gif", "bmp", "svg", "webp", "heic",
               "tiff", "tif", "ico", "raw", "cr2", "nef"],
    "Documents": ["pdf", "doc", "docx", "txt", "rtf", "odt", "md", "tex", "pages"],
    "Spreadsheets": ["xls", "xlsx", "csv", "tsv", "ods", "numbers"],
    "Presentations": ["ppt", "pptx", "odp", "key"],
    "Audio": ["mp3", "wav", "flac", "aac", "ogg", "m4a", "wma", "aiff"],
    "Video": ["mp4", "mov", "avi", "mkv", "wmv", "flv", "webm", "m4v", "mpg", "mpeg"],
    "Archives": ["zip", "tar", "gz", "tgz", "rar", "7z", "bz2", "xz", "iso", "dmg"],
    "Code": ["py", "js", "ts", "jsx", "tsx", "java", "c", "cpp", "h", "hpp",
             "go", "rs", "rb", "php", "html", "css", "json", "xml", "yaml",
             "yml", "sh", "sql", "ipynb"],
    "Installers": ["exe", "msi", "deb", "rpm", "pkg", "apk", "appimage"],
    "Ebooks": ["epub", "mobi", "azw", "azw3"],
    "Fonts": ["ttf", "otf", "woff", "woff2"],
    "Data": ["db", "sqlite", "sqlite3", "parquet", "log"],
}

IGNORE_SUFFIXES = {
    ".crdownload", ".part", ".partial", ".download", ".opdownload",
    ".tmp", ".temp", ".!qb", ".aria2",
}

DEFAULT_CONFIG = {
    "destination": None,
    "date_source": "modified",
    "use_date_subfolders": True,
    "date_format": "{year}/{month}",
    "unmatched_folder": "Other",
    "model_path": None,
    "db_path": None,
    "dedupe": {"enabled": False, "folder": "_Duplicates"},
    "trained_threshold": 0.55,
    "review_below": 0.0,
    "review_folder": "_Review",
    "categories": DEFAULT_CATEGORIES,
    "ollama": {
        "enabled": False,
        "host": "http://localhost:11434",
        "model": "llama3.2:1b",
        "keep_alive": "0",
        "read_content": True,
        "preview_chars": 1500,
        "timeout": 60,
    },
    "rules": [
        {"name": "Invoices & receipts", "any": ["invoice", "receipt", "statement"],
         "ext": ["pdf"], "dest": "Finance/{year}"},
        {"name": "Screenshots", "any": ["screenshot", "screen shot", "screen_shot"],
         "dest": "Images/Screenshots/{year}-{month}"},
    ],
}


def load_config(path=None):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if path and Path(path).exists():
        with open(path) as f:
            user = json.load(f)
        for k, v in user.items():
            if k == "ollama" and isinstance(v, dict):
                cfg["ollama"].update(v)
            else:
                cfg[k] = v
    return cfg


def build_ext_index(cfg):
    index = {}
    for category, exts in cfg["categories"].items():
        for e in exts:
            index[e.lower().lstrip(".")] = category
    return index


def default_config_path():
    p = Path("organizer.config.json")
    return str(p) if p.exists() else None
