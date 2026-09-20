#!/usr/bin/env python3
"""
organize.py — an AI-powered, agentic file organizer.

A local Ollama model decides where each file belongs. It can:
  * organize a folder on demand (plan / run), or
  * WATCH a folder (e.g. ~/Downloads) and file things automatically the
    moment a download finishes.

Decision order for every file:
  1. Custom rules   — fast, deterministic shortcuts you define.
  2. Ollama model   — an LLM reads the name, metadata, and (for text files) a
                      content preview, then picks a destination folder.
  3. Type + date    — graceful fallback if Ollama is unavailable.

Every applied move is journaled, so `undo` can reverse it.
Only Python 3.8+ and a running Ollama are needed. No pip packages.

Usage:
    python3 organize.py init                          Write a starter config
    python3 organize.py plan  <folder> [--ai]         Preview (moves nothing)
    python3 organize.py run   <folder> [--ai]         Plan, then apply on approval
    python3 organize.py watch <folder>                Auto-organize new files (AI on)
    python3 organize.py undo                          Reverse the most recent run

Ollama options (see also the config file):
    --ai                 Use the Ollama model for plan/run (watch uses it by default)
    --no-ai              Disable the model (watch falls back to type sorting)
    --model NAME         Ollama model to use (default: llama3.2)
    --ollama-host URL    Ollama base URL (default: http://localhost:11434)

Watch options:
    --interval SEC       Poll interval (default 2)
    --settle SEC         How long a file's size must hold steady before it's
                         considered "done downloading" (default 4)
    --review             Prompt before each move instead of moving automatically
"""

import argparse
import json
import os
import re
import shutil
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #

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

# Extensions we'll read a short preview from to help the model.
TEXT_PREVIEW_EXTS = {
    "txt", "md", "csv", "tsv", "log", "json", "xml", "yaml", "yml", "ini",
    "cfg", "conf", "py", "js", "ts", "jsx", "tsx", "java", "c", "cpp", "h",
    "go", "rs", "rb", "php", "html", "css", "sh", "sql", "tex", "rtf",
}

# Partial-download / temp extensions we must never touch while writing.
IGNORE_SUFFIXES = {
    ".crdownload", ".part", ".partial", ".download", ".opdownload",
    ".tmp", ".temp", ".!qb", ".aria2",
}

DEFAULT_CONFIG = {
    "destination": None,               # None = organize in place
    "date_source": "modified",
    "use_date_subfolders": True,
    "date_format": "{year}/{month}",
    "unmatched_folder": "Other",
    "categories": DEFAULT_CATEGORIES,
    "ollama": {
        "enabled": False,              # plan/run: off unless --ai. watch: on unless --no-ai
        "host": "http://localhost:11434",
        "model": "llama3.2:1b",
        "read_content": True,          # peek inside text files for better guesses
        "preview_chars": 1500,
        "timeout": 60,
        "keep_alive": 0
    },
    "rules": [
        {
            "name": "Invoices & receipts",
            "any": ["invoice", "receipt", "statement"],
            "ext": ["pdf"],
            "dest": "Finance/{year}",
        },
        {
            "name": "Screenshots",
            "any": ["screenshot", "screen shot", "screen_shot"],
            "dest": "Images/Screenshots/{year}-{month}",
        },
    ],
}

STATE_DIR = Path.home() / ".file-organizer" / "journals"

# --------------------------------------------------------------------------- #
# Color helper
# --------------------------------------------------------------------------- #

class C:
    enabled = True

    @classmethod
    def wrap(cls, code, s):
        return s if not cls.enabled else f"\033[{code}m{s}\033[0m"

    @classmethod
    def dim(cls, s):    return cls.wrap("2", s)
    @classmethod
    def bold(cls, s):   return cls.wrap("1", s)
    @classmethod
    def green(cls, s):  return cls.wrap("32", s)
    @classmethod
    def yellow(cls, s): return cls.wrap("33", s)
    @classmethod
    def blue(cls, s):   return cls.wrap("36", s)
    @classmethod
    def red(cls, s):    return cls.wrap("31", s)
    @classmethod
    def magenta(cls, s): return cls.wrap("35", s)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

def load_config(path):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))  # deep copy
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


# --------------------------------------------------------------------------- #
# Ollama classifier
# --------------------------------------------------------------------------- #

SCHEMA = {
    "type": "object",
    "properties": {
        "folder": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["folder", "reason"],
}


def ollama_available(cfg):
    host = cfg["ollama"]["host"].rstrip("/")
    try:
        with urllib.request.urlopen(host + "/api/tags", timeout=4) as r:
            return r.status == 200
    except Exception:
        return False


def safe_rel(folder):
    """Turn a model-proposed path into a safe relative Path, or None."""
    if not folder:
        return None
    folder = str(folder).strip().replace("\\", "/")
    parts = []
    for raw in folder.split("/"):
        p = raw.strip().strip(".")
        if not p or p in (".", ".."):
            continue
        p = re.sub(r'[<>:"|?*\x00-\x1f]', "", p)      # strip illegal chars
        p = p.strip(" .")
        if p:
            parts.append(p)
    if not parts:
        return None
    if len(parts) > 6:                                 # keep depth sane
        parts = parts[:6]
    return Path(*parts)


def read_preview(path, cfg):
    o = cfg["ollama"]
    if not o.get("read_content", True):
        return None
    ext = path.suffix.lower().lstrip(".")
    if ext not in TEXT_PREVIEW_EXTS:
        return None
    try:
        with open(path, "r", errors="replace") as f:
            return f.read(o.get("preview_chars", 1500))
    except Exception:
        return None


def ollama_classify(path, cfg):
    """Ask the model where the file goes. Returns (Path, reason) or None."""
    o = cfg["ollama"]
    host = o["host"].rstrip("/")
    categories = ", ".join(cfg["categories"].keys())
    try:
        size_kb = max(1, path.stat().st_size // 1024)
        modified = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
    except OSError:
        size_kb, modified = 0, "?"
    preview = read_preview(path, cfg)

    system = (
        "You sort files into folders. Given a file's metadata (and maybe a "
        "content preview), reply with the best destination folder as a short "
        "relative path using forward slashes, plus a brief reason.\n"
        f"Prefer these top-level categories when they fit: {categories}.\n"
        "You may add a meaningful subfolder, e.g. 'Documents/Taxes/2024', "
        "'Finance/Invoices', 'Images/Screenshots', 'Code/Projects'. "
        "Base the choice on what the file actually is. Never use absolute "
        "paths, '..', or a drive letter. Keep it at most 3 levels deep."
    )
    user = (
        f"Filename: {path.name}\n"
        f"Extension: {path.suffix.lower().lstrip('.') or '(none)'}\n"
        f"Size: {size_kb} KB\n"
        f"Modified: {modified}\n\n"
        "Content preview:\n"
        + (preview if preview else "(binary or not previewed)")
        + '\n\nReturn JSON: {"folder": "...", "reason": "..."}'
    )

    body = {
        "model": o["model"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "format": SCHEMA,
        "options": {"temperature": 0},
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        host + "/api/chat", data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=o.get("timeout", 60)) as r:
            resp = json.load(r)
        content = resp["message"]["content"]
        parsed = json.loads(content)
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError, OSError):
        return None

    folder = safe_rel(parsed.get("folder", ""))
    if folder is None:
        return None
    reason = (parsed.get("reason") or "").strip().replace("\n", " ")[:120]
    return folder, ("ai \u00b7 " + reason if reason else "ai")


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #

def file_date(path, source):
    st = path.stat()
    ts = st.st_ctime if source == "created" else st.st_mtime
    return datetime.fromtimestamp(ts)


def category_for(ext, ext_index, cfg):
    return ext_index.get(ext, cfg.get("unmatched_folder", "Other"))


def rule_matches(path, ext, rule):
    name = path.name.lower()
    if rule.get("ext"):
        if ext not in [e.lower().lstrip(".") for e in rule["ext"]]:
            return False
    if rule.get("any") and not any(s.lower() in name for s in rule["any"]):
        return False
    if rule.get("all") and not all(s.lower() in name for s in rule["all"]):
        return False
    if rule.get("regex") and not re.search(rule["regex"], path.name):
        return False
    if rule.get("min_size_mb") is not None:
        if path.stat().st_size < rule["min_size_mb"] * 1024 * 1024:
            return False
    if rule.get("max_size_mb") is not None:
        if path.stat().st_size > rule["max_size_mb"] * 1024 * 1024:
            return False
    return True


def resolve_relative_dest(path, cfg, ext_index, use_ai=False, ai_ok=False):
    """Return (relative_dest_dir: Path, reason: str)."""
    ext = path.suffix.lower().lstrip(".")
    dt = file_date(path, cfg["date_source"])
    ctx = {
        "year": f"{dt.year:04d}", "month": f"{dt.month:02d}",
        "day": f"{dt.day:02d}", "ext": ext or "none", "stem": path.stem,
    }

    # 1) Custom rules first (deterministic, user-controlled).
    for rule in cfg.get("rules", []):
        if rule_matches(path, ext, rule):
            ctx["category"] = category_for(ext, ext_index, cfg)
            return Path(rule["dest"].format(**ctx)), f"rule \u00b7 {rule['name']}"

    # 2) Ask the model.
    if use_ai and ai_ok:
        res = ollama_classify(path, cfg)
        if res:
            return res

    # 3) Fallback: type, optionally nested by date.
    category = category_for(ext, ext_index, cfg)
    ctx["category"] = category
    parts = [category]
    reason = f"type \u00b7 {category}"
    if cfg.get("use_date_subfolders"):
        date_part = cfg["date_format"].format(**ctx)
        parts.append(date_part)
        reason += f" \u00b7 {date_part}"
    return Path(*parts), reason


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #

class Move:
    __slots__ = ("src", "dst", "reason")

    def __init__(self, src, dst, reason):
        self.src, self.dst, self.reason = src, dst, reason


def iter_files(root, recursive, include_hidden):
    if recursive:
        walker = (Path(dp) / f for dp, _, fs in os.walk(root) for f in fs)
    else:
        walker = (p for p in root.iterdir() if p.is_file())
    for p in walker:
        if not include_hidden and any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        yield p


def unique_destination(dst, claimed):
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


def dest_root_for(root, cfg):
    if cfg.get("destination"):
        return Path(cfg["destination"]).expanduser().resolve()
    return root


def build_plan(root, cfg, args, use_ai, ai_ok):
    ext_index = build_ext_index(cfg)
    dest_root = dest_root_for(root, cfg)
    self_path = Path(__file__).resolve()
    claimed, moves = set(), []

    files = sorted(iter_files(root, args.recursive, args.include_hidden))
    if use_ai and ai_ok and len(files) > 3:
        print(C.dim(f"Asking {cfg['ollama']['model']} about {len(files)} files\u2026"))

    for f in files:
        rf = f.resolve()
        if rf == self_path or f.suffix.lower() in IGNORE_SUFFIXES:
            continue
        rel_dir, reason = resolve_relative_dest(f, cfg, ext_index, use_ai, ai_ok)
        target = dest_root / rel_dir / f.name
        if target.resolve() == rf:
            continue
        target = unique_destination(target, claimed)
        moves.append(Move(rf, target, reason))
    return moves, dest_root


# --------------------------------------------------------------------------- #
# Presentation
# --------------------------------------------------------------------------- #

def print_plan(moves, dest_root):
    if not moves:
        print(C.dim("Nothing to organize — everything's already in place."))
        return
    groups = {}
    for m in moves:
        groups.setdefault(m.dst.parent, []).append(m)
    print(C.bold(f"\nProposed plan  ({len(moves)} file"
                 f"{'s' if len(moves) != 1 else ''} \u2192 "
                 f"{len(groups)} folder{'s' if len(groups) != 1 else ''})\n"))
    for folder in sorted(groups, key=lambda p: str(p)):
        try:
            shown = folder.relative_to(dest_root)
        except ValueError:
            shown = folder
        items = groups[folder]
        print(C.blue(f"  {shown}{os.sep}") + C.dim(f"   ({len(items)})"))
        for m in items:
            print(f"    {m.src.name}  " + C.dim(f"[{m.reason}]"))
        print()


# --------------------------------------------------------------------------- #
# Applying
# --------------------------------------------------------------------------- #

def do_move(m):
    m.dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(m.src), str(m.dst))


def prompt_each(moves, dest_root):
    executed, yes_to_all = [], False
    print(C.dim("\nReviewing each move — [y]es  [n]o  [a]ll  [e]dit dest  [q]uit\n"))
    for i, m in enumerate(moves, 1):
        try:
            rel = m.dst.relative_to(dest_root)
        except ValueError:
            rel = m.dst
        if not yes_to_all:
            print(f"  ({i}/{len(moves)}) {C.bold(m.src.name)}")
            print(f"        \u2192 {C.blue(str(rel))}   " + C.dim(f"[{m.reason}]"))
            try:
                choice = input("        move? [y/n/a/e/q] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n" + C.yellow("Stopped."))
                break
            if choice in ("q", "quit"):
                print(C.yellow("Stopped — no further moves."))
                break
            if choice in ("n", "no", ""):
                print(C.dim("        skipped\n"))
                continue
            if choice in ("a", "all"):
                yes_to_all = True
            elif choice in ("e", "edit"):
                new = input("        new destination folder (relative): ").strip()
                if new:
                    m.dst = dest_root / new / m.src.name
            elif choice not in ("y", "yes"):
                print(C.dim("        skipped\n"))
                continue
        try:
            do_move(m)
            executed.append(m)
            print(C.green("        moved\n"))
        except Exception as e:  # noqa: BLE001
            print(C.red(f"        failed: {e}\n"))
    return executed


def apply_all(moves):
    executed = []
    for m in moves:
        try:
            do_move(m)
            executed.append(m)
        except Exception as e:  # noqa: BLE001
            print(C.red(f"  failed to move {m.src.name}: {e}"))
    return executed


# --------------------------------------------------------------------------- #
# Journaling / undo
# --------------------------------------------------------------------------- #

def write_journal(executed):
    if not executed:
        return None
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = STATE_DIR / f"run-{stamp}.json"
    with open(path, "w") as f:
        json.dump({"timestamp": stamp,
                   "moves": [{"from": str(m.src), "to": str(m.dst)} for m in executed]},
                  f, indent=2)
    return path


class JournalWriter:
    """Incrementally journals moves during a long-running watch session."""
    def __init__(self):
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = STATE_DIR / f"watch-{stamp}.json"
        self.moves = []
        self._flush(stamp)

    def _flush(self, stamp):
        with open(self.path, "w") as f:
            json.dump({"timestamp": stamp, "moves": self.moves}, f, indent=2)

    def add(self, m):
        self.moves.append({"from": str(m.src), "to": str(m.dst)})
        self._flush(self.path.stem.replace("watch-", ""))


def latest_journal():
    if not STATE_DIR.exists():
        return None
    js = sorted(STATE_DIR.glob("*.json"))
    return js[-1] if js else None


def undo_last():
    j = latest_journal()
    if not j:
        print(C.yellow("No runs to undo."))
        return
    with open(j) as f:
        data = json.load(f)
    moves = data["moves"]
    print(C.bold(f"Undoing {j.stem}  ({len(moves)} files)\n"))
    restored = 0
    for m in reversed(moves):
        src, dst = Path(m["from"]), Path(m["to"])
        if not dst.exists():
            print(C.dim(f"  skip (missing): {dst.name}"))
            continue
        if src.exists():
            print(C.yellow(f"  skip (original path occupied): {src}"))
            continue
        try:
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst), str(src))
            restored += 1
        except Exception as e:  # noqa: BLE001
            print(C.red(f"  failed: {dst.name}: {e}"))
    pruned = 0
    for d in sorted({Path(m["to"]).parent for m in moves},
                    key=lambda p: len(p.parts), reverse=True):
        p = d
        while p.exists() and p.is_dir() and not any(p.iterdir()):
            try:
                p.rmdir(); pruned += 1; p = p.parent
            except OSError:
                break
    j.rename(j.with_suffix(".undone"))
    print(C.green(f"\nRestored {restored}/{len(moves)} file(s)."))
    if pruned:
        print(C.dim(f"Removed {pruned} empty folder(s)."))


# --------------------------------------------------------------------------- #
# Watch mode
# --------------------------------------------------------------------------- #

def should_ignore_watch(p):
    if p.name.startswith(".") or p.name.startswith("~"):
        return True
    if p.suffix.lower() in IGNORE_SUFFIXES:
        return True
    return False


def watch_loop(root, cfg, args, use_ai, ai_ok):
    ext_index = build_ext_index(cfg)
    dest_root = dest_root_for(root, cfg)
    journal = JournalWriter()

    interval = max(1, args.interval)
    settle = max(1, args.settle)
    mode = "reviewing each" if args.review else "auto-moving"
    engine = f"{cfg['ollama']['model']} (AI)" if (use_ai and ai_ok) else "type sorting"

    print(C.bold(f"Watching {root}"))
    print(C.dim(f"  engine: {engine}   mode: {mode}   "
                f"settle: {settle}s   dest: {dest_root}"))
    print(C.dim("  Press Ctrl-C to stop.\n"))

    stability = {}      # path -> (size, last_change_monotonic)
    processed = set()

    try:
        while True:
            try:
                entries = [p for p in root.iterdir() if p.is_file()]
            except FileNotFoundError:
                print(C.red("Watched folder disappeared. Stopping."))
                break

            now = time.monotonic()
            for f in entries:
                key = str(f.resolve())
                if key in processed or should_ignore_watch(f):
                    continue
                try:
                    size = f.stat().st_size
                except OSError:
                    continue
                prev = stability.get(key)
                if prev is None or prev[0] != size:
                    stability[key] = (size, now)          # still changing
                    continue
                if now - prev[1] < settle:                # not settled yet
                    continue

                # Settled — classify and (maybe) move.
                rel_dir, reason = resolve_relative_dest(
                    f, cfg, ext_index, use_ai, ai_ok)
                target = unique_destination(dest_root / rel_dir / f.name, set())
                m = Move(f.resolve(), target, reason)
                ts = time.strftime("%H:%M:%S")
                try:
                    rel = target.relative_to(dest_root)
                except ValueError:
                    rel = target

                if args.review:
                    print(f"{C.dim(ts)}  {C.bold(f.name)}")
                    print(f"          \u2192 {C.blue(str(rel))}  {C.dim('['+reason+']')}")
                    try:
                        ans = input("          move? [y/N] ").strip().lower()
                    except (EOFError, KeyboardInterrupt):
                        raise KeyboardInterrupt
                    if ans not in ("y", "yes"):
                        print(C.dim("          skipped"))
                        processed.add(key)
                        continue

                try:
                    do_move(m)
                    journal.add(m)
                    processed.add(key)
                    print(f"{C.dim(ts)}  {C.green('moved')}  {f.name}  "
                        f"{C.dim('→')} {C.blue(str(rel))}  {C.dim('['+reason+']')}")
                except Exception as e:  # noqa: BLE001
                    print(f"{C.dim(ts)}  {C.red('failed')} {f.name}: {e}")
                    processed.add(key)

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n" + C.yellow(f"Stopped. Journaled to {journal.path}"))
        print(C.dim(f"Undo this session with:  python3 {Path(__file__).name} undo"))


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def resolve_ai(cfg, args, default_on):
    """Decide whether to use AI and whether Ollama is reachable."""
    if getattr(args, "no_ai", False):
        return False, False
    want = getattr(args, "ai", False) or cfg["ollama"]["enabled"] or default_on
    if not want:
        return False, False
    ok = ollama_available(cfg)
    if not ok:
        print(C.yellow(
            f"\u26a0  Ollama not reachable at {cfg['ollama']['host']}. "
            "Falling back to type-based sorting."))
        print(C.dim("   Start it with 'ollama serve' and pull a model, e.g. "
                    f"'ollama pull {cfg['ollama']['model']}'."))
    return True, ok


def apply_cli_overrides(cfg, args):
    if getattr(args, "dest", None):
        cfg["destination"] = args.dest
    if getattr(args, "no_date", False):
        cfg["use_date_subfolders"] = False
    if getattr(args, "date_source", None):
        cfg["date_source"] = args.date_source
    if getattr(args, "model", None):
        cfg["ollama"]["model"] = args.model
    if getattr(args, "ollama_host", None):
        cfg["ollama"]["host"] = args.ollama_host


def _load(args, default_on):
    cfg = load_config(args.config or default_config_path())
    apply_cli_overrides(cfg, args)
    root = Path(args.folder).expanduser().resolve()
    if not root.is_dir():
        print(C.red(f"Not a folder: {root}"))
        sys.exit(1)
    use_ai, ai_ok = resolve_ai(cfg, args, default_on)
    return cfg, root, use_ai, ai_ok


def cmd_init(args):
    path = Path(args.config or "organizer.config.json")
    if path.exists():
        print(C.yellow(f"{path} already exists — not overwriting."))
        return
    with open(path, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    print(C.green(f"Wrote starter config to {path}"))
    print(C.dim("Set ollama.enabled=true (or pass --ai) and edit 'rules' to taste."))


def cmd_plan(args):
    cfg, root, use_ai, ai_ok = _load(args, default_on=False)
    moves, dest_root = build_plan(root, cfg, args, use_ai, ai_ok)
    print_plan(moves, dest_root)


def cmd_run(args):
    cfg, root, use_ai, ai_ok = _load(args, default_on=False)
    moves, dest_root = build_plan(root, cfg, args, use_ai, ai_ok)
    print_plan(moves, dest_root)
    if not moves:
        return
    if args.auto:
        executed = apply_all(moves)
    else:
        try:
            ans = input(C.bold("Apply this plan? [a]ll  [r]eview each  [c]ancel: ")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n" + C.yellow("Cancelled.")); return
        if ans in ("a", "all"):
            executed = apply_all(moves)
        elif ans in ("r", "review"):
            executed = prompt_each(moves, dest_root)
        else:
            print(C.yellow("Cancelled — nothing moved.")); return
    write_journal(executed)
    print(C.green(f"\nDone. Moved {len(executed)} file(s)."))
    print(C.dim(f"Undo with:  python3 {Path(__file__).name} undo"))


def cmd_watch(args):
    cfg, root, use_ai, ai_ok = _load(args, default_on=True)
    watch_loop(root, cfg, args, use_ai, ai_ok)


def cmd_undo(args):
    undo_last()


# --------------------------------------------------------------------------- #
# CLI plumbing
# --------------------------------------------------------------------------- #

def default_config_path():
    p = Path("organizer.config.json")
    return str(p) if p.exists() else None


def build_parser():
    p = argparse.ArgumentParser(
        prog="organize.py",
        description="AI-powered file organizer (Ollama). Proposes moves and can watch a folder.")
    sub = p.add_subparsers(dest="command", required=True)

    def ai_opts(sp):
        sp.add_argument("--ai", action="store_true", help="Use the Ollama model")
        sp.add_argument("--no-ai", action="store_true", help="Disable the model")
        sp.add_argument("--model", help="Ollama model name (default llama3.2)")
        sp.add_argument("--ollama-host", help="Ollama base URL")

    def common(sp):
        sp.add_argument("folder", help="Folder to organize")
        sp.add_argument("--config", help="Path to JSON config")
        sp.add_argument("--dest", help="Destination root (default: in place)")
        sp.add_argument("--recursive", action="store_true", help="Descend into subfolders")
        sp.add_argument("--no-date", action="store_true", help="Skip year/month subfolders")
        sp.add_argument("--date-source", choices=["modified", "created"])
        sp.add_argument("--include-hidden", action="store_true")
        sp.add_argument("--no-color", action="store_true")

    sp = sub.add_parser("init", help="Write a starter config file")
    sp.add_argument("--config"); sp.add_argument("--no-color", action="store_true")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("plan", help="Preview the plan without moving anything")
    common(sp); ai_opts(sp); sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("run", help="Propose a plan and apply it after approval")
    common(sp); ai_opts(sp)
    sp.add_argument("--auto", action="store_true", help="Apply without prompting")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("watch", help="Auto-organize new files as they arrive (AI on)")
    common(sp); ai_opts(sp)
    sp.add_argument("--interval", type=int, default=2, help="Poll seconds (default 2)")
    sp.add_argument("--settle", type=int, default=4,
                    help="Seconds a file's size must be stable before moving (default 4)")
    sp.add_argument("--review", action="store_true", help="Prompt before each move")
    sp.set_defaults(func=cmd_watch)

    sp = sub.add_parser("undo", help="Reverse the most recent run")
    sp.add_argument("--no-color", action="store_true")
    sp.set_defaults(func=cmd_undo)

    return p


def main():
    args = build_parser().parse_args()
    if getattr(args, "no_color", False) or not sys.stdout.isatty():
        C.enabled = False
    args.func(args)


if __name__ == "__main__":
    main()
