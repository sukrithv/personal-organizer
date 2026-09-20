"""
cli.py — commands: init / plan / run / watch / undo / history / correct.

Phase 2: moves are recorded in SQLite (cross-session undo + history), optional
--dedupe routes exact duplicates to a holding folder, and `correct` captures a
labeled example for Phase 3 training.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from .classifiers import ollama_available
from .config import DEFAULT_CONFIG, IGNORE_SUFFIXES, default_config_path, load_config
from .core import FileContext
from .dedupe import DuplicateIndex
from .extractors import extract
from .features import featurize
from .pipeline import (
    Move,
    build_chain,
    build_plan,
    classify_one,
    dest_root_for,
    final_placement,
    unique_destination,
)
from .store import DEFAULT_DB, CorrectionLog, TransactionLog
from .training import DEFAULT_MODEL_PATH, bootstrap_dataset, retrain_with_corrections, train


class C:
    enabled = True

    @classmethod
    def w(cls, code, s):
        return s if not cls.enabled else f"\033[{code}m{s}\033[0m"

    @classmethod
    def dim(cls, s):    return cls.w("2", s)
    @classmethod
    def bold(cls, s):   return cls.w("1", s)
    @classmethod
    def green(cls, s):  return cls.w("32", s)
    @classmethod
    def yellow(cls, s): return cls.w("33", s)
    @classmethod
    def blue(cls, s):   return cls.w("36", s)
    @classmethod
    def red(cls, s):    return cls.w("31", s)


def db_path_for(cfg):
    return cfg.get("db_path") or DEFAULT_DB


# --------------------------------------------------------------------------- #
# shared setup
# --------------------------------------------------------------------------- #

def apply_overrides(cfg, args):
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
    if getattr(args, "dedupe", False):
        cfg["dedupe"]["enabled"] = True
    if getattr(args, "review_below", None) is not None:
        cfg["review_below"] = args.review_below


def resolve_llm(cfg, args, default_on):
    if getattr(args, "no_ai", False):
        return False, False
    want = getattr(args, "ai", False) or cfg["ollama"]["enabled"] or default_on
    if not want:
        return False, False
    ok = ollama_available(cfg["ollama"]["host"])
    if not ok:
        print(C.yellow(f"\u26a0  Ollama not reachable at {cfg['ollama']['host']}. "
                       "Falling back to type-based sorting."))
        print(C.dim(f"   Try 'ollama serve' and 'ollama pull {cfg['ollama']['model']}'."))
    return True, ok


def setup(args, default_on):
    cfg = load_config(args.config or default_config_path())
    apply_overrides(cfg, args)
    root = Path(args.folder).expanduser().resolve()
    if not root.is_dir():
        print(C.red(f"Not a folder: {root}"))
        sys.exit(1)
    use_llm, llm_ok = resolve_llm(cfg, args, default_on)
    dupe_index = None
    if cfg["dedupe"]["enabled"]:
        dupe_index = DuplicateIndex()
        seed = dest_root_for(root, cfg)
        print(C.dim(f"Indexing {seed} for duplicates…"))
        dupe_index.scan(seed)
    chain = build_chain(cfg, use_llm, llm_ok, dupe_index)
    return cfg, root, chain


# --------------------------------------------------------------------------- #
# presentation
# --------------------------------------------------------------------------- #

def print_plan(moves, dest_root):
    if not moves:
        print(C.dim("Nothing to organize \u2014 everything's already in place."))
        return
    groups = {}
    for m in moves:
        groups.setdefault(m.dst.parent, []).append(m)
    print(C.bold(f"\nProposed plan  ({len(moves)} file{'s' if len(moves) != 1 else ''} "
                 f"\u2192 {len(groups)} folder{'s' if len(groups) != 1 else ''})\n"))
    for folder in sorted(groups, key=lambda p: str(p)):
        try:
            shown = folder.relative_to(dest_root)
        except ValueError:
            shown = folder
        print(C.blue(f"  {shown}{os.sep}") + C.dim(f"   ({len(groups[folder])})"))
        for m in groups[folder]:
            print(f"    {m.src.name}  " + C.dim(f"[{m.reason}]"))
        print()


def do_move(m):
    m.dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(m.src), str(m.dst))


def apply_all(moves, tx):
    executed = []
    for m in moves:
        try:
            do_move(m)
            tx.record(m)
            executed.append(m)
        except Exception as e:  # noqa: BLE001
            print(C.red(f"  failed to move {m.src.name}: {e}"))
    return executed


def prompt_each(moves, dest_root, tx):
    executed, yes_all = [], False
    print(C.dim("\nReviewing \u2014 [y]es [n]o [a]ll [e]dit [q]uit\n"))
    for i, m in enumerate(moves, 1):
        try:
            rel = m.dst.relative_to(dest_root)
        except ValueError:
            rel = m.dst
        if not yes_all:
            print(f"  ({i}/{len(moves)}) {C.bold(m.src.name)}")
            print(f"        \u2192 {C.blue(str(rel))}   " + C.dim(f"[{m.reason}]"))
            try:
                choice = input("        move? [y/n/a/e/q] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n" + C.yellow("Stopped.")); break
            if choice in ("q", "quit"):
                print(C.yellow("Stopped.")); break
            if choice in ("n", "no", ""):
                print(C.dim("        skipped\n")); continue
            if choice in ("a", "all"):
                yes_all = True
            elif choice in ("e", "edit"):
                new = input("        new folder (relative): ").strip()
                if new:
                    m.dst = dest_root / new / m.src.name
            elif choice not in ("y", "yes"):
                print(C.dim("        skipped\n")); continue
        try:
            do_move(m); tx.record(m); executed.append(m); print(C.green("        moved\n"))
        except Exception as e:  # noqa: BLE001
            print(C.red(f"        failed: {e}\n"))
    return executed


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_init(args):
    path = Path(args.config or "organizer.config.json")
    if path.exists():
        print(C.yellow(f"{path} already exists \u2014 not overwriting.")); return
    with open(path, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    print(C.green(f"Wrote starter config to {path}"))


def cmd_plan(args):
    cfg, root, chain = setup(args, default_on=False)
    moves, dest_root = build_plan(root, cfg, chain, args.recursive, args.include_hidden)
    print_plan(moves, dest_root)


def cmd_run(args):
    cfg, root, chain = setup(args, default_on=False)
    moves, dest_root = build_plan(root, cfg, chain, args.recursive, args.include_hidden)
    print_plan(moves, dest_root)
    if not moves:
        return
    tx = TransactionLog(db_path_for(cfg))
    if args.auto:
        executed = apply_all(moves, tx)
    else:
        try:
            ans = input(C.bold("Apply? [a]ll [r]eview [c]ancel: ")).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n" + C.yellow("Cancelled.")); return
        if ans in ("a", "all"):
            executed = apply_all(moves, tx)
        elif ans in ("r", "review"):
            executed = prompt_each(moves, dest_root, tx)
        else:
            print(C.yellow("Cancelled.")); return
    print(C.green(f"\nDone. Moved {len(executed)} file(s).  ") + C.dim("undo: organizer undo"))


def cmd_watch(args):
    cfg, root, chain = setup(args, default_on=True)
    dest_root = dest_root_for(root, cfg)
    tx = TransactionLog(db_path_for(cfg))
    settle, interval = max(1, args.settle), max(1, args.interval)
    print(C.bold(f"Watching {root}"))
    print(C.dim(f"  settle {settle}s  poll {interval}s  dest {dest_root}  (Ctrl-C to stop)\n"))
    stability, processed = {}, set()
    try:
        while True:
            try:
                entries = [p for p in root.iterdir() if p.is_file()]
            except FileNotFoundError:
                print(C.red("Watched folder gone.")); break
            now = time.monotonic()
            for f in entries:
                key = str(f.resolve())
                if key in processed or f.name.startswith((".", "~")):
                    continue
                if f.suffix.lower() in IGNORE_SUFFIXES:
                    continue
                try:
                    size = f.stat().st_size
                except OSError:
                    continue
                prev = stability.get(key)
                if prev is None or prev[0] != size:
                    stability[key] = (size, now); continue
                if now - prev[1] < settle:
                    continue
                ctx = FileContext.from_path(f)
                decision = classify_one(ctx, chain, cfg)
                if decision is None:
                    processed.add(key); continue
                folder, reason = final_placement(decision, cfg)
                target = unique_destination(dest_root / folder / f.name, set())
                m = Move(f.resolve(), target, reason,
                         decision.source, decision.confidence)
                try:
                    rel = target.relative_to(dest_root)
                except ValueError:
                    rel = target
                ts = time.strftime("%H:%M:%S")
                if args.review:
                    print(f"{C.dim(ts)}  {C.bold(f.name)} \u2192 {C.blue(str(rel))} "
                          + C.dim(f"[{reason}]"))
                    try:
                        if input("          move? [y/N] ").strip().lower() not in ("y", "yes"):
                            processed.add(key); print(C.dim("          skipped")); continue
                    except (EOFError, KeyboardInterrupt):
                        raise KeyboardInterrupt from None
                try:
                    do_move(m); tx.record(m); processed.add(key)
                    print(f"{C.dim(ts)}  {C.green('moved')}  {f.name}  {C.dim('→')} "
                          f"{C.blue(str(rel))}  " + C.dim(f"[{reason}]"))
                except Exception as e:  # noqa: BLE001
                    processed.add(key)
                    print(f"{C.dim(ts)}  {C.red('failed')} {f.name}: {e}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n" + C.yellow("Stopped."))


def cmd_undo(args):
    cfg = load_config(getattr(args, "config", None) or default_config_path())
    TransactionLog(db_path_for(cfg)).undo_last(
        log=lambda s: print(C.green(s) if s.startswith("Restored") else s))


def cmd_history(args):
    cfg = load_config(getattr(args, "config", None) or default_config_path())
    tx = TransactionLog(db_path_for(cfg))
    print(C.bold("\nBy classifier:"))
    for r in tx.source_stats():
        conf = f"{r['avg_conf']:.0%}" if r["avg_conf"] is not None else "n/a"
        print(f"  {C.blue(r['source'] or '?'):<12} {r['n']:>4} moves   avg conf {conf}")
    print(C.bold(f"\nLast {args.limit} moves:"))
    for r in tx.history(args.limit):
        flag = C.dim("(undone)") if r["undone"] else ""
        print(f"  {C.dim(r['ts'])}  {Path(r['src']).name}  {C.dim('→')} "
              f"{Path(r['dst']).parent.name}/  [{r['source']}] {flag}")
    print()


def cmd_correct(args):
    cfg = load_config(getattr(args, "config", None) or default_config_path())
    src = Path(args.file).expanduser().resolve()
    if not src.is_file():
        print(C.red(f"Not a file: {src}")); return
    ctx = extract(FileContext.from_path(src), cfg["ollama"].get("preview_chars", 1500))
    was = src.parent.name
    target_dir = Path(args.folder).expanduser()
    if not target_dir.is_absolute():
        target_dir = src.parent / args.folder     # relative to where the file is now
    target = unique_destination(target_dir / src.name, set())
    tx = TransactionLog(db_path_for(cfg))
    try:
        do_move(Move(src, target, f"correction -> {args.folder}", "correction", 1.0))
        tx.record(Move(src, target, "correction", "correction", 1.0))
    except Exception as e:  # noqa: BLE001
        print(C.red(f"move failed: {e}")); return
    CorrectionLog(db_path_for(cfg)).record(featurize(ctx), args.folder, was=was)
    print(C.green(f"Moved and logged correction: {src.name} \u2192 {args.folder}"))
    print(C.dim("This example will train the Phase 3 classifier."))


def cmd_train(args):
    cfg = load_config(getattr(args, "config", None) or default_config_path())
    root = Path(args.organized).expanduser().resolve()
    if not root.is_dir():
        print(C.red(f"Not a folder: {root}")); return
    model_out = Path(args.model_path).expanduser() if args.model_path else DEFAULT_MODEL_PATH
    print(C.dim(f"Learning from your organized folders under {root} …"))
    try:
        if args.with_corrections:
            stats = retrain_with_corrections(
                root, CorrectionLog(db_path_for(cfg)), model_out,
                label_depth=args.label_depth, log=lambda s: print(C.dim(str(s))))
        else:
            data = bootstrap_dataset(root, label_depth=args.label_depth)
            stats = train(list(data), model_out,
                          min_per_class=args.min_per_class,
                          log=lambda s: print(C.dim(str(s))))
    except ValueError as e:
        print(C.yellow(str(e))); return
    acc = f"{stats['accuracy']:.0%}" if stats["accuracy"] is not None else "n/a (small set)"
    print(C.green(f"\nTrained on {stats['examples']} files across "
                  f"{stats['classes']} folders. Held-out accuracy: {acc}."))
    if stats["dropped"]:
        print(C.dim(f"  ({stats['dropped']} files in too-rare folders were skipped)"))
    print(C.dim(f"  Model saved to {stats['path']}"))
    print(C.dim("  It now joins the chain automatically (ahead of the LLM) on your next run."))


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #

def build_parser():
    p = argparse.ArgumentParser(prog="organizer",
                                description="AI-powered file organizer.")
    sub = p.add_subparsers(dest="command", required=True)

    def ai_opts(sp):
        sp.add_argument("--ai", action="store_true")
        sp.add_argument("--no-ai", action="store_true")
        sp.add_argument("--model")
        sp.add_argument("--ollama-host")

    def common(sp):
        sp.add_argument("folder")
        sp.add_argument("--config")
        sp.add_argument("--dest")
        sp.add_argument("--recursive", action="store_true")
        sp.add_argument("--no-date", action="store_true")
        sp.add_argument("--date-source", choices=["modified", "created"])
        sp.add_argument("--include-hidden", action="store_true")
        sp.add_argument("--dedupe", action="store_true")
        sp.add_argument("--review-below", type=float, default=None,
                        help="Route guesses below this confidence to the review folder")
        sp.add_argument("--no-color", action="store_true")

    s = sub.add_parser("init"); s.add_argument("--config")
    s.add_argument("--no-color", action="store_true"); s.set_defaults(func=cmd_init)

    s = sub.add_parser("plan"); common(s); ai_opts(s); s.set_defaults(func=cmd_plan)

    s = sub.add_parser("run"); common(s); ai_opts(s)
    s.add_argument("--auto", action="store_true"); s.set_defaults(func=cmd_run)

    s = sub.add_parser("watch"); common(s); ai_opts(s)
    s.add_argument("--interval", type=int, default=2)
    s.add_argument("--settle", type=int, default=4)
    s.add_argument("--review", action="store_true"); s.set_defaults(func=cmd_watch)

    s = sub.add_parser("train", help="Learn from your already-organized folders")
    s.add_argument("organized", help="A folder you've already organized")
    s.add_argument("--config")
    s.add_argument("--model-path")
    s.add_argument("--label-depth", type=int, default=2)
    s.add_argument("--min-per-class", type=int, default=2)
    s.add_argument("--with-corrections", action="store_true",
                   help="Also include examples logged via `organizer correct`")
    s.add_argument("--no-color", action="store_true"); s.set_defaults(func=cmd_train)

    s = sub.add_parser("undo"); s.add_argument("--config")
    s.add_argument("--no-color", action="store_true"); s.set_defaults(func=cmd_undo)

    s = sub.add_parser("history"); s.add_argument("--config")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--no-color", action="store_true"); s.set_defaults(func=cmd_history)

    s = sub.add_parser("correct", help="Move a file to the right folder and log it for training")
    s.add_argument("file"); s.add_argument("folder"); s.add_argument("--config")
    s.add_argument("--no-color", action="store_true"); s.set_defaults(func=cmd_correct)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "no_color", False) or not sys.stdout.isatty():
        C.enabled = False
    args.func(args)


if __name__ == "__main__":
    main()
