# ai-file-organizer

An extensible, AI-assisted file organizer. Each file runs through a **chain of
classifiers** — duplicate check, your rules, an optional local model, then type
+ date sorting. It organizes on demand or **watches a folder** and files new
downloads as they land. Every move is recorded in SQLite, so `undo` and
`history` work across sessions.

See `roadmap.md` for what's next (confidence gating, a daemon/UI, and
plugin support).

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # core + pytest/ruff
pip install -e ".[pdf,images,audio]"    # content extractors (recommended)
```

Optional local model for smarter, content-aware decisions:

```bash
ollama pull llama3.2:1b     # or gemma3:4b for better accuracy
```

## Use

```bash
organizer plan  ~/Downloads --ai          # preview; moves nothing
organizer run   ~/Downloads --ai          # plan, then apply on approval
organizer watch ~/Downloads               # auto-file new downloads
organizer run   ~/Downloads --ai --dedupe # also route exact duplicates aside
organizer undo                            # reverse the last run (reads SQLite)
organizer history                         # what got moved, by which classifier
organizer correct <file> <folder>         # fix a placement + log it for training
organizer init                            # write organizer.config.json
```

## How a file is decided

The chain, cheap to expensive (`organizer/pipeline.py:build_chain`):

1. **DuplicateClassifier** — with `--dedupe`, routes an exact byte-duplicate to
   `_Duplicates/` (moved together, never deleted). Add `--near-duplicates` to
   also catch re-saved/re-exported near-duplicates by cosine similarity of
   their leading content.
2. **RuleClassifier** — your deterministic rules (confidence 1.0).
3. **TrainedClassifier** — learns your folders. Skipped unless a model file
   exists.
4. **LLMClassifier** — Ollama, only if enabled and reachable. Sees extracted
   **content** (PDF/DOCX text, audio tags), not just the filename.
5. **TypeClassifier** — type + date fallback that always answers. Uses a photo's
   EXIF "date taken" when available, not the download date.

`run_chain` returns the first confident answer, so ordering sets precedence.

## Content extraction

`organizer/extractors.py` fills a content preview and metadata per file type:
PDF text (pdfminer.six), DOCX text (stdlib), image EXIF date/camera (Pillow),
audio tags (mutagen). A missing library or unreadable file just means less
signal — extraction never breaks a run. Scanned PDFs (no text layer) are flagged
for future OCR.

## State

SQLite at `~/.file-organizer/organizer.db`:

- **actions** — every move (source classifier, confidence, reason); powers
  `undo` and `history`.
- **corrections** — labeled examples from `correct`; the training set the
  trained classifier consumes.

## Develop

```bash
pytest
ruff check .
```

## Layout

```
organizer/
  core.py         FileContext, Decision, Classifier, run_chain, Policy
  config.py       defaults + config loading
  extractors.py   PDF / DOCX / EXIF / audio content extraction
  features.py     featurize() shared by corrections + the trainer
  dedupe.py       hashing/cosine-similarity duplicate detection + DuplicateClassifier
  classifiers.py  Duplicate / Rule / Type / LLM / Trained classifiers
  pipeline.py     build_chain, build_plan, classify_one
  store.py        SQLite TransactionLog + CorrectionLog
  cli.py          init / plan / run / watch / undo / history / correct
tests/
```
