# Roadmap

This document tracks the project's progression from a rule-based file mover
into a tool that learns a user's organizational structure, along with what
remains ahead.

Design principle: extend the classifier chain rather than special-casing new
behavior. Each capability is a new classifier inserted into the chain; nothing
else needs to change.

## Architecture

Every decision-maker implements one method — `classify(ctx) -> Decision | None`
— and `run_chain` tries them cheap-to-expensive, taking the first confident
answer.

```
file -> extractors (enrich) -> [Duplicate -> Rule -> Trained -> LLM -> Type] -> Decision -> move + journal
```

## Completed

**Package and classifier interface**
- Installable package, `organizer` console command, JSON-journal undo.
- Rules / LLM / type-sorting unified behind the `Classifier` interface.
- `pytest` suite covering core, config, classifiers, pipeline, and a mocked LLM.

**Content extraction and duplicate detection**
- Content extractors for PDF, DOCX, image EXIF, and audio tags
  (`extractors.py`), installed via the `pdf` / `images` / `audio` extras.
- Exact-duplicate detection via file hashing, and near-duplicate detection via
  cosine similarity over each document's leading tokens (`dedupe.py`).
- SQLite-backed `TransactionLog` replacing the original JSON journal, giving
  queryable history and multi-session undo.

**Trained classifier**
- `TrainedClassifier` bootstraps labels from already-organized folders (the
  folder name is the label), fits TF-IDF (char 3–5 grams) + LogisticRegression,
  and saves a compact model. Installed via the `ml` extra; once
  `cfg["model_path"]` points at a trained model it joins the chain ahead of
  the LLM classifier.
- `CorrectionLog` records manual corrections as labeled examples for
  retraining.

## Planned

**Confidence gating**
- Route low-confidence decisions to a review/quarantine folder instead of
  applying an uncertain guess. `Policy` (`core.py`) defines the confidence
  thresholds; wiring it into the move pipeline is the remaining step.

**Product surface**
- Daemon (systemd / launchd) with multi-folder support and config hot-reload.
- Event-driven watcher (watchdog/inotify/FSEvents) behind the existing
  interface, with polling retained as the zero-dependency fallback.
- Review UI (a TUI or a small local web page) backed by `CorrectionLog`.
- Plugin support via entry points for third-party classifiers, extractors,
  and actions.

## Note on corrections

Corrections logged via `organizer correct` are the training data the trained
classifier, and any future retrieval or fine-tuning work, depends on. Logging
them consistently from early use avoids having to reconstruct that data later.
