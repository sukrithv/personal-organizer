# Roadmap

The plan for growing this from a working organizer into a tool that learns your
structure. **Phases 1–3 are done** (this package); Phase 4 is ahead.

Guiding principle: don't build ahead of pain. Run it on your real Downloads,
collect annoyances, and let that list drive what you build next.

## The architecture (the part to protect)

Every decision-maker implements one method — `classify(ctx) -> Decision | None`
— and `run_chain` tries them cheap-to-expensive, taking the first confident
answer. New capability = new classifier, inserted in the chain. Nothing else
changes.

```
file -> extractors (enrich) -> [Rule -> Trained -> LLM -> Type] -> Decision -> move + journal
```

## Phase 1 — package + the seam  ✅ DONE

- Installable package, `organizer` console command, JSON-journal undo.
- Rules / LLM / type-sorting all behind the `Classifier` interface.
- `pytest` suite (core, config, classifiers, pipeline, mocked LLM).

## Phase 2 — see more, decide better  ✅ DONE

- **Extractors**: implement the PDF / EXIF / audio stubs in `extractors.py`
  (they're registered no-ops now). PDF text and photo "date taken" are the big
  wins. Install via the `pdf` / `images` / `audio` extras.
- **Duplicate detection**: hash files, skip or link exact dupes.
- **SQLite** `TransactionLog` (stubbed in `store.py`) replacing the JSON journal
  — buys queryable history and multi-session undo.

## Phase 3 — make it learn you  ✅ DONE

- **Trained classifier** (`TrainedClassifier` is a live stub that defers today):
  bootstrap labels from folders you've *already* organized (the folder is the
  label), fit TF-IDF(char 3–5 grams) + LogisticRegression, save a few-MB model.
  Install via the `ml` extra. Wire `cfg["model_path"]` and it auto-joins the
  chain ahead of the LLM.
- **Confidence gating**: `Policy` (already in `core.py`) routes low-confidence
  files to a review/quarantine folder instead of a wrong guess.
- **Learn from corrections**: log every undo / manual move as a labeled example
  (`CorrectionLog`, stubbed) and retrain periodically — cheap, seconds.

## Phase 4 — product surface

- Daemon (systemd / launchd), multi-folder, config hot-reload.
- Event-driven watcher (watchdog/inotify/FSEvents) behind the same interface;
  keep polling as the zero-dep fallback.
- Review UI (Textual TUI or tiny local web page) feeding the CorrectionLog.
- Plugins via entry points: third-party classifiers / extractors / actions.

## The one habit to start now

Log corrections from day one — even before training anything. It's the raw
material the trained model, retrieval, and any eventual fine-tuning all need,
and it's expensive to recreate later.
