"""Bootstrap from organized folders, train, and classify.

Skips cleanly if scikit-learn isn't installed (it's in the `ml` extra)."""
import random
from pathlib import Path

import pytest

sklearn = pytest.importorskip("sklearn")

from organizer.classifiers import TrainedClassifier
from organizer.core import FileContext
from organizer.extractors import extract
from organizer.pipeline import build_chain
from organizer.training import bootstrap_dataset, load_model, train


FIN = "invoice receipt payment amount due balance billing statement total charge".split()
ACA = "syllabus lecture homework exam grade course professor semester assignment".split()
CODE = "import def class function return numpy pandas model training dataset loop".split()


def _doc(words, rng):
    return " ".join(rng.choice(words) for _ in range(20))


@pytest.fixture
def organized(tmp_path):
    rng = random.Random(0)
    root = tmp_path / "Archive"
    for i in range(12):
        (root / "Finance/Bills").mkdir(parents=True, exist_ok=True)
        (root / f"Finance/Bills/f{i}.txt").write_text(_doc(FIN, rng))
        (root / "School/Courses").mkdir(parents=True, exist_ok=True)
        (root / f"School/Courses/a{i}.txt").write_text(_doc(ACA, rng))
        (root / "Code/Projects").mkdir(parents=True, exist_ok=True)
        (root / f"Code/Projects/c{i}.txt").write_text(_doc(CODE, rng))
    return root


def test_bootstrap_labels_are_folders(organized):
    data = list(bootstrap_dataset(organized, label_depth=2))
    labels = {lab for _, lab in data}
    assert labels == {"Finance/Bills", "School/Courses", "Code/Projects"}
    assert len(data) == 36


def test_bootstrap_skips_loose_root_files(organized):
    (organized / "loose.txt").write_text("no folder label")
    assert all(lab for _, lab in bootstrap_dataset(organized, 2))
    assert len(list(bootstrap_dataset(organized, 2))) == 36  # loose file excluded


def test_train_and_predict(organized, tmp_path):
    rng = random.Random(1)
    model = tmp_path / "m.pkl"
    stats = train(list(bootstrap_dataset(organized, 2)), model, log=lambda *_: None)
    assert stats["classes"] == 3 and stats["examples"] == 36
    clf = TrainedClassifier(model, threshold=0.5)

    def ctx(name, text):
        p = tmp_path / name
        p.write_text(text)
        return extract(FileContext.from_path(p))

    for words, expect in [(FIN, "Finance/Bills"), (ACA, "School/Courses"),
                          (CODE, "Code/Projects")]:
        d = clf.classify(ctx(f"{expect[:3]}.txt", _doc(words, rng)))
        assert d is not None and str(d.folder) == expect


def test_can_only_output_existing_folders(organized, tmp_path):
    model = tmp_path / "m.pkl"
    train(list(bootstrap_dataset(organized, 2)), model, log=lambda *_: None)
    labels = {str(c) for c in load_model(model)["pipeline"].classes_}
    assert labels == {"Finance/Bills", "School/Courses", "Code/Projects"}


def test_defers_when_below_threshold(organized, tmp_path):
    model = tmp_path / "m.pkl"
    train(list(bootstrap_dataset(organized, 2)), model, log=lambda *_: None)
    clf = TrainedClassifier(model, threshold=0.999)  # nothing will clear this
    p = tmp_path / "q.txt"; p.write_text("invoice payment due")
    assert clf.classify(extract(FileContext.from_path(p))) is None


def test_too_little_data_raises(tmp_path):
    with pytest.raises(ValueError):
        train([("only one example", "SoloFolder")], tmp_path / "m.pkl",
              log=lambda *_: None)


def test_trained_joins_chain_when_model_exists(organized, tmp_path):
    from organizer.config import load_config
    model = tmp_path / "m.pkl"
    train(list(bootstrap_dataset(organized, 2)), model, log=lambda *_: None)
    cfg = load_config()
    cfg["model_path"] = str(model)
    names = [c.name for c in build_chain(cfg, use_llm=False, llm_ok=False)]
    assert "trained" in names
    assert names.index("trained") < names.index("type")  # ahead of the fallback
