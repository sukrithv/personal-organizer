from pathlib import Path

from organizer.classifiers import LLMClassifier, RuleClassifier, TypeClassifier
from organizer.config import load_config
from organizer.core import FileContext


def ctx(tmp_path, name, when="2024-03-14", content=""):
    import os
    import time
    p = tmp_path / name
    p.write_text(content)
    t = time.mktime(time.strptime(when, "%Y-%m-%d"))
    os.utime(p, (t, t))
    return FileContext.from_path(p)


def test_rule_matches_invoice(tmp_path):
    cfg = load_config()
    d = RuleClassifier(cfg).classify(ctx(tmp_path, "invoice_jan.pdf", "2024-01-05"))
    assert d is not None and d.folder == Path("Finance/2024") and d.confidence == 1.0


def test_rule_returns_none_when_no_match(tmp_path):
    cfg = load_config()
    assert RuleClassifier(cfg).classify(ctx(tmp_path, "random.pdf")) is None


def test_type_classifier_with_date(tmp_path):
    cfg = load_config()
    d = TypeClassifier(cfg).classify(ctx(tmp_path, "vacation.jpg", "2022-11-20"))
    assert d.folder == Path("Images/2022/11") and d.source == "type"


def test_type_classifier_unknown_ext(tmp_path):
    cfg = load_config()
    d = TypeClassifier(cfg).classify(ctx(tmp_path, "weird.xyz"))
    assert str(d.folder).startswith("Other")


def test_type_classifier_no_date(tmp_path):
    cfg = load_config()
    cfg["use_date_subfolders"] = False
    d = TypeClassifier(cfg).classify(ctx(tmp_path, "song.mp3"))
    assert d.folder == Path("Audio")


def test_llm_sanitize():
    s = LLMClassifier._sanitize
    assert s("../../etc/passwd") == Path("etc/passwd")
    assert s("/abs/path") == Path("abs/path")
    assert s("Finance/2024") == Path("Finance/2024")
    assert s("") is None
    assert s("  weird:name?  ") == Path("weirdname")
