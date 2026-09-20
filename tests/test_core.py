from pathlib import Path

from organizer.core import Classifier, Decision, FileContext, Outcome, Policy, run_chain


class Fixed(Classifier):
    def __init__(self, decision):
        self._d = decision

    def classify(self, ctx):
        return self._d


def _ctx(tmp_path):
    (tmp_path / "f.txt").write_text("x")
    return FileContext.from_path(tmp_path / "f.txt")


def test_from_path(tmp_path):
    ctx = _ctx(tmp_path)
    assert ctx.name == "f.txt" and ctx.ext == "txt" and ctx.size == 1


def test_run_chain_first_confident_wins(tmp_path):
    ctx = _ctx(tmp_path)
    a = Fixed(Decision(Path("A"), 0.9, "a", "x"))
    b = Fixed(Decision(Path("B"), 1.0, "b", "y"))
    assert run_chain(ctx, [a, b], min_confidence=0.5).folder == Path("A")


def test_run_chain_skips_none(tmp_path):
    ctx = _ctx(tmp_path)
    none = Fixed(None)
    b = Fixed(Decision(Path("B"), 1.0, "b", "y"))
    assert run_chain(ctx, [none, b]).source == "y"


def test_run_chain_returns_best_below_bar(tmp_path):
    ctx = _ctx(tmp_path)
    lo = Fixed(Decision(Path("L"), 0.3, "l", "x"))
    hi = Fixed(Decision(Path("H"), 0.4, "h", "y"))
    assert run_chain(ctx, [lo, hi], min_confidence=0.9).folder == Path("H")


def test_run_chain_empty(tmp_path):
    assert run_chain(_ctx(tmp_path), [Fixed(None)]) is None


def test_policy():
    p = Policy(auto_above=0.8, review_above=0.5)
    assert p.decide(Decision(Path("x"), 0.9, "", "")) is Outcome.AUTO
    assert p.decide(Decision(Path("x"), 0.6, "", "")) is Outcome.REVIEW
    assert p.decide(Decision(Path("x"), 0.2, "", "")) is Outcome.QUARANTINE
    assert p.decide(None) is Outcome.QUARANTINE
