"""Test the LLM path with a stubbed HTTP layer (no Ollama needed)."""
import json
from pathlib import Path

import organizer.classifiers as clf
from organizer.config import load_config
from organizer.core import FileContext


class FakeResp:
    def __init__(self, obj):
        self._d = json.dumps(obj).encode()
    def read(self):
        return self._d
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def make_ctx(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    ctx = FileContext.from_path(p)
    ctx.text_preview = content
    return ctx


def test_llm_uses_content(tmp_path, monkeypatch):
    def fake_urlopen(req, timeout=None):
        body = json.loads(req.data.decode())
        assert body["format"]                     # structured schema sent
        assert body["keep_alive"] == "0"          # unloads when idle
        user = [m["content"] for m in body["messages"] if m["role"] == "user"][0]
        folder = "Finance/Invoices" if "amount due" in user.lower() else "Misc"
        return FakeResp({"message": {"content": json.dumps(
            {"folder": folder, "reason": "test"})}})
    monkeypatch.setattr(clf.urllib.request, "urlopen", fake_urlopen)

    cfg = load_config()
    ctx = make_ctx(tmp_path, "doc.txt", "Invoice\nAmount due: $50")
    d = clf.LLMClassifier(cfg).classify(ctx)
    assert d.folder == Path("Finance/Invoices") and d.source == "llm"


def test_llm_returns_none_on_error(tmp_path, monkeypatch):
    def boom(req, timeout=None):
        raise OSError("connection refused")
    monkeypatch.setattr(clf.urllib.request, "urlopen", boom)
    cfg = load_config()
    ctx = make_ctx(tmp_path, "doc.txt", "hi")
    assert clf.LLMClassifier(cfg).classify(ctx) is None
