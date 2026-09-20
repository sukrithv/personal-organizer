"""
classifiers.py — the decision-makers, each implementing core.Classifier.

Phase 1 chain (built in pipeline.build_chain):
    RuleClassifier   deterministic user rules            (conf 1.0)
    TrainedClassifier  inert until Phase 3 (returns None)
    LLMClassifier    Ollama, keep_alive=0 so it unloads  (conf ~0.65)
    TypeClassifier   type + date fallback, always answers (conf 0.4)
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from .config import build_ext_index
from .core import Classifier, Decision, FileContext


def _template_vars(ctx: FileContext, cfg, category=None):
    # Prefer a real capture date from metadata (e.g. EXIF "taken") over file mtime.
    dt = ctx.metadata.get("taken") or ctx.date(cfg.get("date_source", "modified"))
    return {
        "year": f"{dt.year:04d}", "month": f"{dt.month:02d}", "day": f"{dt.day:02d}",
        "ext": ctx.ext or "none", "stem": ctx.path.stem, "category": category or "",
    }


def _category_for(ext, ext_index, cfg):
    return ext_index.get(ext, cfg.get("unmatched_folder", "Other"))


# --------------------------------------------------------------------------- #

class RuleClassifier(Classifier):
    name = "rule"

    def __init__(self, cfg):
        self.cfg = cfg
        self.ext_index = build_ext_index(cfg)

    def _matches(self, ctx, rule):
        name = ctx.name.lower()
        ext = ctx.ext
        if rule.get("ext") and ext not in [e.lower().lstrip(".") for e in rule["ext"]]:
            return False
        if rule.get("any") and not any(s.lower() in name for s in rule["any"]):
            return False
        if rule.get("all") and not all(s.lower() in name for s in rule["all"]):
            return False
        if rule.get("regex") and not re.search(rule["regex"], ctx.name):
            return False
        if rule.get("min_size_mb") is not None and ctx.size < rule["min_size_mb"] * 1024 * 1024:
            return False
        if rule.get("max_size_mb") is not None and ctx.size > rule["max_size_mb"] * 1024 * 1024:
            return False
        return True

    def classify(self, ctx: FileContext) -> Decision | None:
        for rule in self.cfg.get("rules", []):
            if self._matches(ctx, rule):
                cat = _category_for(ctx.ext, self.ext_index, self.cfg)
                folder = rule["dest"].format(**_template_vars(ctx, self.cfg, cat))
                return Decision(Path(folder), 1.0, f"rule \u00b7 {rule['name']}", "rule")
        return None


class TypeClassifier(Classifier):
    name = "type"

    def __init__(self, cfg):
        self.cfg = cfg
        self.ext_index = build_ext_index(cfg)

    def classify(self, ctx: FileContext) -> Decision | None:
        category = _category_for(ctx.ext, self.ext_index, self.cfg)
        parts = [category]
        reason = f"type \u00b7 {category}"
        if self.cfg.get("use_date_subfolders"):
            vars_ = _template_vars(ctx, self.cfg, category)
            date_part = self.cfg["date_format"].format(**vars_)
            parts.append(date_part)
            reason += f" \u00b7 {date_part}"
        return Decision(Path(*parts), 0.4, reason, "type")


class TrainedClassifier(Classifier):
    """Learns from your already-organized folders. Can ONLY output folders that
    exist in the training data, so it can't invent new ones. Returns None when
    it isn't confident enough, deferring to the LLM / type fallback."""
    name = "trained"

    def __init__(self, model_path, threshold: float = 0.55):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self._model = None

    def _ensure(self):
        if self._model is None:
            from .training import load_model
            self._model = load_model(self.model_path)
        return self._model

    def classify(self, ctx: FileContext) -> Decision | None:
        from .features import featurize
        try:
            pipe = self._ensure()["pipeline"]
        except (OSError, KeyError):
            return None
        probs = pipe.predict_proba([featurize(ctx)])[0]
        i = int(probs.argmax())
        conf = float(probs[i])
        label = str(pipe.classes_[i])
        if conf < self.threshold:
            return None
        return Decision(Path(label), conf, f"learned · {label} ({conf:.0%})", "trained")


class LLMClassifier(Classifier):
    name = "llm"

    SCHEMA = {
        "type": "object",
        "properties": {"folder": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["folder", "reason"],
    }

    def __init__(self, cfg):
        o = cfg["ollama"]
        self.host = o["host"].rstrip("/")
        self.model = o["model"]
        self.keep_alive = o.get("keep_alive", "0")
        self.timeout = o.get("timeout", 60)
        self.categories = list(cfg["categories"].keys())

    def classify(self, ctx: FileContext) -> Decision | None:
        system = (
            "You sort files into folders. Given a file's metadata (and maybe a "
            "content preview), reply with the best destination folder as a short "
            "relative path using forward slashes, plus a brief reason.\n"
            f"Prefer these top-level categories when they fit: {', '.join(self.categories)}.\n"
            "You may add a meaningful subfolder. Never use absolute paths, '..', "
            "or a drive letter. Keep it at most 3 levels deep."
        )
        preview = ctx.text_preview or "(binary or not previewed)"
        user = (
            f"Filename: {ctx.name}\nExtension: {ctx.ext or '(none)'}\n"
            f"Size: {max(1, ctx.size // 1024)} KB\n"
            f"Modified: {ctx.modified:%Y-%m-%d}\n\nContent preview:\n{preview}\n\n"
            'Return JSON: {"folder": "...", "reason": "..."}'
        )
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,
            "format": self.SCHEMA,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0},
        }
        req = urllib.request.Request(
            self.host + "/api/chat", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                resp = json.load(r)
            parsed = json.loads(resp["message"]["content"])
        except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError, OSError):
            return None
        folder = self._sanitize(parsed.get("folder", ""))
        if folder is None:
            return None
        reason = (parsed.get("reason") or "").strip().replace("\n", " ")[:120]
        return Decision(folder, 0.65, "ai \u00b7 " + reason if reason else "ai", "llm")

    @staticmethod
    def _sanitize(folder) -> Path | None:
        if not folder:
            return None
        parts = []
        for raw in str(folder).replace("\\", "/").split("/"):
            p = re.sub(r'[<>:"|?*\x00-\x1f]', "", raw).strip(" .")
            if p and p not in (".", ".."):
                parts.append(p)
        return Path(*parts[:6]) if parts else None


def ollama_available(host: str) -> bool:
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=4) as r:
            return r.status == 200
    except Exception:
        return False
