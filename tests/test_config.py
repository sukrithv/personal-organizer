import json

from organizer.config import build_ext_index, load_config


def test_defaults_load():
    cfg = load_config()
    assert cfg["ollama"]["keep_alive"] == "0"
    assert "Images" in cfg["categories"]


def test_partial_override_merges(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"ollama": {"model": "gemma3:1b"}, "unmatched_folder": "Misc"}))
    cfg = load_config(p)
    assert cfg["ollama"]["model"] == "gemma3:1b"
    assert cfg["ollama"]["keep_alive"] == "0"      # untouched default preserved
    assert cfg["unmatched_folder"] == "Misc"


def test_ext_index():
    idx = build_ext_index(load_config())
    assert idx["jpg"] == "Images" and idx["pdf"] == "Documents"
