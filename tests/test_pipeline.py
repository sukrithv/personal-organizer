from pathlib import Path

from organizer.config import load_config
from organizer.pipeline import build_chain, build_plan, iter_files


def chain(cfg):
    return build_chain(cfg, use_llm=False, llm_ok=False)


def test_iter_files_top_level_only(messy):
    (messy / "sub").mkdir()
    (messy / "sub" / "deep.txt").write_text("x")
    names = {p.name for p in iter_files(messy, recursive=False, include_hidden=False)}
    assert "deep.txt" not in names and "notes.txt" in names


def test_build_plan_routes_files(messy):
    cfg = load_config()
    moves, dest_root = build_plan(messy, cfg, chain(cfg))
    by_name = {m.src.name: m.dst.relative_to(dest_root) for m in moves}
    assert by_name["invoice_jan.pdf"] == Path("Finance/2024/invoice_jan.pdf")
    assert by_name["vacation.jpg"] == Path("Images/2022/11/vacation.jpg")
    assert by_name["song.mp3"] == Path("Audio/2023/07/song.mp3")
    assert str(by_name["weird.xyz"]).startswith("Other")


def test_build_plan_skips_partial_downloads(messy):
    (messy / "big.zip.crdownload").write_text("partial")
    cfg = load_config()
    moves, _ = build_plan(messy, cfg, chain(cfg))
    assert all(".crdownload" not in m.src.name for m in moves)


def test_collision_suffix(tmp_path):
    import os
    import time
    a = tmp_path / "a"; b = tmp_path / "b"; a.mkdir(); b.mkdir()
    for d in (a, b):
        p = d / "report.pdf"; p.write_text("x")
        t = time.mktime(time.strptime("2024-01-01", "%Y-%m-%d"))
        os.utime(p, (t, t))
    cfg = load_config()
    moves, dest_root = build_plan(tmp_path, cfg, chain(cfg), recursive=True)
    targets = sorted(str(m.dst.name) for m in moves)
    assert targets == ["report (1).pdf", "report.pdf"]
