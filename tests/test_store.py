from pathlib import Path

from organizer.pipeline import Move
from organizer.store import CorrectionLog, TransactionLog


def _move(tmp_path, name):
    src = tmp_path / name; src.write_text("x")
    dst = tmp_path / "Sorted" / name
    return Move(src, dst, "type \u00b7 test", "type", 0.4)


def test_transaction_record_and_undo(tmp_path):
    db = tmp_path / "t.db"
    tx = TransactionLog(db)
    m = _move(tmp_path, "f.txt")
    m.dst.parent.mkdir(parents=True); import shutil; shutil.move(str(m.src), str(m.dst))
    tx.record(m)
    assert m.dst.exists() and not m.src.exists()
    restored = tx.undo_last(log=lambda *_: None)
    assert restored == 1 and m.src.exists() and not m.dst.exists()


def test_undo_nothing(tmp_path):
    assert TransactionLog(tmp_path / "t.db").undo_last(log=lambda *_: None) == 0


def test_source_stats(tmp_path):
    db = tmp_path / "t.db"; tx = TransactionLog(db)
    for i in range(3):
        m = _move(tmp_path, f"f{i}.txt")
        m.dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil; shutil.move(str(m.src), str(m.dst))
        tx.record(m)
    stats = {r["source"]: r["n"] for r in tx.source_stats()}
    assert stats["type"] == 3


def test_correction_log_export(tmp_path):
    db = tmp_path / "t.db"
    cl = CorrectionLog(db)
    cl.record("resume cv work experience", "Documents/Career", was="Downloads")
    cl.record("invoice amount due", "Finance/Invoices", was="Misc")
    pairs = list(cl.export())
    assert ("invoice amount due", "Finance/Invoices") in pairs
    assert cl.count() == 2
