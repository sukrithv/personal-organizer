from pathlib import Path

from organizer.core import FileContext
from organizer.dedupe import DuplicateClassifier, DuplicateIndex, file_sha256


def test_hash_stable_and_distinct(tmp_path):
    a = tmp_path / "a"; a.write_text("hello")
    b = tmp_path / "b"; b.write_text("hello")
    c = tmp_path / "c"; c.write_text("different")
    assert file_sha256(a) == file_sha256(b)
    assert file_sha256(a) != file_sha256(c)


def test_index_flags_second_copy(tmp_path):
    a = tmp_path / "a.txt"; a.write_text("same")
    b = tmp_path / "b.txt"; b.write_text("same")
    idx = DuplicateIndex()
    assert idx.check_and_add(a) is None          # first is original
    assert idx.check_and_add(b) == a             # second is a dup of a


def test_duplicate_classifier_routes(tmp_path):
    a = tmp_path / "a.txt"; a.write_text("same")
    b = tmp_path / "b.txt"; b.write_text("same")
    idx = DuplicateIndex()
    clf = DuplicateClassifier(idx, folder="_Duplicates")
    assert clf.classify(FileContext.from_path(a)) is None
    d = clf.classify(FileContext.from_path(b))
    assert d is not None and d.folder == Path("_Duplicates") and d.source == "dupe"


def test_scan_seeds_index(tmp_path):
    (tmp_path / "existing.txt").write_text("payload")
    idx = DuplicateIndex(); idx.scan(tmp_path)
    newf = tmp_path / "download.txt"; newf.write_text("payload")
    assert idx.check_and_add(newf) is not None    # caught against pre-existing file
