from pathlib import Path

import pytest

from organizer.core import FileContext
from organizer.dedupe import (
    DuplicateClassifier,
    DuplicateIndex,
    NearDuplicateIndex,
    cosine_similarity,
    file_sha256,
    leading_term_vector,
    tokenize,
)


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


# --------------------------------------------------------------------------- #
# near-duplicate detection (cosine similarity over the first N tokens)
# --------------------------------------------------------------------------- #

def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Invoice #4021, Acme Corp.") == ["invoice", "4021", "acme", "corp"]


def test_leading_term_vector_only_uses_first_n_tokens():
    text = "a a a b b b"
    assert leading_term_vector(text, n_tokens=3) == {"a": 3}


def test_cosine_similarity_identical_and_orthogonal():
    v1 = leading_term_vector("the quick brown fox jumps")
    v2 = leading_term_vector("the quick brown fox jumps")
    v3 = leading_term_vector("totally unrelated content here now")
    assert cosine_similarity(v1, v2) == pytest.approx(1.0)
    assert cosine_similarity(v1, v3) < 0.3


def _padded(sentence: str, repeat: int = 5) -> str:
    """Repeat sentence so its tokens dominate and clear min_tokens, without
    diluting similarity with shared filler words across unrelated texts."""
    return " ".join([sentence] * repeat)


def test_near_duplicate_index_flags_similar_leading_text():
    idx = NearDuplicateIndex(threshold=0.9, min_tokens=5)
    original = _padded("Quarterly report for the finance team covering Q3 results")
    edited = _padded("Quarterly report for the finance team covering Q3 results revised")
    unrelated = _padded("Grandmas recipe for banana bread with walnuts and cinnamon")

    assert idx.check_and_add(Path("a.txt"), original) is None       # first: recorded
    hit = idx.check_and_add(Path("b.txt"), edited)                  # near-dup of a
    assert hit is not None
    assert hit[0] == Path("a.txt") and hit[1] >= 0.9
    assert idx.check_and_add(Path("c.txt"), unrelated) is None       # not similar enough


def test_near_duplicate_index_ignores_short_previews():
    idx = NearDuplicateIndex(threshold=0.5, min_tokens=20)
    assert idx.check_and_add(Path("a.txt"), "too short") is None
    assert idx.check_and_add(Path("b.txt"), "too short") is None    # never flagged


def test_near_duplicate_index_ignores_empty_preview():
    idx = NearDuplicateIndex()
    assert idx.check_and_add(Path("a.txt"), None) is None
    assert idx.check_and_add(Path("a.txt"), "") is None


def test_duplicate_classifier_uses_near_index_when_no_exact_match(tmp_path):
    a = tmp_path / "a.txt"; a.write_text("original bytes")
    b = tmp_path / "b.txt"; b.write_text("different bytes, same wording")
    idx = DuplicateIndex()
    near = NearDuplicateIndex(threshold=0.9, min_tokens=5)
    clf = DuplicateClassifier(idx, folder="_Duplicates", near_index=near)

    ctx_a = FileContext.from_path(a)
    ctx_a.text_preview = _padded("Annual budget summary for the org")
    assert clf.classify(ctx_a) is None                              # first: recorded

    ctx_b = FileContext.from_path(b)
    ctx_b.text_preview = _padded("Annual budget summary for the org.")
    d = clf.classify(ctx_b)
    assert d is not None
    assert d.folder == Path("_Duplicates") and d.source == "dupe"
    assert "near-duplicate" in d.reason


def test_scan_seeds_near_duplicate_index(tmp_path):
    (tmp_path / "existing.txt").write_text(_padded("Employee handbook policy section one"))
    near = NearDuplicateIndex(threshold=0.9, min_tokens=5)
    near.scan(tmp_path)

    newf = tmp_path / "handbook_copy.txt"
    newf.write_text(_padded("Employee handbook policy section one!"))
    hit = near.check_and_add(newf, _padded("Employee handbook policy section one!"))
    assert hit is not None
