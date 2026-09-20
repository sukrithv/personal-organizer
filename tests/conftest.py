import time

import pytest


@pytest.fixture
def messy(tmp_path):
    """A messy folder with predictable files and fixed mtimes."""
    def touch(name, when=None, content=""):
        p = tmp_path / name
        p.write_text(content)
        if when:
            t = time.mktime(time.strptime(when, "%Y-%m-%d"))
            import os
            os.utime(p, (t, t))
        return p
    touch("vacation.jpg", "2022-11-20")
    touch("invoice_jan.pdf", "2024-01-05")
    touch("report.pdf", "2024-03-14")
    touch("song.mp3", "2023-07-01")
    touch("notes.txt", "2024-05-05", content="just some notes")
    touch("weird.xyz", "2024-05-05")
    return tmp_path
