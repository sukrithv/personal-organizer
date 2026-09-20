import zipfile
from datetime import datetime
from pathlib import Path

from organizer.core import FileContext
from organizer.extractors import extract


def _ctx(p):
    return FileContext.from_path(p)


def test_pdf_extraction(tmp_path):
    from fpdf import FPDF
    pdf = FPDF(); pdf.add_page(); pdf.set_font("helvetica", size=12)
    pdf.multi_cell(0, 10, "INVOICE 4471 Amount due 250 Acme Corporation")
    out = tmp_path / "acme.pdf"; pdf.output(str(out))
    ctx = extract(_ctx(out))
    assert ctx.text_preview and "Amount due" in ctx.text_preview


def test_docx_extraction_stdlib(tmp_path):
    xml = ('<?xml version="1.0"?><w:document xmlns:w="x"><w:body>'
           '<w:p><w:r><w:t>Quarterly revenue report</w:t></w:r></w:p>'
           '</w:body></w:document>')
    out = tmp_path / "report.docx"
    with zipfile.ZipFile(out, "w") as z:
        z.writestr("word/document.xml", xml)
    ctx = extract(_ctx(out))
    assert ctx.text_preview == "Quarterly revenue report"


def test_text_extraction(tmp_path):
    p = tmp_path / "note.md"; p.write_text("# Heading\nbody text")
    ctx = extract(_ctx(p))
    assert ctx.text_preview is not None
    assert "body text" in ctx.text_preview


def test_extract_never_raises_on_junk(tmp_path):
    p = tmp_path / "fake.jpg"; p.write_bytes(b"not a jpeg")
    ctx = extract(_ctx(p))            # must not raise
    assert ctx.text_preview is None


def test_exif_date_flows_into_folder(tmp_path):
    # We don't need a real EXIF file: set metadata['taken'] and confirm the
    # classifier uses it for the year/month folder instead of file mtime.
    from organizer.classifiers import TypeClassifier
    from organizer.config import load_config
    p = tmp_path / "photo.jpg"; p.write_bytes(b"x")
    ctx = _ctx(p)
    ctx.metadata["taken"] = datetime(2019, 8, 15)
    d = TypeClassifier(load_config()).classify(ctx)
    assert d is not None
    assert d.folder == Path("Images/2019/08")
