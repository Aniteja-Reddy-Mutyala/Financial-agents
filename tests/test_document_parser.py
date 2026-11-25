"""Tests for document_parser.parse_document — real pypdf against small
in-test-built fixtures (pypdf is a pure local library, not a network
boundary, so no fake is needed per write-compliance-tests' mocking
convention — only the LLM/embedder/searcher boundary needs faking)."""
import io

import pytest

from document_parser import MAX_EXTRACTED_CHARS, DocumentParseError, parse_document


def _build_pdf_bytes(text):
    """A hand-built, minimal single-page PDF with a real text content
    stream — pypdf has no simple "write text" helper of its own (it's a
    reader/manipulator, not a generator like reportlab), so this writes
    the handful of PDF objects a real .extract_text() call needs directly,
    rather than pulling in a whole new dependency just for a test fixture."""
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 300 300] /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    stream_content = f"BT /F1 12 Tf 10 100 Td ({text}) Tj ET".encode("latin-1")
    objects.append(b"<< /Length " + str(len(stream_content)).encode() + b" >>\nstream\n"
                   + stream_content + b"\nendstream")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{i} 0 obj\n".encode())
        out.write(obj)
        out.write(b"\nendobj\n")
    xref_pos = out.tell()
    n = len(objects) + 1
    out.write(f"xref\n0 {n}\n".encode())
    out.write(b"0000000000 65535 f \n")
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(b"trailer\n")
    out.write(f"<< /Size {n} /Root 1 0 R >>\n".encode())
    out.write(b"startxref\n")
    out.write(str(xref_pos).encode())
    out.write(b"\n%%EOF")
    out.seek(0)
    return out


def test_clear_cut_pdf_extraction():
    source = _build_pdf_bytes("Experienced engineer skilled in Kubernetes")
    result = parse_document(source, filename_hint="resume.pdf")
    assert "Kubernetes" in result


def test_clear_cut_plain_text_extraction():
    source = io.BytesIO(b"Experienced engineer skilled in Python and Kubernetes.")
    result = parse_document(source, filename_hint="resume.txt")
    assert result == "Experienced engineer skilled in Python and Kubernetes."


def test_clear_cut_str_path_txt(tmp_path):
    path = tmp_path / "resume.txt"
    path.write_text("Backend developer with AWS experience.")
    result = parse_document(str(path))
    assert result == "Backend developer with AWS experience."


def test_edge_unsupported_extension_raises():
    source = io.BytesIO(b"whatever")
    with pytest.raises(DocumentParseError, match="Unsupported file type"):
        parse_document(source, filename_hint="resume.docx")


def test_edge_corrupt_pdf_raises():
    source = io.BytesIO(b"not a real pdf")
    with pytest.raises(DocumentParseError):
        parse_document(source, filename_hint="resume.pdf")


def test_edge_empty_text_raises():
    source = io.BytesIO(b"   \n\n   ")
    with pytest.raises(DocumentParseError, match="No extractable text"):
        parse_document(source, filename_hint="resume.txt")


def test_edge_truncates_to_max_extracted_chars(tmp_path):
    path = tmp_path / "long.txt"
    path.write_text("x" * (MAX_EXTRACTED_CHARS + 500))
    result = parse_document(str(path))
    assert len(result) == MAX_EXTRACTED_CHARS


def test_edge_no_filename_hint_and_no_name_raises():
    source = io.BytesIO(b"whatever")
    with pytest.raises(DocumentParseError, match="Can't determine file type"):
        parse_document(source)
