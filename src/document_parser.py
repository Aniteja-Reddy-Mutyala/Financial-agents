"""Document text extraction for the Research agent (src/research_agent.py).

Supports .pdf (via pypdf) and .txt/.md (plain read). A single
DocumentParseError covers every failure mode (unsupported extension,
corrupt/unreadable PDF, or a scanned/image-only PDF with no extractable
text — pypdf doesn't OCR) — one exception type per module, same as
BedrockUnavailable/GroqUnavailable/WebSearchUnavailable, differentiated
by message text rather than subclasses.

`source` can be a path/str OR a file-like object (e.g. Streamlit's
UploadedFile) so the Streamlit page never needs to write a temp file for
an uploaded resume — `filename_hint` supplies the extension when `source`
has no `.suffix` of its own (a file-like object does, via `.name`, which
UploadedFile provides; pass filename_hint explicitly for anything else).
"""
from pathlib import Path

MAX_EXTRACTED_CHARS = 20_000  # bounds prompt size/cost, same spirit as retriever.py's top_k/MIN_SCORE


class DocumentParseError(Exception):
    pass


def parse_document(source, filename_hint=None) -> str:
    """Extract and return text from `source`, truncated to MAX_EXTRACTED_CHARS."""
    suffix = _resolve_suffix(source, filename_hint)
    if suffix == ".pdf":
        text = _extract_pdf_text(source)
    elif suffix in (".txt", ".md"):
        text = _extract_plain_text(source)
    else:
        raise DocumentParseError(f"Unsupported file type {suffix!r} — expected .pdf or .txt")

    text = text.strip()
    if not text:
        raise DocumentParseError(
            "No extractable text found — likely a scanned/image-only PDF (no OCR support here)"
        )
    return text[:MAX_EXTRACTED_CHARS]


def _resolve_suffix(source, filename_hint):
    if filename_hint:
        return Path(filename_hint).suffix.lower()
    name = getattr(source, "name", None)
    if name:
        return Path(name).suffix.lower()
    if isinstance(source, (str, Path)):
        return Path(source).suffix.lower()
    raise DocumentParseError(
        "Can't determine file type — pass filename_hint for a file-like source with no .name"
    )


def _extract_plain_text(source):
    if isinstance(source, (str, Path)):
        return Path(source).read_text(encoding="utf-8", errors="replace")
    raw = source.read()
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return raw


def _extract_pdf_text(source):
    try:
        from pypdf import PdfReader  # lazy import, same pattern as BedrockEmbedder's lazy client

        reader = PdfReader(source)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except DocumentParseError:
        raise
    except Exception as exc:
        raise DocumentParseError(str(exc)) from exc


if __name__ == "__main__":
    import io

    sample = io.BytesIO(b"Not a real PDF, just checking the error path")
    try:
        parse_document(sample, filename_hint="fake.pdf")
    except DocumentParseError as exc:
        print("Expected failure on bogus PDF bytes:", exc)

    text_sample = io.BytesIO(b"Experienced engineer skilled in Python and Kubernetes.")
    print("Plain text extraction:", parse_document(text_sample, filename_hint="resume.txt"))
