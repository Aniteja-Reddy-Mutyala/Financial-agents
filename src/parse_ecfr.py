"""Parse eCFR part/section XML (any title) into flat section records.

Raw files are named `title<N>_<label>.xml` so the CFR title number (which
eCFR bulk XML doesn't repeat on every section) can be recovered from the
filename. Each file's root is either a whole PART (DIV5, containing many
DIV8 SECTIONs) or a hand-merged bag of individually-fetched DIV8 SECTIONs
(see data/raw/title17_sox.xml) — both shapes work the same way below since
`Element.iter("DIV8")` finds every SECTION regardless of nesting depth.
"""
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "kb.json"

_FILENAME_RE = re.compile(r"^title(\d+)_")


def text_of(elem):
    return "".join(elem.itertext())


def clean(s):
    s = re.sub(r"\s+", " ", s or "").strip()
    return s


def parse_part(xml_path):
    match = _FILENAME_RE.match(xml_path.name)
    if not match:
        raise ValueError(
            f"{xml_path.name}: expected filename like 'title<N>_<label>.xml' "
            "so the CFR title number can be recovered"
        )
    cfr_title = match.group(1)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    part_citation = root.attrib.get("N", "")
    records = []
    for sec in root.iter("DIV8"):  # DIV8 = SECTION level in eCFR bulk XML
        if sec.attrib.get("TYPE") != "SECTION":
            continue
        sec_num = sec.attrib.get("N", "")
        head_elem = sec.find("HEAD")
        heading = clean(text_of(head_elem)) if head_elem is not None else ""
        body_parts = []
        for child in sec:
            if child.tag == "HEAD":
                continue
            body_parts.append(clean(text_of(child)))
        body = " ".join(p for p in body_parts if p)
        if not body:
            continue
        records.append(
            {
                "id": f"{cfr_title} CFR {sec_num}",
                "title": heading,
                "part": part_citation,
                "text": body,
            }
        )
    return records


def main():
    all_records = []
    for xml_file in sorted(RAW_DIR.glob("title*.xml")):
        recs = parse_part(xml_file)
        print(f"{xml_file.name}: {len(recs)} sections")
        all_records.extend(recs)
    OUT_PATH.write_text(json.dumps(all_records, indent=2))
    print(f"Wrote {len(all_records)} sections to {OUT_PATH}")


if __name__ == "__main__":
    main()
