---
name: add-ecfr-regulation
description: Use when asked to add, update, or expand the regulatory knowledge base (data/kb.json) with more CFR text — a new part, chapter, title, or specific section beyond the existing BSA/AML (31 CFR 1010/1020), FDA (21 CFR 11), HIPAA (45 CFR 160/164), and SOX (17 CFR, selected sections) domains. Covers fetching from the eCFR API and rebuilding the parsed knowledge base.
---

## Where the data comes from

`data/kb.json` is parsed from real eCFR XML in `data/raw/`, across four
domains: BSA/AML (31 CFR Chapter X, Parts 1010 & 1020), FDA electronic
records (21 CFR Part 11), HIPAA (45 CFR Parts 160 & 164), and
Sarbanes-Oxley implementing SEC rules (17 CFR, five hand-picked sections —
not a whole part, since SOX itself lives mostly in the U.S. Code and the
CFR parts these sections sit in are mostly unrelated to SOX). Don't
hand-write or fabricate regulatory text — always pull real text from the
[eCFR API](https://www.ecfr.gov/developers/documentation/api/v1). **GDPR
and SOC2 are deliberately not in this knowledge base** — GDPR isn't on
eCFR at all (EU law), and SOC2 has no free authoritative public text
(proprietary AICPA framework). Don't add either without first telling the
user there's no real-text source for them here.

## Filename convention — required for the parser

`src/parse_ecfr.py` recovers the CFR title number from the filename, since
eCFR bulk XML doesn't repeat the title number on every section. Every raw
file **must** be named `title<N>_<label>.xml`, e.g. `title21_part11.xml`,
`title45_part164.xml`, `title17_sox.xml`. A file that doesn't match this
pattern makes `parse_part()` raise, by design — don't rename around it.

## Finding the right part/title/section

Browse structure first to find the citation you need, e.g.:

```bash
curl -s --max-time 10 "https://www.ecfr.gov/api/versioner/v1/structure/<date>/title-<N>.json" \
  | python3 -m json.tool | less
```

Use a recent date (today or close to it) in `<date>` (format `YYYY-MM-DD`).
If the date you pick is after the title's most recent issue date, the API
returns a JSON error naming the correct date to use instead — read it and
retry, don't guess an earlier date blindly.

## Fetching the XML

The full-text endpoint **requires compression** — a plain `curl` without
`--compressed` returns an error body instead of XML. For a whole part:

```bash
curl -s --compressed --max-time 30 \
  "https://www.ecfr.gov/api/versioner/v1/full/<date>/title-<N>.xml?part=<PART>" \
  -o data/raw/title<N>_part<PART>.xml
```

For a single section within a part (useful when the part is huge and only
a few sections are relevant — this is how the SOX sections were pulled
out of Parts 229/240 instead of fetching those whole parts):

```bash
curl -s --compressed --max-time 20 \
  "https://www.ecfr.gov/api/versioner/v1/full/<date>/title-<N>.xml?part=<PART>&section=<SECTION>" \
  -o /tmp/<section>.xml
```

Each single-section fetch returns one bare `DIV8` element (no wrapping
part). To combine several hand-picked sections into one file the parser
can read, strip each one's `<?xml ...?>` declaration line and concatenate
the `DIV8` bodies inside a synthetic wrapping root — see
`data/raw/title17_sox.xml` for the exact shape
(`<SOXSECTIONS N="17">...<DIV8 .../>...</SOXSECTIONS>`); the wrapper tag
name doesn't matter, only that `Element.iter("DIV8")` can still find every
section inside it.

Sanity-check every download before moving on — `wc -l` it and `head -c 500`
it; a few hundred bytes of `{"error": ...}` means compression wasn't
requested, the date was wrong, or the part/section number is wrong.

## Rebuilding the knowledge base

`src/parse_ecfr.py` globs every `data/raw/title*.xml` file, so dropping a
correctly-named new XML file there is enough — then rebuild:

```bash
python3 src/parse_ecfr.py
```

It parses `DIV8`/`SECTION`-level elements (the eCFR bulk-XML section unit)
regardless of nesting depth, and will print how many sections it found
per file plus the total written to `data/kb.json`. If a new file parses
to 0 sections, check that the XML actually uses `DIV8` for sections (most
CFR titles do) before assuming the part/section number is wrong.

## After adding data: rebuild the dense embedding index — do not skip this

`data/kb.json` is not what gets searched at query time — `data/embeddings.npz`
is, and it's a separate, precomputed file. Adding sections to `kb.json`
without re-running the embedding build means the new content is parsed
and real, but **completely invisible to retrieval**: not poorly ranked,
just absent, because its vectors don't exist yet. See
`bedrock-rag-retrieval` before touching anything retrieval-related, but
at minimum:

```bash
python3 src/build_embeddings.py   # re-embeds chunks not already in data/embeddings.npz
```

This makes real, billed Bedrock calls (one per new chunk — see
`bedrock-rag-retrieval` for the cost estimate and why it's deliberately
slow/low-concurrency) and is resumable if interrupted.

## After adding data: the rest

1. Confirm the new sections are retrievable — query the `Retriever`
   directly with a question specific to the new domain (this also makes
   a real embedding call, same cost note as above):
   ```bash
   PYTHONPATH=src python3 -c "from retriever import Retriever; r = Retriever.build(); [print(h['source_id'], h['score']) for h in r.search('<question>')]"
   ```
2. If the new domain needs its own intent (it will, unless it's closely
   related to an existing one), add training examples and a label to
   `LABEL_DESCRIPTIONS` in `src/classifier.py` — see
   `edit-compliance-graph` if that also means touching `agents.py`.
3. Run `python3 demo.py "<question about the new domain>"` to confirm
   end-to-end retrieval + grounding + Bedrock summarization works.
4. Update the "Dataset" section (and "Honest limitations" if the new data
   has a caveat worth flagging) of README.md — don't leave it undocumented.
