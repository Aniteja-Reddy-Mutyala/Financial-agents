"""Opt-in end-to-end tests of REAL dense retrieval against the REAL
215-section knowledge base via Amazon Bedrock Titan embeddings.

Every test here makes at least one real, billed Bedrock embedding call
(fractions of a cent — see bedrock-rag-retrieval skill) and requires
data/embeddings.npz to already exist (run `python3 src/build_embeddings.py`
first). Skipped by default, same gate as test_live_bedrock.py, since both
exist for the same reason: real AWS cost. Run explicitly with:

    RUN_LIVE_BEDROCK_TESTS=1 python3 -m pytest tests/test_live_rag.py -v

These are the domain-relevance checks that can't be done with a fake
embedder (tests/test_retriever.py covers the retrieval *machinery* —
sorting, dedup, top_k, the empty-query short-circuit — for free, against
a synthetic corpus). This file checks the real thing: does "what's the
CTR threshold" actually find 31 CFR 1010.311 out of the real KB.
"""
import os

import pytest

from agents import ComplianceAgentGraph
from retriever import Retriever

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_BEDROCK_TESTS") != "1",
    reason="live Bedrock/embedding test disabled by default (costs real money); "
    "set RUN_LIVE_BEDROCK_TESTS=1 to run it",
)


@pytest.fixture(scope="module")
def real_retriever():
    return Retriever.build()


def _source_ids(hits):
    return [h["source_id"] for h in hits]


# --- MIN_SCORE calibration --------------------------------------------------


def test_min_score_separates_relevant_from_off_topic_queries(real_retriever):
    # Regression guard for a real finding: when retrieval moved from TF-IDF
    # to dense Titan embeddings, agents.MIN_SCORE (left at its old TF-IDF
    # value, 0.08) stopped doing its job — dense cosine similarity has a
    # much higher "noise floor" than TF-IDF (irrelevant text still shares
    # general vocabulary/structure), so every off-topic query cleared 0.08
    # and the escalate branch was effectively dead. Measured against this
    # real index: relevant queries scored 0.37-0.70 top-hit; off-topic
    # queries scored 0.13-0.16. MIN_SCORE was recalibrated to 0.25. This
    # test locks in that there's still real margin on both sides — if a
    # future KB/model change erodes that margin, this will catch it.
    from agents import MIN_SCORE

    relevant_queries = [
        "suspicious activity report filing deadline for banks",
        "what is the CTR threshold for a cash deposit",
        "HIPAA security rule safeguards for protected health information",
        "management internal control over financial reporting certification",
        "electronic signature validation requirements for FDA records",
    ]
    off_topic_queries = [
        "what's the best pizza topping in Chicago",
        "the weather in Seattle is usually rainy in autumn",
        "recommend a good science fiction novel",
    ]
    for q in relevant_queries:
        top_score = real_retriever.search(q, top_k=1)[0]["score"]
        assert top_score > MIN_SCORE, f"{q!r} scored {top_score:.3f}, expected > {MIN_SCORE}"
    for q in off_topic_queries:
        top_score = real_retriever.search(q, top_k=1)[0]["score"]
        assert top_score < MIN_SCORE, f"{q!r} scored {top_score:.3f}, expected < {MIN_SCORE}"


def test_fully_off_topic_query_now_correctly_escalates():
    # Regression test for the "known rough edge" fix: planner_node used to
    # pad the retrieval query with the classified intent's description
    # unconditionally, so a genuinely off-topic question could still score
    # above MIN_SCORE purely from the padding, regardless of whether the
    # classified intent was even real. Root cause, found by testing the
    # cheaper alternatives first and empirically ruling them out (a raw-
    # query-only score floor and a classifier-confidence floor both risked
    # new false escalations on real vague-but-legitimate queries — see
    # README's "Honest limitations" for the measured data): the classifier
    # shares literally zero training vocabulary with a truly off-topic
    # query (has_signal=False — see test_classifier.py), meaning its top
    # "intent" is pure class-prior noise. planner_node now only pads the
    # retrieval query when has_signal is True; otherwise it searches the
    # raw query alone, which correctly scores below MIN_SCORE for this
    # query (measured ~0.146, see test_min_score_separates_... above).
    # This was an `xfail(strict=True)` before the fix landed — if it ever
    # starts failing again, that's a real regression, not flakiness.
    agent = ComplianceAgentGraph()
    result = agent.handle("what's the best pizza topping in Chicago")
    assert result.verification["grounded"] is False
    assert "⚠️" in result.answer
    assert "human compliance officer" in result.answer


# --- Real domain relevance, one per regulatory domain in the KB ------------


def test_sar_query_surfaces_sar_section(real_retriever):
    hits = real_retriever.search("suspicious activity report filing deadline for banks", top_k=3)
    assert hits
    assert any(h["source_id"] == "31 CFR 1020.320" for h in hits)


def test_ctr_query_surfaces_ctr_section(real_retriever):
    hits = real_retriever.search(
        "what is the CTR threshold for a cash deposit "
        "Currency Transaction Report (CTR) filing obligations",
        top_k=5,
    )
    assert hits
    assert any("1010.311" in h["source_id"] for h in hits)


def test_bare_ctr_phrasing_surfaces_a_ctr_filing_section(real_retriever):
    # Dense embeddings do bridge "CTR"/"threshold"/"cash" to the
    # regulation's own "currency transaction report"/"$10,000"/"currency"
    # wording natively (measured: 31 CFR 1010.311 scores 0.456 here, well
    # inside the "relevant" range — see MIN_SCORE's calibration comment in
    # agents.py) — but 31 CFR actually has *several* equally-valid
    # currency-transaction-reporting sections (1010.310, 1010.311,
    # 1010.330, 1020.310), all clustering closely together, so bare
    # phrasing with no other context can surface any of them in the top 5
    # depending on exact wording. Asserting one specific section ID here
    # (as the old TF-IDF-era test did) overfits the test to one arbitrary
    # "correct" answer among several real ones. Accept any of them.
    ctr_filing_sections = {
        "31 CFR 1010.310",
        "31 CFR 1010.311",
        "31 CFR 1010.330",
        "31 CFR 1020.310",
    }
    hits = real_retriever.search("currency transaction report threshold cash deposit", top_k=5)
    assert any(h["source_id"] in ctr_filing_sections for h in hits)


def test_hipaa_query_surfaces_part_164(real_retriever):
    hits = real_retriever.search("HIPAA security rule safeguards for protected health information", top_k=3)
    assert hits
    assert any("45 CFR 164" in h["source_id"] for h in hits)


def test_sox_query_surfaces_internal_controls_section(real_retriever):
    hits = real_retriever.search("management internal control over financial reporting certification", top_k=3)
    assert hits
    assert any("229.308" in h["source_id"] for h in hits)


def test_fda_e_records_query_surfaces_part_11(real_retriever):
    hits = real_retriever.search("electronic signature validation requirements for FDA records", top_k=3)
    assert hits
    assert any("21 CFR 11" in h["source_id"] for h in hits)


# --- Full-graph regression: the real "Tell me the format of SAR" bug ------


def test_wrong_intent_does_not_pollute_evidence_with_other_domain():
    # Regression test: "Tell me the format of SAR" used to misclassify as
    # sox_internal_controls (see test_classifier.py), and because
    # planner_node pads the retrieval query with the (wrong) intent's
    # description, a 17 CFR SOX section was showing up as "grounded"
    # evidence for a plain BSA/AML question. Needs the real retriever
    # (not a synthetic one) because the bug was specifically about real
    # cross-domain evidence leaking in — a fake corpus can't reproduce it.
    class _FakeSummarizer:
        model_id = "fake-test-model"

        def summarize(self, query, intent_description, evidence):
            return "fake answer"

    agent = ComplianceAgentGraph(llm=_FakeSummarizer(), retriever=Retriever.build())
    result = agent.handle("Tell me the format of SAR")

    assert result.intent["intent"] == "sar_filing"
    assert result.verification["grounded"] is True
    for citation in result.verification["citations"]:
        assert citation.startswith("31 CFR"), citation
