"""Tests for the intent classifier, including ambiguous and edge-case inputs.

No network/AWS calls — pure scikit-learn, fast.
"""
import pytest

from classifier import IntentClassifier, LABEL_DESCRIPTIONS

VALID_LABELS = set(LABEL_DESCRIPTIONS.keys())


@pytest.fixture(scope="module")
def clf():
    return IntentClassifier()


# --- Clear-cut cases: one intent, unambiguous phrasing ----------------------

CLEAR_CASES = [
    ("do we need to file a SAR for this wire transfer", "sar_filing"),
    ("what is the suspicious activity report filing deadline", "sar_filing"),
    # Regression test: this used to misclassify as sox_internal_controls at
    # 15% confidence (near-random across 8 classes) because the classifier's
    # vectorizer didn't strip stop words, so "of"/"the" diluted the one real
    # signal word ("sar") down to noise. Fixed by adding stop_words="english"
    # to IntentClassifier's TfidfVectorizer (matching retriever.py's existing
    # stop-word filtering).
    ("Tell me the format of SAR", "sar_filing"),
    ("what's the CTR threshold for a $12,000 cash deposit", "ctr_filing"),
    ("currency transaction report requirements for banks", "ctr_filing"),
    ("how long must we retain wire transfer records", "recordkeeping"),
    ("customer due diligence record retention period", "recordkeeping"),
    ("what must a bank's AML compliance program include", "aml_program"),
    ("do we need a designated BSA compliance officer", "aml_program"),
    ("can we share SAR information with another financial institution", "info_sharing"),
    ("what are the audit trail requirements for electronic signatures on FDA records", "fda_e_records"),
    ("21 CFR part 11 requirements for electronic signature validation", "fda_e_records"),
    ("what safeguards does HIPAA require for protected health information", "hipaa_privacy_security"),
    ("minimum necessary standard for disclosing patient health information", "hipaa_privacy_security"),
    ("does the CEO need to certify internal control over financial reporting", "sox_internal_controls"),
    ("sox 404 management assessment of internal controls", "sox_internal_controls"),
]


@pytest.mark.parametrize("query,expected_intent", CLEAR_CASES)
def test_clear_intent_classification(clf, query, expected_intent):
    result = clf.predict(query)
    assert result["intent"] == expected_intent
    assert result["description"] == LABEL_DESCRIPTIONS[expected_intent]
    assert 0.0 <= result["confidence"] <= 1.0


# --- Ambiguous cases: phrasing that plausibly spans two intents -------------
# We don't assert a single "correct" label here (that would be over-fitting
# the test to the model's current quirks) — we assert the classifier still
# returns *some* valid, well-formed prediction instead of crashing or
# returning garbage.

AMBIGUOUS_QUERIES = [
    "what are our SAR and CTR filing deadlines",  # spans sar_filing + ctr_filing
    "recordkeeping requirements for suspicious activity reports",  # recordkeeping + sar_filing
    "does our AML program need SAR filing procedures",  # aml_program + sar_filing
    "filing",  # single word, intent-bearing but no context at all
]


@pytest.mark.parametrize("query", AMBIGUOUS_QUERIES)
def test_ambiguous_queries_degrade_gracefully(clf, query):
    result = clf.predict(query)
    assert result["intent"] in VALID_LABELS
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0
    # Ambiguous phrasing should not look artificially confident.
    assert result["confidence"] <= 0.95


# --- Edge cases --------------------------------------------------------------


def test_empty_string_does_not_crash(clf):
    result = clf.predict("")
    assert result["intent"] in VALID_LABELS


def test_whitespace_only_does_not_crash(clf):
    result = clf.predict("   \n\t  ")
    assert result["intent"] in VALID_LABELS


def test_punctuation_only_does_not_crash(clf):
    result = clf.predict("??? !!! ...")
    assert result["intent"] in VALID_LABELS


def test_very_long_query_does_not_crash(clf):
    long_query = "SAR filing requirements " * 500
    result = clf.predict(long_query)
    assert result["intent"] == "sar_filing"


def test_non_english_query_does_not_crash(clf):
    # The model is English-only (documented limitation) — this should
    # degrade gracefully (some label, not an exception), not necessarily
    # classify correctly.
    result = clf.predict("¿Necesitamos presentar un informe de actividad sospechosa?")
    assert result["intent"] in VALID_LABELS


def test_mixed_case_and_special_characters(clf):
    result = clf.predict("DO WE NEED TO FILE A ¡SAR!? ($10,000+++)")
    assert result["intent"] in VALID_LABELS


def test_off_topic_query_does_not_crash(clf):
    # Off-topic input (not a compliance question at all) — the classifier
    # has no "none of the above" option, so it will force-fit a label.
    # This is a known limitation: assert it degrades gracefully rather than
    # asserting any particular (meaningless) label.
    result = clf.predict("what's the best pizza topping in Chicago")
    assert result["intent"] in VALID_LABELS


# --- has_signal: distinguishes a real classification from pure noise -------


@pytest.mark.parametrize(
    "query",
    [
        "what's the best pizza topping in Chicago",
        "the weather in Seattle is usually rainy in autumn",
        "recommend a good science fiction novel",
        "what is the capital of France",
    ],
)
def test_off_topic_queries_have_no_signal(clf, query):
    # These share zero vocabulary with the training set, so the model's
    # top "intent" is pure class-prior noise, not a real classification —
    # has_signal=False is what lets planner_node know not to trust it
    # enough to pad the retrieval query with it (see edit-compliance-graph
    # skill's "known rough edge" fix).
    result = clf.predict(query)
    assert result["has_signal"] is False


@pytest.mark.parametrize("query,_expected", CLEAR_CASES)
def test_clear_cut_queries_have_signal(clf, query, _expected):
    result = clf.predict(query)
    assert result["has_signal"] is True
