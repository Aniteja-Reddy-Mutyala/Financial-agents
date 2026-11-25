"""Tiny intent classifier for financial compliance queries.

Stands in for the transformer-based intent/sentiment classification in the
original project. Trained on a handful of labeled example queries so the
pipeline has a real (if small) ML component instead of hand-written if/else.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

TRAIN_EXAMPLES = [
    ("do we need to file a sar for this wire transfer", "sar_filing"),
    ("when must a suspicious activity report be submitted", "sar_filing"),
    ("sar filing deadline after detection", "sar_filing"),
    ("what is the threshold for filing a currency transaction report", "ctr_filing"),
    ("ctr requirements for cash deposits over 10000", "ctr_filing"),
    ("does a $12000 cash deposit trigger a ctr", "ctr_filing"),
    ("what records must a bank keep on wire transfers", "recordkeeping"),
    ("how long do we retain customer due diligence records", "recordkeeping"),
    ("recordkeeping requirements for funds transfers", "recordkeeping"),
    ("what does a bank's aml compliance program need to include", "aml_program"),
    ("minimum requirements for an anti-money laundering program", "aml_program"),
    ("do we need a designated compliance officer for bsa", "aml_program"),
    ("can we share suspicious activity information with another bank", "info_sharing"),
    ("rules on sharing sar information between financial institutions", "info_sharing"),
    ("what are the requirements for electronic signatures on FDA records", "fda_e_records"),
    ("does our electronic record system need an audit trail for 21 CFR part 11", "fda_e_records"),
    ("validation requirements for electronic signature components and controls", "fda_e_records"),
    ("what safeguards does HIPAA require for protected health information", "hipaa_privacy_security"),
    ("minimum necessary standard for disclosing patient health information", "hipaa_privacy_security"),
    ("administrative requirements for a HIPAA privacy and security program", "hipaa_privacy_security"),
    ("does the CEO need to certify internal control over financial reporting", "sox_internal_controls"),
    ("sox 404 management assessment of internal controls", "sox_internal_controls"),
    ("CFO certification requirements for quarterly SEC filings", "sox_internal_controls"),
]

# Not a regulatory domain, and deliberately NOT one of IntentClassifier's
# trained labels — a query that's just a greeting, small talk, or asking
# about the assistant itself. A first attempt added "chitchat" as a 9th
# label on the real classifier; measured regression: with only ~3
# training examples per class already, the new class's intercept/prior
# was enough to flip several previously-correct real compliance queries
# (e.g. "Tell me the format of SAR", which shares the token "tell" with
# one chitchat example) to "chitchat" instead — not from genuine
# ambiguity, just the fragile model's tiny-sample decision boundaries
# shifting. See is_chitchat() below for the actual (deterministic,
# classifier-independent) detection mechanism, and agents.py's
# route_after_classify/CHITCHAT_LABEL.
CHITCHAT_LABEL = "chitchat"
CHITCHAT_DESCRIPTION = (
    "Casual conversation — a greeting, small talk, or a question about the "
    "assistant itself, not a regulatory compliance question"
)

# Deliberately a short, exact-match (after light normalization) phrase
# list, not a fuzzy/substring check — a substring check on words like
# "hi" or "help" would false-positive on real compliance text ("this
# bank" contains "hi"... no it doesn't as a whole word, but overly loose
# matching is exactly the kind of fragility that broke the classifier-
# based approach above). Expand this list deliberately, re-running
# tests/test_classifier.py's real-query regression cases after each
# addition — see write-compliance-tests.
_CHITCHAT_PHRASES = {
    "hi", "hi there", "hello", "hello there", "hey", "hey there",
    "good morning", "good afternoon", "good evening",
    "tell me about yourself", "who are you", "what are you",
    "what can you do", "what do you do", "how can you help", "how can you help me",
    "thanks", "thank you", "thanks for your help", "thank you for your help",
    "bye", "goodbye", "see you",
}


def is_chitchat(query: str) -> bool:
    """True if `query` is a greeting/small-talk aside rather than a real
    compliance question — see CHITCHAT_LABEL above for why this is a
    separate, deterministic check instead of a label on IntentClassifier.
    """
    normalized = query.strip().lower().rstrip("!.?")
    return normalized in _CHITCHAT_PHRASES


LABEL_DESCRIPTIONS = {
    "sar_filing": "Suspicious Activity Report (SAR) filing obligations",
    "ctr_filing": "Currency Transaction Report (CTR) filing obligations",
    "recordkeeping": "BSA recordkeeping requirements",
    "aml_program": "AML compliance program requirements",
    "info_sharing": "Inter-institution suspicious-activity information sharing",
    "fda_e_records": "FDA electronic records and electronic signature requirements (21 CFR Part 11)",
    "hipaa_privacy_security": "HIPAA Privacy and Security Rule requirements for protected health information",
    "sox_internal_controls": "Sarbanes-Oxley internal control and executive certification requirements",
}


class IntentClassifier:
    def __init__(self):
        texts = [t for t, _ in TRAIN_EXAMPLES]
        labels = [l for _, l in TRAIN_EXAMPLES]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, labels)

    def predict(self, query):
        X = self.vectorizer.transform([query])
        label = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        confidence = float(max(proba))
        return {
            "intent": label,
            "description": LABEL_DESCRIPTIONS.get(label, label),
            "confidence": confidence,
            # Whether the query shared ANY vocabulary with the training
            # set (X.nnz > 0) — not the same thing as `confidence`. A
            # genuinely correct classification can legitimately have low
            # confidence (softmax over 8 classes with a tiny training
            # set), but has_signal=False means the model had literally
            # nothing real to go on and its top guess is pure intercept-
            # driven noise. planner_node uses this to decide whether the
            # intent description is trustworthy enough to pad the
            # retrieval query with — see edit-compliance-graph.
            "has_signal": bool(X.nnz > 0),
        }


if __name__ == "__main__":
    clf = IntentClassifier()
    for q in [
        "do I have to file a SAR for a $50,000 unexplained wire",
        "what's the CTR threshold again",
        "how long do we keep wire transfer records",
    ]:
        print(q, "->", clf.predict(q))
