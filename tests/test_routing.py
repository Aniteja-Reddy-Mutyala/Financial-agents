"""Unit tests for the compliance gate (verifier_node + route_after_verify +
escalate_node) and retriever_node's raw/augmented-query merge. These operate
on hand-built state dicts and fake retrievers, so they run with no
network/AWS calls and exercise the logic directly rather than relying on
retrieval happening to produce (or not produce) particular scores.
"""
import pytest

from agents import (
    MIN_SCORE,
    NO_PAD_INTENTS,
    _unique_evidence,
    escalate_node,
    planner_node,
    retriever_node,
    route_after_verify,
    verifier_node,
)


def _evidence(source_id, score, title="Some section", text="Some regulatory text."):
    return {"chunk_id": f"{source_id}#0", "source_id": source_id, "title": title, "text": text, "score": score}


def _base_state(evidence):
    return {
        "query": "do we need to file a SAR",
        "intent": {"intent": "sar_filing", "description": "SAR filing obligations", "confidence": 0.9},
        "plan": {"retrieval_query": "do we need to file a SAR", "top_k": 3},
        "evidence": evidence,
        "trace": [],
    }


# --- verifier_node: grounding decision --------------------------------------


def test_grounded_when_evidence_clears_threshold():
    state = _base_state([_evidence("31 CFR 1020.320", MIN_SCORE + 0.1)])
    update = verifier_node(state)
    assert update["verification"]["grounded"] is True
    assert update["verification"]["citations"] == ["31 CFR 1020.320"]
    assert any("grounded=True" in line for line in update["trace"])


def test_not_grounded_with_no_evidence():
    state = _base_state([])
    update = verifier_node(state)
    assert update["verification"]["grounded"] is False
    assert update["verification"]["citations"] == []
    assert "escalate" in update["verification"]["reason"]


def test_not_grounded_when_all_scores_below_threshold():
    state = _base_state(
        [_evidence("31 CFR 1020.320", MIN_SCORE - 0.01), _evidence("31 CFR 1010.311", 0.0)]
    )
    update = verifier_node(state)
    assert update["verification"]["grounded"] is False
    assert update["verification"]["citations"] == []


def test_grounded_exactly_at_threshold_boundary():
    # MIN_SCORE itself should count as clearing the bar (>=, not >).
    state = _base_state([_evidence("31 CFR 1020.320", MIN_SCORE)])
    update = verifier_node(state)
    assert update["verification"]["grounded"] is True


def test_mixed_scores_keep_only_strong_evidence_as_citations():
    state = _base_state(
        [
            _evidence("31 CFR 1020.320", MIN_SCORE + 0.2),
            _evidence("31 CFR 1010.311", MIN_SCORE - 0.05),  # below bar, excluded
        ]
    )
    update = verifier_node(state)
    assert update["verification"]["grounded"] is True
    assert update["verification"]["citations"] == ["31 CFR 1020.320"]


def test_duplicate_source_ids_are_deduplicated_in_order():
    state = _base_state(
        [
            _evidence("31 CFR 1020.320", 0.5),
            _evidence("31 CFR 1020.320", 0.4),  # same section, different chunk
            _evidence("31 CFR 1010.311", 0.3),
        ]
    )
    update = verifier_node(state)
    assert update["verification"]["citations"] == ["31 CFR 1020.320", "31 CFR 1010.311"]


# --- route_after_verify: the conditional edge --------------------------------


def test_route_to_summarize_when_grounded():
    state = {"verification": {"grounded": True, "citations": ["31 CFR 1020.320"]}}
    assert route_after_verify(state) == "summarize"


def test_route_to_escalate_when_not_grounded():
    state = {"verification": {"grounded": False, "citations": []}}
    assert route_after_verify(state) == "escalate"


# --- escalate_node: the ungrounded branch (LLM, never given evidence) ------


class _FakeEscalateSummarizer:
    model_id = "fake-test-model"

    def __init__(self, reply="Hi! Ask me about BSA/AML, FDA, HIPAA, or SOX."):
        self.reply = reply
        self.calls = []

    def respond_ungrounded(self, query):
        self.calls.append(query)
        return self.reply


class _FailingEscalateSummarizer:
    model_id = "fake-test-model"

    def respond_ungrounded(self, query):
        from bedrock_llm import BedrockUnavailable

        raise BedrockUnavailable("simulated provider outage")


def test_escalate_node_returns_llm_reply():
    state = _base_state([])
    state["verification"] = {"grounded": False, "citations": [], "reason": "no evidence"}
    fake = _FakeEscalateSummarizer(reply="Hello! Ask me about BSA/AML, FDA, HIPAA, or SOX.")
    update = escalate_node(state, llm=fake)
    assert update["answer"] == "Hello! Ask me about BSA/AML, FDA, HIPAA, or SOX."
    assert fake.calls == ["do we need to file a SAR"]  # the query, nothing else
    assert any("generated ungrounded reply" in line for line in update["trace"])


def test_escalate_node_falls_back_to_static_warning_when_llm_unavailable():
    state = _base_state([])
    state["verification"] = {"grounded": False, "citations": [], "reason": "no evidence"}
    update = escalate_node(state, llm=_FailingEscalateSummarizer())
    assert "⚠️" in update["answer"]
    assert "human compliance officer" in update["answer"]
    assert "Citations:" not in update["answer"]
    assert any("LLM unavailable" in line for line in update["trace"])


# --- _unique_evidence: feeds the Summarizer ---------------------------------


def test_unique_evidence_filters_to_citations_only():
    state = _base_state(
        [
            _evidence("31 CFR 1020.320", 0.5),
            _evidence("31 CFR 1020.320", 0.4),  # dup, should collapse to one
            _evidence("31 CFR 1010.311", MIN_SCORE - 0.05),  # below bar
        ]
    )
    state["verification"] = {"grounded": True, "citations": ["31 CFR 1020.320"], "reason": "ok"}
    evidence = _unique_evidence(state)
    assert [e["source_id"] for e in evidence] == ["31 CFR 1020.320"]


def test_unique_evidence_empty_when_no_citations():
    state = _base_state([_evidence("31 CFR 1020.320", 0.5)])
    state["verification"] = {"grounded": False, "citations": [], "reason": "none"}
    assert _unique_evidence(state) == []


# --- retriever_node: merging raw-query + intent-augmented-query hits -------


class _ScriptedRetriever:
    """Returns different canned hits depending on which query string it's
    asked to search, so we can test the merge logic deterministically
    without depending on the real TF-IDF corpus or the intent classifier."""

    def __init__(self, by_query):
        self.by_query = by_query

    def search(self, query, top_k=3):
        return self.by_query.get(query, [])[:top_k]


def test_retriever_node_merges_raw_and_augmented_hits():
    # A wrong/low-confidence intent can make the augmented query surface a
    # section the raw query never would (and vice versa). Both should be
    # considered, deduplicated by source_id, best score kept.
    raw_only = _evidence("31 CFR 1020.320", 0.40)
    augmented_only = _evidence("17 CFR 229.308", 0.17)
    shared_raw_score = _evidence("31 CFR 1010.540", 0.39)
    shared_augmented_score = _evidence("31 CFR 1010.540", 0.25)  # same section, lower score

    retriever = _ScriptedRetriever(
        {
            "raw query": [raw_only, shared_raw_score],
            "raw query SOX description": [shared_augmented_score, augmented_only],
        }
    )
    state = {
        "query": "raw query",
        "plan": {"retrieval_query": "raw query SOX description", "top_k": 3},
        "trace": [],
    }
    result = retriever_node(state, retriever=retriever)
    hits = result["evidence"]

    source_ids = [h["source_id"] for h in hits]
    assert len(source_ids) == len(set(source_ids)), "no section should appear twice"
    assert "31 CFR 1020.320" in source_ids  # raw-only hit must survive
    assert "17 CFR 229.308" in source_ids  # augmented-only hit also kept (not discarded)

    # The duplicated section keeps its higher (raw-query) score, not the
    # lower augmented-query one.
    merged_1010_540 = next(h for h in hits if h["source_id"] == "31 CFR 1010.540")
    assert merged_1010_540["score"] == 0.39


def test_retriever_node_respects_top_k_after_merging():
    hits_by_query = {
        "q": [_evidence(f"31 CFR {i}", 1.0 - i / 10) for i in range(5)],
        "q aug": [_evidence(f"45 CFR {i}", 0.9 - i / 10) for i in range(5)],
    }
    retriever = _ScriptedRetriever(hits_by_query)
    state = {"query": "q", "plan": {"retrieval_query": "q aug", "top_k": 3}, "trace": []}
    result = retriever_node(state, retriever=retriever)
    assert len(result["evidence"]) == 3


def test_retriever_node_skips_redundant_search_when_planner_did_not_pad():
    # Regression guard for the has_signal fix: when planner_node decides
    # NOT to pad (because the classifier had zero real vocabulary
    # overlap), plan["retrieval_query"] equals state["query"] exactly.
    # retriever_node must not spend a second (identical, wasted) Bedrock
    # embedding call searching the same string twice.
    search_calls = []

    class _CountingRetriever:
        def search(self, query, top_k=3):
            search_calls.append(query)
            return [_evidence("31 CFR 1010.100", 0.15)]

    state = {"query": "same query", "plan": {"retrieval_query": "same query", "top_k": 3}, "trace": []}
    result = retriever_node(state, retriever=_CountingRetriever())

    assert search_calls == ["same query"]  # searched exactly once, not twice
    assert len(result["evidence"]) == 1


# --- planner_node: has_signal gates whether the query gets padded ----------


def test_planner_pads_query_when_classifier_has_signal():
    state = {
        "query": "do we need to file a SAR",
        "intent": {"intent": "sar_filing", "description": "SAR filing obligations", "has_signal": True},
        "trace": [],
    }
    result = planner_node(state)
    assert result["plan"]["retrieval_query"] == "do we need to file a SAR SAR filing obligations"


def test_planner_does_not_pad_query_when_classifier_has_no_signal():
    # The real-world case this closes: a genuinely off-topic query (e.g.
    # "what's the best pizza topping in Chicago") shares zero vocabulary
    # with the classifier's tiny training set, so its top "intent" is
    # statistical noise — padding the retrieval query with that noise's
    # description was artificially inflating the retrieval score for
    # whatever domain the model happened to guess, regardless of the
    # query's actual topic. See README's "Honest limitations" for the
    # measured before/after scores.
    state = {
        "query": "what's the best pizza topping in Chicago",
        "intent": {"intent": "sox_internal_controls", "description": "Sarbanes-Oxley...", "has_signal": False},
        "trace": [],
    }
    result = planner_node(state)
    assert result["plan"]["retrieval_query"] == state["query"]


# --- planner_node: NO_PAD_INTENTS override has_signal for broad intents ---


@pytest.mark.parametrize("intent", sorted(NO_PAD_INTENTS))
def test_planner_does_not_pad_broad_multi_topic_intents_even_with_signal(intent):
    # GitHub issue #1: fda_e_records's and hipaa_privacy_security's single
    # fixed description can't represent all of their sub-topics, so padding
    # with it skewed retrieval toward whichever sub-topic the description's
    # wording favors (e.g. signatures) even for a real, on-topic, clearly
    # classified question about a different sub-topic (e.g. audit trails).
    # has_signal=True here proves the override applies regardless of
    # classifier confidence/signal, not just as a special case of it.
    state = {
        "query": "does our electronic record system need an audit trail",
        "intent": {"intent": intent, "description": "some broad description", "has_signal": True},
        "trace": [],
    }
    result = planner_node(state)
    assert result["plan"]["retrieval_query"] == state["query"]


def test_planner_still_pads_non_broad_intents_with_signal():
    # Regression guard: the NO_PAD_INTENTS override must not swallow the
    # normal padding path for every other intent.
    state = {
        "query": "do we need to file a SAR",
        "intent": {"intent": "sar_filing", "description": "SAR filing obligations", "has_signal": True},
        "trace": [],
    }
    result = planner_node(state)
    assert result["plan"]["retrieval_query"] == "do we need to file a SAR SAR filing obligations"
