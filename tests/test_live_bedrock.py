"""Opt-in end-to-end test against REAL Amazon Bedrock.

This makes an actual billed API call (fractions of a cent with Nova Micro,
but real money — see the bedrock-llm-integration skill). It is skipped by
default so the rest of the test suite stays free and runs on CI with no
AWS credentials. Run explicitly with:

    RUN_LIVE_BEDROCK_TESTS=1 python3 -m pytest tests/test_live_bedrock.py -v
"""
import os

import pytest

from agents import ComplianceAgentGraph

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_BEDROCK_TESTS") != "1",
    reason="live Bedrock test disabled by default (costs real money); "
    "set RUN_LIVE_BEDROCK_TESTS=1 to run it",
)


def test_live_grounded_query_produces_real_bedrock_answer():
    agent = ComplianceAgentGraph()  # real BedrockSummarizer, real Retriever
    result = agent.handle("do we need to file a SAR for a structured cash deposit pattern")

    assert result.verification["grounded"] is True
    assert "31 CFR" in result.answer
    assert "Citations:" in result.answer
    assert any("generated via Bedrock" in line for line in result.trace)
