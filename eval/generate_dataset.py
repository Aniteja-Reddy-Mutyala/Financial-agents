"""Generates a large, templated evaluation dataset of realistic compliance
scenarios — ~1,400 cases across the 8 intents/4 domains, each with
auto-derived ground truth (expected intent, acceptable citations, and
"fact keywords" the generated answer should mention if it's actually
correct, not just plausible-sounding).

Free and fast (no AWS calls, no network) — this only builds the dataset.
Running the dataset against the real pipeline is a separate, deliberately
gated step: see run_eval.py and the real-cost/time estimate it prints
before doing anything.

Ground truth is derived from citations and facts already validated
elsewhere in this project's test suite (tests/test_live_rag.py), not
guessed — see GROUND_TRUTH below for the exact source section(s) and
distinctive phrase/number each intent's answer should contain if correct.
"""
import itertools
import json
import random
from pathlib import Path

OUT_PATH = Path(__file__).resolve().parent / "dataset.json"
SEED = 20261124  # fixed seed: dataset is reproducible, not re-randomized per run

# Ground truth per intent, two tiers:
#
# 1. `citation_prefix` — the CFR title (and, for the 4 non-BSA/AML
#    domains, part) a correct answer MUST cite from. This is intentionally
#    broad, not one specific section: a pilot run on the first dataset
#    draft showed several genuinely correct answers citing a *different*
#    real section than one narrow hand-picked "the" answer (e.g. FDA
#    citing 21 CFR 11.10 — a real, correct section — when only 11.100/
#    11.200/11.70 were in the original narrow set). All 5 BSA/AML intents
#    share "31 CFR" as their prefix rather than being split further,
#    because they also legitimately cite across each other's sections
#    (e.g. a CTR question's evidence can correctly include a SAR section
#    too) — the real signal worth checking at this dataset's scale is
#    "did it ground in the right regulatory family," not "the one exact
#    section I guessed."
# 2. `keywords` — a distinctive number or phrase from the real regulatory
#    text, reported as a softer "specific-fact mention rate," not a
#    pass/fail gate — the pilot also showed correct, well-grounded answers
#    that simply didn't restate a specific dollar figure in that exact
#    wording, which is a phrasing choice, not an error.
GROUND_TRUTH = {
    "sar_filing": {
        "citation_prefix": "31 CFR",
        "keywords": ["suspicious activity report", "sar"],
    },
    "ctr_filing": {
        "citation_prefix": "31 CFR",
        "keywords": ["$10,000", "10,000"],
    },
    "recordkeeping": {
        "citation_prefix": "31 CFR",
        "keywords": ["$3,000", "3,000", "retain", "retention"],
    },
    "aml_program": {
        "citation_prefix": "31 CFR",
        "keywords": ["anti-money laundering program", "aml program", "compliance program"],
    },
    "info_sharing": {
        "citation_prefix": "31 CFR",
        "keywords": ["information sharing", "safe harbor", "sharing", "disclos"],
    },
    "fda_e_records": {
        "citation_prefix": "21 CFR 11",
        "keywords": ["electronic signature", "electronic record"],
    },
    "hipaa_privacy_security": {
        "citation_prefix": "45 CFR",
        "keywords": ["protected health information", "phi"],
    },
    "sox_internal_controls": {
        "citation_prefix": "17 CFR",
        "keywords": ["internal control over financial reporting", "certif", "internal control"],
    },
}

# Each intent gets several scenario templates (factual + "given this
# situation" phrasing, mirroring README's Example queries) with
# placeholders filled from realistic value pools — combinatorial
# expansion gives natural variety without 1,400 hand-written sentences.

AMOUNTS = ["$9,500", "$10,000", "$10,500", "$12,000", "$15,000", "$25,000", "$50,000", "$100,000"]
SMALL_AMOUNTS = ["$2,500", "$3,000", "$3,500", "$5,000", "$7,500"]
TRANSACTION_TYPES = ["cash deposit", "wire transfer", "cashier's check purchase", "currency exchange", "money order purchase"]
CUSTOMER_TYPES = ["a new retail customer", "a longtime business customer", "a non-resident alien", "a shell company", "a cash-intensive business"]
SUSPICION_DESCRIPTORS = [
    "structured to avoid reporting thresholds",
    "inconsistent with the customer's known business",
    "linked to a high-risk jurisdiction",
    "part of a rapid series of similar transactions",
    "otherwise unexplained",
]
RECORD_TYPES = ["funds transfer records", "wire transfer originator information", "payment order records", "customer due diligence records"]
RETENTION_CONTEXTS = ["a domestic wire transfer", "an international funds transfer", "a payment order we originated", "a payment order we received as a beneficiary's bank"]
INSTITUTION_TYPES = [
    "a community bank", "a national bank", "a credit union", "a savings association",
    "a regional bank holding company", "a de novo bank", "an industrial loan company",
]
AML_PROGRAM_ELEMENTS = [
    "designated compliance officer", "employee training program", "independent testing function",
    "customer due diligence procedures", "risk-based policies and procedures",
    "ongoing monitoring process", "beneficial ownership identification process",
]
COUNTERPARTIES = [
    "another bank", "a law enforcement agency", "a regulatory examiner", "a correspondent bank",
    "a foreign financial institution", "a grand jury", "a self-regulatory organization",
]
INFO_TYPES = [
    "SAR information", "the fact that a SAR was filed", "suspicious activity details",
    "customer identification information", "the existence of a law enforcement inquiry",
    "underlying transaction records referenced in a SAR",
]
PHI_ACTIONS = [
    "disclosed to a third party", "accessed without authorization", "used for marketing purposes",
    "shared with a business associate", "disclosed in response to a subpoena",
    "transmitted electronically to another provider", "included in a research dataset",
]
ENTITY_TYPES = [
    "a hospital", "a covered entity", "a business associate", "a health plan",
    "a clearinghouse", "a physician practice", "a third-party administrator",
]
FDA_REQUIREMENT_TYPES = [
    "audit trail", "signature validation", "identity verification", "record retention",
    "system access control", "data integrity check", "time-stamped record control",
]
FDA_RECORD_TYPES = [
    "signatures", "batch records", "laboratory records", "clinical trial records",
    "manufacturing records", "quality control records", "device history records",
]
ROLES = [
    "the CEO", "the CFO", "the principal executive officer", "the principal financial officer",
    "the audit committee", "the registrant's management", "the controller",
]
CONTROL_TYPES = [
    "internal control over financial reporting", "disclosure controls and procedures",
    "the effectiveness of internal controls", "material weaknesses identified during the fiscal year",
    "changes in internal control over financial reporting", "the scope of management's assessment",
]

random.seed(SEED)


def _cap(s):
    return s[0].upper() + s[1:] if s else s


def _gen_sar_filing():
    templates = [
        "do we need to file a SAR for a {amount} {txn} that is {suspicion}?",
        "a customer made a {amount} {txn} that is {suspicion} — what should we do?",
        "what is the SAR filing deadline after detecting {suspicion} activity on a {txn}?",
        "does {customer} conducting a {amount} {txn} that is {suspicion} require a SAR?",
    ]
    cases = []
    for t, amt, txn, susp, cust in itertools.product(
        templates, AMOUNTS, TRANSACTION_TYPES, SUSPICION_DESCRIPTORS, CUSTOMER_TYPES
    ):
        cases.append(t.format(amount=amt, txn=txn, suspicion=susp, customer=cust))
    return cases


def _gen_ctr_filing():
    templates = [
        "does a {amount} {txn} by {customer} trigger a CTR?",
        "what is the CTR threshold for a {txn}?",
        "a customer made a {amount} {txn} today — do we need to file a CTR?",
        "is a {amount} {txn} in currency reportable under the CTR rules?",
    ]
    cases = []
    for t, amt, txn, cust in itertools.product(templates, AMOUNTS, TRANSACTION_TYPES, CUSTOMER_TYPES):
        cases.append(t.format(amount=amt, txn=txn, customer=cust))
    return cases


def _gen_recordkeeping():
    templates = [
        "how long must we retain {record} for {context}?",
        "what records must a bank keep for {context} of {amount} or more?",
        "are {record} required to be retained for {context}?",
        "what is the retention period for {record} related to {context}?",
    ]
    cases = []
    for t, rec, ctx, amt in itertools.product(templates, RECORD_TYPES, RETENTION_CONTEXTS, SMALL_AMOUNTS):
        cases.append(t.format(record=rec, context=ctx, amount=amt))
    return cases


def _gen_aml_program():
    templates = [
        "what must {inst}'s AML program include regarding a {element}?",
        "do we need a {element} as part of our AML compliance program at {inst}?",
        "what are the minimum requirements for an AML program's {element} at {inst}?",
        "is a {element} legally required for {inst}'s anti-money laundering program?",
    ]
    cases = []
    for t, inst, elem in itertools.product(templates, INSTITUTION_TYPES, AML_PROGRAM_ELEMENTS):
        cases.append(t.format(inst=inst, element=elem))
    return cases


def _gen_info_sharing():
    templates = [
        "can we share {info} with {party}?",
        "are we permitted to disclose {info} to {party}?",
        "what are the rules on sharing {info} with {party}?",
        "is there a safe harbor for sharing {info} with {party}?",
    ]
    cases = []
    for t, info, party in itertools.product(templates, INFO_TYPES, COUNTERPARTIES):
        cases.append(t.format(info=info, party=party))
    return cases


def _gen_fda_e_records():
    templates = [
        "what {req} is required for electronic {rec} under 21 CFR Part 11?",
        "does our system need {req} for {rec} to comply with FDA electronic records rules?",
        "what audit trail requirements apply to {rec}?",
        "what does the FDA require for {req} of {rec}?",
    ]
    cases = []
    for t, req, rec in itertools.product(templates, FDA_REQUIREMENT_TYPES, FDA_RECORD_TYPES):
        cases.append(t.format(req=req, rec=rec))
    return cases


def _gen_hipaa_privacy_security():
    templates = [
        "what safeguards apply when PHI is {action} by {entity}?",
        "what are {entity}'s obligations when protected health information is {action}?",
        "does HIPAA require notification when PHI is {action} at {entity}?",
        "what administrative safeguards must {entity} have if PHI is {action}?",
    ]
    cases = []
    for t, action, entity in itertools.product(templates, PHI_ACTIONS, ENTITY_TYPES):
        cases.append(t.format(action=action, entity=entity))
    return cases


def _gen_sox_internal_controls():
    templates = [
        "what must {role} certify about {control}?",
        "does {role} need to sign off on {control} each quarter?",
        "what is required of {role} regarding {control} before a filing?",
        "what does SOX require {role} to certify about {control}?",
    ]
    cases = []
    for t, role, control in itertools.product(templates, ROLES, CONTROL_TYPES):
        cases.append(t.format(role=role, control=control))
    return cases


GENERATORS = {
    "sar_filing": _gen_sar_filing,
    "ctr_filing": _gen_ctr_filing,
    "recordkeeping": _gen_recordkeeping,
    "aml_program": _gen_aml_program,
    "info_sharing": _gen_info_sharing,
    "fda_e_records": _gen_fda_e_records,
    "hipaa_privacy_security": _gen_hipaa_privacy_security,
    "sox_internal_controls": _gen_sox_internal_controls,
}

PER_INTENT_TARGET = 180  # ~180 x 8 = ~1,440 total


def build_dataset():
    dataset = []
    for intent, gen_fn in GENERATORS.items():
        all_cases = gen_fn()
        random.shuffle(all_cases)
        sampled = all_cases[:PER_INTENT_TARGET]
        gt = GROUND_TRUTH[intent]
        for query in sampled:
            dataset.append(
                {
                    "query": _cap(query),
                    "expected_intent": intent,
                    "expected_citation_prefix": gt["citation_prefix"],
                    "expected_keywords": gt["keywords"],
                }
            )
    random.shuffle(dataset)
    for i, case in enumerate(dataset):
        case["case_id"] = i
    return dataset


if __name__ == "__main__":
    dataset = build_dataset()
    OUT_PATH.write_text(json.dumps(dataset, indent=2))
    print(f"Generated {len(dataset)} cases -> {OUT_PATH}")
    from collections import Counter

    print(Counter(c["expected_intent"] for c in dataset))
