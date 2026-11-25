# Evaluation Report

**Cases scored:** 1415  **Errored (after retries):** 1

Metrics are computed on **intent classification** (precision/recall/F1 from the
confusion matrix below) plus three pipeline-quality rates per class: whether the
answer was marked grounded, whether its citations came from the correct CFR
title/part (the primary correctness check — see generate_dataset.py's
GROUND_TRUTH comment for why this is prefix-based, not an exact-section match),
and an informational fact-keyword mention rate (phrasing varies even in correct
answers, so this is not a pass/fail signal).

## Overall

| Metric | Value |
|---|---|
| Accuracy (intent) | 80.3% |
| Macro-avg precision | 85.1% |
| Macro-avg recall | 79.9% |
| Macro-avg F1 | 79.3% |
| Grounded rate | 100.0% |
| Citation domain-correct rate | 99.9% |
| Fact-keyword mention rate (informational) | 95.9% |

## Per-intent class

| Intent | N | Precision | Recall | F1 | Grounded | Domain✓ | Fact-kw |
|---|---|---|---|---|---|---|---|
| aml_program | 180 | 95.1% | 97.8% | 96.4% | 100.0% | 100.0% | 95.6% |
| ctr_filing | 179 | 78.7% | 65.9% | 71.7% | 100.0% | 100.0% | 92.7% |
| fda_e_records | 180 | 98.4% | 99.4% | 98.9% | 100.0% | 100.0% | 94.4% |
| hipaa_privacy_security | 180 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| info_sharing | 168 | 100.0% | 26.8% | 42.3% | 100.0% | 100.0% | 98.8% |
| recordkeeping | 180 | 72.1% | 80.6% | 76.1% | 100.0% | 100.0% | 98.9% |
| sar_filing | 180 | 46.4% | 78.3% | 58.3% | 100.0% | 100.0% | 91.1% |
| sox_internal_controls | 168 | 90.5% | 90.5% | 90.5% | 100.0% | 99.4% | 95.8% |

## Confusion matrix (rows = expected, columns = actual predicted intent)

| expected \\ actual | aml_program | ctr_filing | fda_e_records | hipaa_privacy_security | info_sharing | recordkeeping | sar_filing | sox_internal_controls |
|---|---|---|---|---|---|---|---|---|
| **aml_program** | 176 | 0 | 0 | 0 | 0 | 4 | 0 | 0 |
| **ctr_filing** | 0 | 118 | 0 | 0 | 0 | 0 | 61 | 0 |
| **fda_e_records** | 0 | 0 | 179 | 0 | 0 | 0 | 0 | 1 |
| **hipaa_privacy_security** | 0 | 0 | 0 | 180 | 0 | 0 | 0 | 0 |
| **info_sharing** | 0 | 0 | 0 | 0 | 45 | 45 | 63 | 15 |
| **recordkeeping** | 0 | 0 | 0 | 0 | 0 | 145 | 35 | 0 |
| **sar_filing** | 0 | 32 | 0 | 0 | 0 | 7 | 141 | 0 |
| **sox_internal_controls** | 9 | 0 | 3 | 0 | 0 | 0 | 4 | 152 |

## Failures (279 of 1415)

A "failure" here means wrong intent label, not grounded, or grounded in the
wrong regulatory family entirely — not a missing fact-keyword, which is
informational only (see above).

- **[info_sharing → recordkeeping]** 'Is there a safe harbor for sharing the existence of a law enforcement inquiry with another bank?'  
  grounded=True, citations=['31 CFR 1010.520', '31 CFR 1010.540', '31 CFR 1020.320']
- **[aml_program → recordkeeping]** "What must a credit union's AML program include regarding a customer due diligence procedures?"  
  grounded=True, citations=['31 CFR 1010.610', '31 CFR 1010.620', '31 CFR 1020.210']
- **[sox_internal_controls → sar_filing]** 'What is required of the controller regarding disclosure controls and procedures before a filing?'  
  grounded=True, citations=['17 CFR 240.13a-15', '31 CFR 1010.540', '31 CFR 1020.320']
- **[info_sharing → sar_filing]** 'Are we permitted to disclose suspicious activity details to a law enforcement agency?'  
  grounded=True, citations=['31 CFR 1010.540', '31 CFR 1010.955', '31 CFR 1020.320']
- **[ctr_filing → sar_filing]** "A customer made a $50,000 cashier's check purchase today — do we need to file a CTR?"  
  grounded=True, citations=['31 CFR 1010.330', '31 CFR 1010.415', '31 CFR 1020.320']
- **[sar_filing → ctr_filing]** "A customer made a $50,000 cash deposit that is inconsistent with the customer's known business — what should we do?"  
  grounded=True, citations=['31 CFR 1010.310', '31 CFR 1010.311', '31 CFR 1010.330']
- **[ctr_filing → sar_filing]** 'Is a $15,000 wire transfer in currency reportable under the CTR rules?'  
  grounded=True, citations=['31 CFR 1010.310', '31 CFR 1010.311', '31 CFR 1010.330']
- **[info_sharing → sar_filing]** 'Are we permitted to disclose underlying transaction records referenced in a SAR to a self-regulatory organization?'  
  grounded=True, citations=['31 CFR 1010.540', '31 CFR 1010.955', '31 CFR 1020.320']
- **[info_sharing → sar_filing]** 'Is there a safe harbor for sharing underlying transaction records referenced in a SAR with a grand jury?'  
  grounded=True, citations=['31 CFR 1010.520', '31 CFR 1010.540', '31 CFR 1020.320']
- **[sox_internal_controls → aml_program]** 'Does the principal financial officer need to sign off on material weaknesses identified during the fiscal year each quarter?'  
  grounded=True, citations=['17 CFR 240.13a-14', '17 CFR 240.13a-15', '17 CFR 240.15d-14', '45 CFR 164.512']
- **[info_sharing → recordkeeping]** 'Are we permitted to disclose underlying transaction records referenced in a SAR to another bank?'  
  grounded=True, citations=['31 CFR 1020.320', '31 CFR 1020.400', '31 CFR 1020.410']
- **[ctr_filing → sar_filing]** 'A customer made a $100,000 currency exchange today — do we need to file a CTR?'  
  grounded=True, citations=['31 CFR 1010.311', '31 CFR 1020.315', '31 CFR 1020.320']
- **[ctr_filing → sar_filing]** 'A customer made a $12,000 wire transfer today — do we need to file a CTR?'  
  grounded=True, citations=['31 CFR 1010.311', '31 CFR 1020.315', '31 CFR 1020.320']
- **[recordkeeping → sar_filing]** 'What is the retention period for payment order records related to a domestic wire transfer?'  
  grounded=True, citations=['31 CFR 1010.370', '31 CFR 1020.320', '31 CFR 1020.410']
- **[ctr_filing → sar_filing]** 'A customer made a $100,000 wire transfer today — do we need to file a CTR?'  
  grounded=True, citations=['31 CFR 1010.311', '31 CFR 1020.315', '31 CFR 1020.320']
- **[info_sharing → recordkeeping]** 'Can we share underlying transaction records referenced in a SAR with a regulatory examiner?'  
  grounded=True, citations=['31 CFR 1010.520', '31 CFR 1020.320', '45 CFR 164.504']
- **[recordkeeping → sar_filing]** 'Are payment order records required to be retained for a domestic wire transfer?'  
  grounded=True, citations=['31 CFR 1010.370', '31 CFR 1010.410', '31 CFR 1020.410']
- **[sar_filing → ctr_filing]** 'Does a shell company conducting a $25,000 cash deposit that is linked to a high-risk jurisdiction require a SAR?'  
  grounded=True, citations=['31 CFR 1010.311', '31 CFR 1010.330', '31 CFR 1020.311']
- **[ctr_filing → sar_filing]** 'What is the CTR threshold for a wire transfer?'  
  grounded=True, citations=['31 CFR 1010.320', '31 CFR 1020.315', '31 CFR 1020.320']
- **[recordkeeping → sar_filing]** "What is the retention period for wire transfer originator information related to a payment order we received as a beneficiary's bank?"  
  grounded=True, citations=['31 CFR 1010.100', '31 CFR 1020.320', '31 CFR 1020.410']
- **[info_sharing → sar_filing]** 'Is there a safe harbor for sharing underlying transaction records referenced in a SAR with a self-regulatory organization?'  
  grounded=True, citations=['31 CFR 1010.520', '31 CFR 1010.540', '31 CFR 1020.320']
- **[recordkeeping → sar_filing]** 'What is the retention period for wire transfer originator information related to an international funds transfer?'  
  grounded=True, citations=['31 CFR 1010.410', '31 CFR 1010.420', '31 CFR 1020.320']
- **[sox_internal_controls → fda_e_records]** 'What is required of the audit committee regarding disclosure controls and procedures before a filing?'  
  grounded=True, citations=['17 CFR 240.10A-3', '17 CFR 240.13a-15', '21 CFR 11.1']
- **[ctr_filing → sar_filing]** 'A customer made a $9,500 currency exchange today — do we need to file a CTR?'  
  grounded=True, citations=['31 CFR 1010.311', '31 CFR 1020.315', '31 CFR 1020.320']
- **[ctr_filing → sar_filing]** 'A customer made a $25,000 wire transfer today — do we need to file a CTR?'  
  grounded=True, citations=['31 CFR 1010.311', '31 CFR 1020.315', '31 CFR 1020.320']
- **[ctr_filing → sar_filing]** 'What is the CTR threshold for a wire transfer?'  
  grounded=True, citations=['31 CFR 1010.320', '31 CFR 1020.315', '31 CFR 1020.320']
- **[info_sharing → sox_internal_controls]** 'Is there a safe harbor for sharing the existence of a law enforcement inquiry with a grand jury?'  
  grounded=True, citations=['31 CFR 1010.520', '31 CFR 1010.955', '45 CFR 164.512']
- **[sox_internal_controls → sar_filing]** 'What is required of the CEO regarding material weaknesses identified during the fiscal year before a filing?'  
  grounded=True, citations=['31 CFR 1010.540', '31 CFR 1020.315', '31 CFR 1020.320']
- **[recordkeeping → sar_filing]** 'Are funds transfer records required to be retained for a domestic wire transfer?'  
  grounded=True, citations=['31 CFR 1010.410', '31 CFR 1020.320', '31 CFR 1020.410']
- **[recordkeeping → sar_filing]** 'How long must we retain wire transfer originator information for a domestic wire transfer?'  
  grounded=True, citations=['31 CFR 1010.330', '31 CFR 1010.420', '31 CFR 1020.320']

...and 249 more (see results.json for the full list).

## Errored cases (1)

- 'A customer made a $12,000 cash deposit today — do we need to file a CTR?': An error occurred (ThrottlingException) when calling the InvokeModel operation (reached max retries: 4): Too many requests, please wait before trying again.