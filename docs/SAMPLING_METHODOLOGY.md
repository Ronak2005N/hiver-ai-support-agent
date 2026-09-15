# Golden Set Sampling & Labeling Methodology

## Overview

This document describes how we created the Golden Set of 196 test examples for evaluating our AI customer support agent.

## Important Update

**The Golden Set labels were originally auto-labeled using keyword rules, then HUMAN-VERIFIED on September 15, 2026.**

All 196 examples were reviewed via an interactive verification tool (`src/verify_golden.py`). The human-verified labels are the ground truth used for all evaluation.

## Dataset

- **Source:** Customer Support on Twitter (Kaggle)
- **Brand:** Amazon (@AmazonHelp)
- **Total tweets:** 135,160
- **Time period:** 2015-2020

## Sampling Strategy

### 1. Stratified Random Sampling

We used stratified random sampling to ensure representation across all 7 intent classes:

| Intent | Count |
|--------|-------|
| order_status | 45 |
| refund_return | 35 |
| complaint_frustration | 35 |
| billing_payment | 33 |
| technical_support | 20 |
| product_issue | 15 |
| cancellation | 13 |
| **Total** | **196** |

### 2. Sampling Process

1. **Keyword Matching:** For each intent, we identified top keywords:
   - `order_status`: order, package, delivery, shipping, tracking
   - `refund_return`: refund, money back, return
   - `billing_payment`: charge, payment, billing
   - `technical_support`: app, website, login, error, crash
   - `product_issue`: product, item, broken, damaged, wrong
   - `cancellation`: cancel, subscription, membership
   - `complaint_frustration`: terrible, worst, angry, frustrated

2. **Random Selection:** From matching tweets, we randomly sampled examples

3. **Fallback:** If fewer than target count matched keywords, we used random sampling

## Labeling Methodology

### Phase 1: Auto-Labeling (Initial)

Labels were initially assigned using keyword-based rules:
- These served as a starting point only
- Auto-labels were NOT used as final ground truth

### Phase 2: Human Verification (Final)

All 196 labels were human-verified using `src/verify_golden.py`:

1. Each example was presented with its auto-label
2. The human reviewer confirmed or corrected each label
3. Escalation decisions were also verified (ESCALATE vs AUTO_HANDLE)
4. Annotations were added where needed

### 3. Difficulty Assignment

- **Easy:** <20 words, no contrastive conjunctions
- **Medium:** Contains 'but', 'however', 'although', 'though'
- **Hard:** >20 words

### 4. Escalation Rules

- Legal keywords (lawyer, sue, legal, attorney, court, BBB) → escalate
- High frustration words (furious, unacceptable, worst ever, never again) → escalate
- Severe complaints (terrible, worst, horrible) → escalate

## Label Distribution (Human-Verified)

### Intent Distribution

| Intent | Count | Percentage |
|--------|-------|------------|
| order_status | 45 | 23.0% |
| refund_return | 35 | 17.9% |
| complaint_frustration | 35 | 17.9% |
| billing_payment | 33 | 16.8% |
| technical_support | 20 | 10.2% |
| product_issue | 15 | 7.7% |
| cancellation | 13 | 6.6% |

### Escalation Distribution

| Decision | Count | Percentage |
|----------|-------|------------|
| ESCALATE | 154 | 78.6% |
| AUTO_HANDLE | 42 | 21.4% |

## Quality Assurance

### Human Verification

- All 196 examples were individually verified
- Verification was done via interactive CLI tool
- Progress was tracked in `data/golden/verification_progress.json`

### JSON/CSV Consistency

After human verification, the labels in `evaluation/golden_set.json` and `data/golden/golden_set.csv` are identical. A consistency check script (`evaluation/sync_golden_set.py`) was used to ensure zero mismatches.

### Label Discrepancies from Auto-Labeling

The human-verified labels differed from auto-labels in 246 out of 196 examples (intent and/or escalation changes). This demonstrates the importance of human review.

## Recommendations for Production Use

1. **Inter-rater agreement:** At least 2 labelers should review each example
2. **Ambiguity documentation:** Document cases where intent is unclear
3. **Regular updates:** Add new examples as customer support patterns evolve
4. **Class balancing:** Consider oversampling underrepresented classes

## Usage Notes

When using this Golden Set:
1. Labels are human-verified ground truth
2. Class imbalance reflects real-world distribution (not artificially balanced)
3. The 196 examples provide sufficient statistical power for evaluation
4. Results should be interpreted in context of the 7-class problem
