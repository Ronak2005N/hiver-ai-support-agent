# Golden Set Sampling & Labeling Methodology

## Overview

This document describes how we created the Golden Set of 196 test examples for evaluating our AI customer support agent.

## Important Limitations

**This Golden Set was auto-labeled using keyword rules, NOT manually labeled by a human.**

This is a significant limitation that affects:
1. Label accuracy (some examples may be mislabeled)
2. Evaluation reliability (results may not reflect true performance)
3. Generalizability (auto-labels may not match human judgment)

## Dataset

- **Source:** Customer Support on Twitter (Kaggle)
- **Brand:** Amazon (@AmazonHelp)
- **Total tweets:** 135,160
- **Time period:** 2015-2020

## Sampling Strategy

### 1. Stratified Random Sampling

We used stratified random sampling to ensure balanced representation across all 7 intent classes:

| Intent | Target Count | Actual Count |
|--------|--------------|--------------|
| order_status | 28 | 28 |
| refund_return | 28 | 28 |
| billing_payment | 28 | 28 |
| technical_support | 28 | 28 |
| product_issue | 28 | 28 |
| cancellation | 28 | 28 |
| complaint_frustration | 28 | 28 |
| **Total** | **196** | **196** |

### 2. Sampling Process

1. **Keyword Matching:** For each intent, we identified top 3-6 keywords:
   - `order_status`: order, package, delivery, shipping, tracking, where is
   - `refund_return`: refund, money back, return
   - `billing_payment`: charge, payment, billing, price
   - `technical_support`: app, website, login, error, crash, account
   - `product_issue`: product, item, broken, damaged, wrong
   - `cancellation`: cancel, subscription, membership, prime
   - `complaint_frustration`: terrible, worst, angry, frustrated, unacceptable

2. **Random Selection:** From matching tweets, we randomly sampled 28 per intent

3. **Fallback:** If fewer than 28 tweets matched keywords, we used random sampling from the full dataset

## Labeling Methodology

### 1. Rule-Based Labeling (NOT Manual)

All labels were assigned using keyword-based rules:

```python
# Example: refund_return detection
if any(word in text for word in ['refund', 'money back', 'return']):
    intent = 'refund_return'
```

**This is NOT manual labeling.** The labels are algorithmically assigned.

### 2. Difficulty Assignment

- **Easy:** <20 words, no contrastive conjunctions
- **Medium:** Contains 'but', 'however', 'although', 'though'
- **Hard:** >20 words

### 3. Escalation Rules

- Legal keywords (lawyer, sue, legal, attorney, court, BBB) → escalate
- High frustration words (furious, unacceptable, worst ever, never again) → escalate
- Severe complaints (terrible, worst, horrible) → escalate

## Known Limitations

1. **Auto-labeling:** Labels were NOT manually verified
2. **Keyword bias:** Tweets must contain specific keywords to be labeled correctly
3. **Class imbalance:** complaint_frustration dominates the full dataset (54%)
4. **Language:** Some tweets are in non-English languages
5. **Ambiguity:** Some tweets have multiple valid intents
6. **No human review:** Labels have not been verified by a human

## Quality Assurance Issues

### Inter-Rater Agreement (NOT Measured)

Since labels are auto-generated, we cannot measure inter-rater agreement.

### Potential Label Errors

Some examples may be mislabeled:
- "Received broken wall clock" labeled as `order_status` but could be `product_issue`
- "Package lost, want refund" labeled as `order_status` but could be `refund_return`

## Recommendations for Production Use

1. **Human labeling:** All labels should be manually verified
2. **Inter-rater agreement:** At least 2 labelers should review each example
3. **Ambiguity documentation:** Document cases where intent is unclear
4. **Regular updates:** Add new examples as customer support patterns evolve

## Usage Notes

When using this Golden Set:
1. Results should be interpreted with caution due to auto-labeling
2. Consider expanding with human-labeled examples for production use
3. Class imbalance may affect per-class metrics
4. The balanced distribution (28 per intent) may not reflect real-world distribution
