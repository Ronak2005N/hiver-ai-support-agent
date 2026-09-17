# Hiver AI Support Agent

An AI customer support agent for Amazon that classifies tweets, retrieves similar historical conversations, generates replies, and decides when to escalate to humans.

## What It Does

```
Customer message
    → Intent classifier (hybrid rules + ML)
    → Historical retrieval (find similar past conversations)
    → Reply generation (template-based)
    → Escalation decision (rules-based)
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run evaluation (reproduces headline results)
python evaluation/evaluate.py

# 3. Run interactive demo
python src/agent.py
```

## Interactive Demo

```bash
python src/agent.py
```

## Results

### Intent Classification

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **46.43%** |
| Macro Precision | 0.47 |
| Macro Recall | 0.49 |
| Macro F1 | 0.46 |

### Baselines

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Majority Class | 23.0% | 0.04 |
| TF-IDF + LR | ~21% | ~0.19 |
| **Ours (Hybrid)** | **46.43%** | **0.46** |

### Escalation (via evaluate_escalation.py)

| Metric | Value |
|--------|-------|
| Accuracy | 26.53% |
| Precision | 91.67% |
| Recall | 7.14% |
| F1 | 13.25% |

Note: The escalation rules are conservative — high precision but low recall. When the system escalates, it is almost always correct.

### Reply Quality

| Metric | Value |
|--------|-------|
| Template Relevance | 46.43% (matches predicted intent) |
| Greeting | 100% |
| Action-oriented | 100% |
| Concise | 100% |

## Evaluation

### Reproduce Results

```bash
# Full evaluation
python evaluation/evaluate.py

# Intent classification only
python evaluation/evaluate_intent.py

# Escalation only
python evaluation/evaluate_escalation.py

# Reply quality only
python evaluation/evaluate_replies.py

# LLM judge (requires GEMINI_API_KEY)
GEMINI_API_KEY=xxx python evaluation/llm_judge.py

# Failure analysis
python evaluation/failure_analysis.py
```

### Golden Set

- **Size:** 196 examples
- **Source:** Amazon @AmazonHelp tweets
- **Labels:** Human-verified (all 196 examples reviewed)
- **Location:** `evaluation/golden_set.json` and `data/golden/golden_set.csv`

### Metrics

- **Intent:** Accuracy, precision, recall, F1, confusion matrix
- **Escalation:** TP, TN, FP, FN, precision, recall, F1
- **Reply Quality:** Relevance, greeting, action-oriented, conciseness
- **LLM Judge:** 5-dimension scoring (relevance, empathy, actionability, tone, grounding)

## Golden Set Methodology

The golden set was built by:
1. Sampling 196 tweets from the 135K Amazon subset
2. Auto-labeling with keyword rules as initial labels
3. Human verification of all 196 labels via interactive verification tool
4. Final labels are human-verified ground truth

The human-verified labels differ significantly from auto-labeled labels (246 mismatches found across 196 examples), demonstrating the importance of human review.

## Failure Analysis

### Top 5 Failure Modes

1. **billing_payment → cancellation (11 times)** — "cancel" and "membership" keywords override billing context
2. **order_status → complaint_frustration (7 times)** — Frustration keywords override order-specific context
3. **order_status → cancellation (7 times)** — "cancel" appears in text, triggering wrong rules
4. **order_status → technical_support (7 times)** — "website"/"app" keywords match technical support
5. **refund_return → order_status (6 times)** — "package"/"shipping" keywords match order before refund

## Repository Structure

```
hiver-project/
├── README.md
├── requirements.txt
├── src/
│   ├── classifier.py
│   ├── hybrid_classifier.py
│   ├── retriever.py
│   ├── generator.py
│   ├── escalation.py
│   ├── pipeline.py
│   └── agent.py
├── evaluation/
│   ├── evaluate.py
│   ├── evaluate_intent.py
│   ├── evaluate_escalation.py
│   ├── evaluate_replies.py
│   ├── llm_judge.py
│   ├── human_review.py
│   ├── compute_agreement.py
│   ├── failure_analysis.py
│   ├── validate_submission.py
│   ├── sync_golden_set.py
│   ├── golden_set.json
│   └── results.json
├── data/
│   ├── golden/
│   │   ├── golden_set.csv
│   │   └── golden_candidate.csv
│   └── amazon_customers.csv
├── models/
│   └── classifier.pkl
└── docs/
    ├── REPORT.md
    ├── DECISION_LOG.md
    ├── FAILURE_ANALYSIS.md
    ├── SAMPLING_METHODOLOGY.md
    ├── BEGINNER_GUIDE.md
    └── CONTEXT.md
```

## Reproducibility

### What's Included

- **Golden Set:** 196 human-verified examples
- **Trained Model:** TF-IDF + LogisticRegression
- **Amazon Subset:** 135K tweets (filtered from 2.8M)
- **Evaluation Scripts:** All metrics can be reproduced

### To Reproduce

```bash
pip install -r requirements.txt
python evaluation/validate_submission.py
```

### Validate Submission

```bash
python evaluation/validate_submission.py
```

This runs all integrity checks on the golden set, metrics, and documentation.

## Limitations

1. **Template replies:** Not personalized to specific customer issues
2. **Low escalation recall:** Only 7.14% of escalation cases caught
3. **Intent confusion:** Overlapping keywords cause misclassification
4. **English only:** No multi-language support
5. **LLM judge agreement is low (Kappa=0.054):** The judge used keyword-based scoring (not actual LLM) due to API quota limits. Human scores show the keyword fallback gives systematically higher scores than human judgment, reflecting the fundamental limitation of rule-based scoring for nuanced quality assessment.

## Future Work

1. Human-labeled training data
2. Embedding-based retrieval (Sentence-BERT)
3. LLM-powered reply generation
4. Confidence-based escalation
5. Multi-language support
6. Real-time deployment

## License

This project is for educational purposes only.
