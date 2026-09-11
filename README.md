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

Example session:
```
Customer message: I was charged twice for my Prime membership

Intent:
  billing_payment

Confidence:
  95.00%

Retrieved evidence:
  1. @AmazonHelp I was charged twice for my Prime membership...
  2. @AmazonHelp Why was I charged again for Prime?...
  3. @AmazonHelp Double charge on my account...

Suggested reply:
  I see there's a billing concern. Let me look into this for you right away.

Decision:
  AUTO-HANDLE
```

## Architecture

### Components

| Component | File | Purpose |
|-----------|------|---------|
| Classifier | `src/classifier.py` | Intent classification (rules + ML) |
| Retriever | `src/retriever.py` | Historical conversation retrieval |
| Generator | `src/generator.py` | Reply generation (templates) |
| Escalation | `src/escalation.py` | Escalation decision rules |
| Pipeline | `src/pipeline.py` | Combines all components |
| Agent | `src/agent.py` | Interactive CLI agent |

### Intent Taxonomy (7 Classes)

| Intent | Description | Example |
|--------|-------------|---------|
| `order_status` | Package tracking, delivery issues | "Where is my order?" |
| `refund_return` | Refund requests, returns | "I want my money back" |
| `billing_payment` | Charges, payment problems | "I was charged twice" |
| `technical_support` | App/website issues | "The app keeps crashing" |
| `product_issue` | Damaged/wrong items | "Received broken product" |
| `cancellation` | Cancel subscriptions | "Cancel my Prime membership" |
| `complaint_frustration` | General anger, complaints | "Worst service ever!" |

## Results

### Intent Classification

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **70.41%** |
| Macro Precision | 0.73 |
| Macro Recall | 0.70 |
| Macro F1 | 0.71 |

### Baselines

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Majority Class | 14.3% | 0.04 |
| TF-IDF + LR | 21.4% | 0.19 |
| **Ours (Hybrid)** | **70.41%** | **0.71** |

### Escalation

| Metric | Value |
|--------|-------|
| Accuracy | 88.27% |
| Precision | 100.00% |
| Recall | 34.29% |
| F1 | 51.06% |

### LLM Judge

| Metric | Value |
|--------|-------|
| Average Score | 3.91/5.0 |
| Grade | B+ |
| Judge-Human Agreement | Cohen's Kappa = 0.68 |

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

# LLM judge
python evaluation/llm_judge.py

# Failure analysis
python evaluation/failure_analysis.py
```

### Golden Set

- **Size:** 196 examples (28 per intent)
- **Source:** Amazon @AmazonHelp tweets
- **Labels:** Auto-labeled by keyword rules
- **Location:** `evaluation/golden_set.json` and `data/golden/golden_set.csv`

### Metrics

- **Intent:** Accuracy, precision, recall, F1, confusion matrix
- **Escalation:** TP, TN, FP, FN, precision, recall, F1
- **Reply Quality:** Relevance, empathy, actionability, tone, grounding (1-5 scale)

## Failure Analysis

### Top 5 Failure Modes

1. **Product Issue → Refund Return (7 times)**
   - Both share "return" and "product" keywords
   - Fix: Add more specific product-damage keywords

2. **Refund Return → Order Status (4 times)**
   - Mentions both refund and order
   - Fix: Check for refund keywords before order keywords

3. **Technical Support → Complaint Frustration (4 times)**
   - Contains frustration keywords
   - Fix: Check for technical keywords first

4. **Order Status → Product Issue (3 times)**
   - Mentions both order and product
   - Fix: Add "received" to order_status rules

5. **Order Status → Complaint Frustration (3 times)**
   - Contains frustration keywords
   - Fix: Check for order-specific keywords first

## Repository Structure

```
hiver-project/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── src/                         # Source code
│   ├── classifier.py            # Intent classification
│   ├── retriever.py             # Historical retrieval
│   ├── generator.py             # Reply generation
│   ├── escalation.py            # Escalation decisions
│   ├── pipeline.py              # Main pipeline
│   ├── agent.py                 # Interactive CLI agent
│   ├── hybrid_classifier.py     # Legacy classifier (used by evaluation)
│   └── evaluate.py              # Legacy evaluation (used by old scripts)
├── evaluation/                  # Evaluation scripts
│   ├── evaluate.py              # Main evaluation
│   ├── evaluate_intent.py       # Intent evaluation
│   ├── evaluate_escalation.py   # Escalation evaluation
│   ├── evaluate_replies.py      # Reply evaluation
│   ├── llm_judge.py             # LLM-as-a-judge
│   ├── failure_analysis.py      # Failure analysis
│   ├── golden_set.json          # Golden Set (JSON)
│   └── results.json             # Evaluation results
├── data/                        # Data files
│   ├── golden/                  # Golden Set (CSV)
│   │   └── golden_set.csv
│   └── amazon_customers.csv     # Amazon tweets (135K)
├── models/                      # Trained models
│   └── classifier.pkl           # TF-IDF + LogisticRegression
├── docs/                        # Documentation
│   ├── REPORT.md                # 6-page report
│   ├── DECISION_LOG.md          # 15 design decisions
│   ├── FAILURE_ANALYSIS.md      # Top 5 failure modes
│   ├── SAMPLING_METHODOLOGY.md  # Golden Set methodology
│   └── CONTEXT.md               # Project context
└── .gitignore                   # Git ignore file
```

## Reproducibility

### What's Included

- **Golden Set:** 196 labeled examples
- **Trained Model:** TF-IDF + LogisticRegression
- **Amazon Subset:** 135K tweets (filtered from 2.8M)
- **Evaluation Scripts:** All metrics can be reproduced

### What's NOT Included

- **Full 2.8M tweet dataset:** Too large for submission
- **Gemini API key:** Required for LLM generation (optional)
- **Human labels:** Golden Set is auto-labeled

### To Reproduce

```bash
pip install -r requirements.txt
python evaluation/evaluate.py
```

Expected output:
```
Overall Accuracy: 70.41%
Macro F1: 0.71
Escalation Accuracy: 88.27%
```

## Limitations

1. **Auto-labeled Golden Set:** Labels may contain errors
2. **Template replies:** Not as good as LLM-generated replies
3. **Low escalation recall:** Only 34.29% of escalation cases caught
4. **Class imbalance:** complaint_frustration dominates training data
5. **English only:** No multi-language support

## Future Work

1. Human-labeled training data
2. Embedding-based retrieval (Sentence-BERT)
3. LLM-powered reply generation
4. Confidence-based escalation
5. Multi-language support
6. Real-time deployment

## License

This project is for educational purposes only.
