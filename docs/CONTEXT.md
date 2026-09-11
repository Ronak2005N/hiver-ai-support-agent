# Project Context

## Overview

This is an AI customer support agent for the Hiver SDE Intern take-home assignment. The agent classifies customer tweets and generates appropriate responses.

## Project Timeline

### Phase 1: Data Collection (Sept 10-11, 2026)
- Downloaded Kaggle dataset (2.8M tweets)
- Filtered Amazon tweets (135K)
- Extracted relevant fields (tweet_id, text, author_id, response_tweet_id, created_at)

### Phase 2: Taxonomy Design (Sept 11, 2026)
- Analyzed 50+ sample tweets
- Identified 7 intent classes
- Designed keyword-based rules
- User approved final taxonomy

### Phase 3: Classification (Sept 11, 2026)
- Built keyword regex rules (95% confidence)
- Trained TF-IDF LogisticRegression (71% on training data)
- Created hybrid classifier (69.9% on Golden Set)

### Phase 4: Evaluation (Sept 11, 2026)
- Created Golden Set (196 examples, 28 per intent)
- Built evaluation harness
- Computed per-class metrics
- Analyzed failure modes

### Phase 5: Documentation (Sept 11, 2026)
- Wrote 6-page report
- Documented 15 design decisions
- Analyzed top 5 failure modes
- Added LLM-as-judge evaluation

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `src/hybrid_classifier.py` | Core classifier | ✅ Working |
| `src/agent.py` | Full agent pipeline | ✅ Fixed (intent names) |
| `src/evaluate.py` | Evaluation harness | ✅ Working |
| `src/llm_judge.py` | Reply quality judge | ✅ Added |
| `src/audit.py` | Taxonomy audit | ✅ Working |
| `evaluation/golden_set.json` | Test examples | ✅ 196 examples |
| `docs/REPORT.md` | Project report | ✅ Complete |
| `docs/DECISION_LOG.md` | Design decisions | ✅ Complete |
| `docs/FAILURE_ANALYSIS.md` | Failure modes | ✅ Complete |
| `docs/SAMPLING_METHODOLOGY.md` | Golden Set methodology | ✅ Added |
| `docs/CONTEXT.md` | This file | ✅ Added |

## Design Decisions

1. **Hybrid Classifier:** Rules handle clear cases (95% confidence), ML handles ambiguous ones
2. **7 Intent Classes:** Derived from dataset analysis, not arbitrary
3. **Rule Priority:** More specific intents (cancellation) checked before general (order_status)
4. **Template Replies:** Deterministic, no API costs, easy to debug
5. **Rule-Based Escalation:** Based on legal keywords and frustration indicators

## Results

| Metric | Value |
|--------|-------|
| Intent Accuracy | 69.9% |
| Escalation Accuracy | 92.3% |
| Golden Set Size | 196 examples |
| Training Data | 135K Amazon tweets |

## Known Issues

1. **Gemini API:** Models keep returning 404 (deprecated model names)
2. **Auto-labeling:** Golden Set labels not manually verified
3. **Class Imbalance:** complaint_frustration dominates (54% of training data)

## Next Steps

1. Submit to anurag@hiverhq.com by Sept 17, 2026
2. Explain code live when asked
3. Consider human labeling for production use
