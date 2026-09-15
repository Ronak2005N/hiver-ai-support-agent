# Project Context

## Overview

This is an AI customer support agent for the Hiver SDE Intern take-home assignment. The agent classifies customer tweets and generates appropriate responses.

**Status: FINAL** — All metrics computed from human-verified golden set.

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
- Trained TF-IDF LogisticRegression
- Created hybrid classifier

### Phase 4: Golden Set & Evaluation (Sept 11-15, 2026)
- Created Golden Set (196 examples)
- **Human verification of all 196 labels** (Sept 15, 2026)
- Built evaluation harness
- Computed per-class metrics
- Analyzed failure modes

### Phase 5: Documentation (Sept 11-15, 2026)
- Wrote 6-page report
- Documented 15 design decisions
- Analyzed top 5 failure modes
- Added LLM-as-judge evaluation

## Key Files

| File | Purpose | Status |
|------|---------|--------|
| `src/hybrid_classifier.py` | Core classifier | Working |
| `src/agent.py` | Full agent pipeline | Working |
| `evaluation/evaluate.py` | Evaluation harness | Working |
| `evaluation/llm_judge.py` | LLM-as-judge (Gemini) | Working |
| `evaluation/human_review.py` | Human scoring workflow | Ready (pending human scores) |
| `evaluation/compute_agreement.py` | Cohen's Kappa computation | Ready (pending human scores) |
| `evaluation/validate_submission.py` | Submission integrity check | Working |
| `evaluation/sync_golden_set.py` | Sync CSV→JSON labels | Working |
| `evaluation/golden_set.json` | Test examples (human-verified) | 196 examples |
| `docs/REPORT.md` | Project report | Complete |
| `docs/DECISION_LOG.md` | Design decisions | Complete |
| `docs/FAILURE_ANALYSIS.md` | Failure modes | Complete |
| `docs/SAMPLING_METHODOLOGY.md` | Golden Set methodology | Complete |

## Final Results

| Metric | Value |
|--------|-------|
| Intent Accuracy | **46.43%** |
| Macro F1 | **0.46** |
| Escalation Precision | **91.67%** |
| Escalation Recall | **7.14%** |
| Golden Set Size | 196 examples |
| Golden Set Labels | Human-verified |
| Training Data | 135K Amazon tweets |

## Key Lesson

The auto-labeled golden set produced misleadingly optimistic results (70.41% accuracy). After human verification of all 196 labels, the true accuracy is 46.43%. This demonstrates why human-labeled evaluation sets are critical.

## Known Limitations

1. Template replies (not LLM-generated)
2. Low escalation recall (7.14%)
3. Intent confusion with overlapping keywords
4. No multi-language support
5. LLM judge human agreement pending

## Next Steps

1. Submit to anurag@hiverhq.com by Sept 17, 2026
2. Explain code live when asked
3. Consider human labeling for production use
