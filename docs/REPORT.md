# Hiver AI Support Agent Report
## AI Customer Support Agent for Amazon

**Student:** Ronak  
**Date:** September 2026  
**Brand:** Amazon (@AmazonHelp)  
**Dataset:** Customer Support on Twitter (Kaggle)

---

## 1. Problem Framing

### What "Good" Means for This Brand

For Amazon customer support, "good" means:
1. **Correct intent classification** - Understanding what the customer wants (refund, order status, technical issue, etc.)
2. **Appropriate escalation** - Knowing when to auto-handle vs escalate to human
3. **Grounded replies** - Responses based on how Amazon has historically resolved similar issues
4. **Speed** - Fast classification and response generation

### What I Chose NOT to Build

1. **Full RAG system** - Would require embeddings + vector DB, which is overkill for this project scope
2. **LLM fine-tuning** - Not enough clean training data, and template replies are sufficient for demo
3. **Multi-turn conversation** - Focused on single-turn classification + reply
4. **Real-time deployment** - Built for offline evaluation, not production
5. **Web application** - Assignment requires CLI agent and evaluation, not deployment

---

## 2. Implementation

### Architecture

```
Customer Tweet
     ↓
┌─────────────────┐
│ Intent          │ ← Keyword rules (high confidence) + TF-IDF ML (fallback)
│ Classifier      │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Historical      │ ← Find similar past conversations
│ Retriever       │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Reply           │ ← Template-based (deterministic, reproducible)
│ Generator       │
└────────┬────────┘
         ↓
┌─────────────────┐
│ Escalation      │ ← Rules-based: legal keywords, high frustration
│ Decision        │
└─────────────────┘
```

### Key Design Decisions

1. **Hybrid Classifier** - Combined keyword rules (95% confidence) with TF-IDF ML model (fallback)
2. **Rule Priority** - More specific intents (technical_support) checked first, general (complaint_frustration) last
3. **Template Replies** - Deterministic, no API costs, easy to debug
4. **Rule-Based Escalation** - Based on legal keywords and frustration indicators

---

## 3. Results

### Golden Set

| Property | Value |
|----------|-------|
| Total Examples | 196 |
| Source | Amazon @AmazonHelp tweets from Kaggle |
| Labeling Method | Human-verified (reviewed via interactive verification tool) |
| Labeler | Human (Ronak), verified all 196 examples |
| Per-Intent Distribution | 13-45 examples (imbalanced, reflects real-world) |

### Intent Classification

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **46.43%** |
| Macro Precision | 0.47 |
| Macro Recall | 0.49 |
| Macro F1 | 0.46 |
| Total Examples | 196 |
| Total Errors | 91 |

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| billing_payment | 0.70 | 0.42 | 0.53 | 33 |
| cancellation | 0.24 | 0.62 | 0.35 | 13 |
| complaint_frustration | 0.44 | 0.46 | 0.45 | 35 |
| order_status | 0.48 | 0.31 | 0.38 | 45 |
| product_issue | 0.33 | 0.47 | 0.39 | 15 |
| refund_return | 0.60 | 0.60 | 0.60 | 35 |
| technical_support | 0.50 | 0.55 | 0.52 | 20 |

### Baselines Compared

| Baseline | Accuracy | Macro F1 | Description |
|----------|----------|----------|-------------|
| **Majority Class** | 23.0% | 0.04 | Predict most frequent class (order_status) |
| **TF-IDF + LR** | ~21% | ~0.19 | Standard ML baseline (estimated) |
| **Ours (Hybrid)** | **46.43%** | **0.46** | Keyword rules + ML fallback |

### Escalation Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Accuracy | 26.53% | Overall escalation accuracy |
| Precision | 91.67% | When we escalate, we're almost always right |
| Recall | 7.14% | We miss 92.86% of cases that should be escalated |
| F1 Score | 13.25% | Harmonic mean of precision and recall |
| Auto-handle Rate | 93.88% | Percentage of cases auto-handled |
| Escalation Rate | 6.12% | Percentage of cases escalated |

### Escalation Evaluation Note

The escalation evaluation uses two methods:
1. **evaluate.py**: Uses intent prediction as escalation proxy (accuracy: 26.53%, precision: 63.89%, recall: 14.94%, F1: 24.21%)
2. **evaluate_escalation.py**: Uses actual EscalationDecider rules (accuracy: 26.53%, precision: 91.67%, recall: 7.14%, F1: 13.25%)

The rule-based method (2) has higher precision but much lower recall. The intent-proxy method (1) is a weaker escalation signal.

### Reply Quality (LLM Judge)

| Metric | Value |
|--------|-------|
| Relevance (intent matches) | 46.43% |
| Greeting present | 100.00% |
| Action present | 100.00% |
| Concise (<30 words) | 100.00% |

Note: Reply quality is currently template-based. All templates are concise, professional, and action-oriented. The 46.43% relevance reflects intent classification accuracy (replies match predicted intent, not ground truth).

### LLM-as-Judge

| Metric | Value |
|--------|-------|
| Judge Type | Google Gemini API (actual LLM) |
| Fallback | Keyword-based (if no API key) |
| Subset Evaluated | 196/196 examples |
| Human-Reviewed Subset | 0 (workflow created, pending human scoring) |
| Cohen's Kappa | Pending human review |

The LLM judge calls Google Gemini to score replies on 5 dimensions: relevance, empathy, actionability, tone, grounding.

---

## 4. Failure Analysis

### Top 5 Failure Modes

**1. billing_payment → cancellation (11 times)**
- Example: "Can you explain why I've been charged AGAIN for a prime membership I don't have?"
- Root cause: Keywords like "cancel" and "membership" trigger cancellation rules before billing rules
- Impact: Gets cancellation template instead of billing investigation template
- Mitigation: Check for billing-specific keywords (charged, payment) before cancellation keywords

**2. order_status → complaint_frustration (7 times)**
- Example: "I have still not got the exact location and the confirmed delivery date"
- Root cause: Frustration keywords (terrible, worst) override order-specific keywords
- Impact: Gets generic complaint template instead of order tracking help
- Mitigation: Check for order-specific keywords (package, delivery, tracking) before complaint keywords

**3. order_status → cancellation (7 times)**
- Example: "I ordered from ur website but I couldn't find it in my orders"
- Root cause: "cancel" appears in text, triggering cancellation rules
- Impact: Gets cancellation template when user wants order status
- Mitigation: Weight order keywords higher when delivery/tracking context present

**4. order_status → technical_support (7 times)**
- Example: "Yeah on Friday when I ordered it however it's now showing as not available"
- Root cause: "website" and "app" keywords trigger technical support rules
- Impact: Gets technical support template when order investigation needed
- Mitigation: Check for order context before technical support rules

**5. refund_return → order_status (6 times)**
- Example: "I was just notified that my package was lost in shipping. How Can I get a refund?"
- Root cause: "package" and "shipping" keywords match order_status before refund_return
- Impact: Gets order template when refund investigation needed
- Mitigation: Check for refund keywords first when both refund and order context present

### Errors by Difficulty

| Difficulty | Count | Percentage |
|------------|-------|------------|
| Hard | 83 | 79.0% |
| Easy | 20 | 19.0% |
| Medium | 2 | 1.9% |

---

## 5. What Is Misleading About My Headline Number

### 1. The accuracy was previously inflated (70.41% → 46.43%)

The previous 70.41% accuracy was computed on an **auto-labeled** golden set where labels were generated by the same keyword rules used by the classifier. This created a circular evaluation - the test set labels came from a system similar to the system being tested.

After human verification of all 196 labels, the true accuracy is **46.43%**. The auto-labeling inflated accuracy by ~24 percentage points.

### 2. 46.43% accuracy is still better than random

- Random baseline: 14.3% (1/7 classes)
- Majority class: 23.0% (order_status is most common)
- Our improvement over random: +32.1%
- Our improvement over majority class: +23.4%

### 3. Class imbalance affects interpretation

- order_status has 45 examples (23.0%) - largest class
- cancellation has only 13 examples (6.6%) - smallest class
- The classifier performs poorly on cancellation (F1=0.35) partly due to fewer examples

### 4. Escalation metrics are misleading

- 26.53% accuracy sounds low, but:
  - Only 78.6% of cases should be escalated (154/196)
  - The escalation rules are too conservative (only 6.12% escalation rate)
  - Precision is 91.67% - when we escalate, we're almost always right
  - The problem is we almost never escalate (7.14% recall)

### 5. Template replies are limited

- Only 7 unique templates (one per intent)
- No personalization based on tweet content
- Reply quality is deterministic, not adaptive
- The "grounded" claim is weak - templates don't use retrieved examples

### 6. Edge cases not covered

- Multi-language tweets (French, Spanish, German)
- Very short tweets (<5 words)
- Tweets with links only
- Tweets with multiple intents

---

## 6. One-Week-Next-Step Plan

### Week 1 Priorities

1. **Improve Training Data:**
   - Collect 500+ human-labeled examples
   - Balance class distribution
   - Add multi-language support

2. **Better Retrieval:**
   - Use sentence embeddings (Sentence-BERT)
   - Build vector database for fast similarity search
   - Filter by intent before retrieval

3. **Better Escalation:**
   - Confidence-based escalation (low confidence → escalate)
   - Multi-factor scoring (keywords + confidence + sentiment)
   - Target 60%+ recall while maintaining 90%+ precision

4. **Better Reply Generation:**
   - Use LLM (Gemini/GPT) for reply generation
   - Few-shot prompting with similar examples
   - Intent-specific prompts

5. **Production Features:**
   - Real-time API endpoint
   - Logging and monitoring
   - A/B testing framework

---

## 7. Real Examples

### Example 1: Successful Classification
- **Tweet:** "Where is my order? It's been 2 weeks"
- **Intent:** order_status
- **Reply:** "Hi! I'm sorry for the delay. Let me check your order status. Can you share your order number?"
- **Escalation:** Auto-handle

### Example 2: Failure Case
- **Tweet:** "Can you explain why I've been charged AGAIN for a prime membership I don't have?"
- **Expected:** billing_payment
- **Predicted:** cancellation
- **Root Cause:** "cancel" and "membership" keywords trigger cancellation rules first
- **Impact:** Gets cancellation template instead of billing investigation template

### Example 3: Escalation Case
- **Tweet:** "I'm going to sue you for this terrible service"
- **Intent:** complaint_frustration
- **Escalation:** ESCALATE (contains legal terminology)
- **Reason:** Contains legal terminology

---

## 8. Conclusion

This project demonstrates a functional AI customer support agent with:
- **46.43% intent classification accuracy** (vs 23.0% majority class baseline)
- **91.67% escalation precision** (when we escalate, we're right)
- **Template-based replies** that are concise, professional, and action-oriented

The main limitations are:
1. **Intent classification accuracy** - 53.6% error rate, especially with overlapping keywords
2. **Low escalation recall** - Only 7.14% of cases that should be escalated are caught
3. **Template replies** - Not personalized to specific customer issues

### Key Lesson

The most important finding is that **auto-labeled evaluation data produces misleadingly optimistic results**. The previous 70.41% accuracy was inflated by ~24 points because the test labels came from the same system being evaluated. After human verification of all 196 labels, the true accuracy is 46.43%. This is still meaningful (3x better than random), but demonstrates why human-labeled evaluation sets are critical.

Future work should focus on:
1. Human-labeled training data
2. Better escalation rules
3. Embedding-based retrieval
4. LLM-powered reply generation
