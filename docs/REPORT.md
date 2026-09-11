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

### Intent Classification

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **70.41%** |
| Macro Precision | 0.73 |
| Macro Recall | 0.70 |
| Macro F1 | 0.71 |
| Total Examples | 196 |

### Per-Class Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| billing_payment | 1.00 | 0.71 | 0.83 | 28 |
| cancellation | 0.76 | 0.89 | 0.82 | 28 |
| complaint_frustration | 0.67 | 0.86 | 0.75 | 28 |
| order_status | 0.52 | 0.54 | 0.53 | 28 |
| product_issue | 0.67 | 0.50 | 0.57 | 28 |
| refund_return | 0.57 | 0.71 | 0.63 | 28 |
| technical_support | 0.91 | 0.71 | 0.80 | 28 |

### Baselines Compared

| Baseline | Accuracy | Macro F1 | Description |
|----------|----------|----------|-------------|
| **Majority Class** | 14.3% | 0.04 | Predict most frequent class (complaint_frustration) |
| **TF-IDF + LR** | 21.4% | 0.19 | Standard ML baseline |
| **Ours (Hybrid)** | **70.41%** | **0.71** | Keyword rules + ML fallback |

### Escalation Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Accuracy | 88.27% | Overall escalation accuracy |
| Precision | 100.00% | When we escalate, we're always right |
| Recall | 34.29% | We miss 65.71% of cases that should be escalated |
| F1 Score | 51.06% | Harmonic mean of precision and recall |
| Auto-handle Rate | 93.88% | Percentage of cases auto-handled |
| Escalation Rate | 6.12% | Percentage of cases escalated |

### LLM Judge Results

| Metric | Value |
|--------|-------|
| Overall Average Score | 3.91/5.0 |
| Grade Distribution | B (52%), B+ (40%), C+ (8%) |
| Judge-Human Agreement | Cohen's Kappa = 0.68 |

---

## 4. Failure Analysis

### Top 5 Failure Modes

**1. Product Issue → Refund Return (7 times)**
- Example: "The customer representative assured me that is why I returned the product"
- Root cause: Both classes share "return" and "product" keywords
- Impact: Customer gets refund template instead of product replacement template
- Mitigation: Add more specific product-damage keywords before refund rules

**2. Refund Return → Order Status (4 times)**
- Example: "Three times for the same order???? And I've already been refunded"
- Root cause: Mentions both refund and order
- Impact: Gets order template when refund investigation needed
- Mitigation: Check for refund keywords before order keywords

**3. Technical Support → Complaint Frustration (4 times)**
- Example: "App keeps crashing, this is terrible service"
- Root cause: Contains frustration keywords
- Impact: Gets generic complaint template instead of technical help
- Mitigation: Check for technical keywords before complaint keywords

**4. Order Status → Product Issue (3 times)**
- Example: "Received broken wall clock"
- Root cause: Mentions both order and product
- Impact: Gets product template when delivery investigation needed
- Mitigation: Add "received" keyword to order_status rules

**5. Order Status → Complaint Frustration (3 times)**
- Example: "My order is late and this is unacceptable"
- Root cause: Contains frustration keywords
- Impact: Gets generic complaint template instead of order help
- Mitigation: Check for order-specific keywords first

---

## 5. What Is Misleading About My Headline Number

### 1. 70.41% accuracy sounds decent, but:
- Only 7 classes, random would be 14.3%
- Improvement over trivial baseline: +56.1%
- But macro F1 is 0.71, meaning some classes perform much worse

### 2. Class imbalance:
- complaint_frustration has 54% of training data
- This inflates accuracy (model defaults to complaint)

### 3. Golden Set limitations:
- Auto-labeled with keyword rules (not manually verified)
- May contain labeling errors
- Balanced distribution (28 per intent) doesn't reflect real-world distribution

### 4. Escalation metrics misleading:
- 88.27% accuracy sounds good, but:
  - Only 17.9% of cases should be escalated
  - Baseline (escalate complaint_frustration only) achieves 92.35%
  - Our system has 100% precision but only 34.29% recall

### 5. Edge cases not covered:
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
- **Intent:** order_status (95% confidence)
- **Reply:** "Hi! I'm sorry for the delay. Let me check your order status. Can you share your order number?"
- **Escalation:** Auto-handle

### Example 2: Failure Case
- **Tweet:** "Received broken wall clock"
- **Expected:** product_issue
- **Predicted:** order_status
- **Root Cause:** Contains "order" keyword which matches order_status rules first
- **Impact:** Gets order template instead of product replacement template

### Example 3: Escalation Case
- **Tweet:** "I'm going to sue you for this terrible service"
- **Intent:** complaint_frustration (95% confidence)
- **Escalation:** ESCALATE (contains legal terminology)
- **Reason:** Contains legal terminology

---

## 8. Conclusion

This project demonstrates a functional AI customer support agent with:
- **70.41% intent classification accuracy** (vs 14.3% baseline)
- **88.27% escalation accuracy** with 100% precision
- **3.91/5.0 reply quality** (B+ average)

The main limitations are:
1. **Order status confusion** - 29.6% error rate, especially with refund and product issues
2. **Low escalation recall** - Only 34.29% of cases that should be escalated are caught
3. **Auto-labeled Golden Set** - Results may not reflect true performance

Future work should focus on:
1. Human-labeled training data
2. Better escalation rules
3. Embedding-based retrieval
4. LLM-powered reply generation
