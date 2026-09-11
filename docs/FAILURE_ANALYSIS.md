# Failure Analysis
## Top 5 Failure Modes with Examples and Hypotheses

### Failure Mode 1: Product Issue → Refund Return
**Count:** 7 times (most common error)

**Example:**
- Tweet: "I received the wrong item in my order, I want to return it"
- Actual: product_issue
- Predicted: refund_return

**Root Cause Hypothesis:**
Both classes share keywords: "return", "item", "wrong". The refund_return rules match first because they're more specific (checked before product_issue in priority order).

**Impact:**
Customer gets refund template instead of product replacement template. Wrong first response.

**Potential Fix:**
Add product-specific keywords before refund in priority order: "received wrong", "damaged item", "broken product"

---

### Failure Mode 2: Order Status → Refund Return
**Count:** 4 times

**Example:**
- Tweet: "My order never arrived, I want my money back"
- Actual: order_status
- Predicted: refund_return

**Root Cause Hypothesis:**
Tweet mentions both "order" and "money back". Refund rules match first.

**Impact:**
Customer gets refund template when order investigation might be more appropriate.

**Potential Fix:**
Check for "never arrived", "not received" before refund keywords.

---

### Failure Mode 3: Billing Payment → Cancellation
**Count:** 4 times

**Example:**
- Tweet: "I was charged for Prime but want to cancel"
- Actual: billing_payment
- Predicted: cancellation

**Root Cause Hypothesis:**
Tweet mentions both "charged" and "cancel". Cancellation rules match first.

**Impact:**
Customer gets cancellation template when billing dispute might be more appropriate.

**Potential Fix:**
Check for charge amount keywords ("charged twice", "wrong amount") before cancellation.

---

### Failure Mode 4: Order Status → Product Issue
**Count:** 3 times

**Example:**
- Tweet: "My order arrived but the item is broken"
- Actual: order_status
- Predicted: product_issue

**Root Cause Hypothesis:**
Tweet mentions both "order" and "item/broken". Product rules match.

**Impact:**
Customer gets product template when delivery investigation might be needed.

**Potential Fix:**
Add "arrived" to order_status rules to distinguish from product issues.

---

### Failure Mode 5: Order Status → Complaint Frustration
**Count:** 3 times

**Example:**
- Tweet: "My order is late and this is unacceptable"
- Actual: order_status
- Predicted: complaint_frustration

**Root Cause Hypothesis:**
Tweet contains frustration keywords ("unacceptable"). Complaint rules match.

**Impact:**
Customer gets generic complaint template instead of order investigation.

**Potential Fix:**
Check for order-specific keywords before complaint keywords.

---

## Overall Failure Statistics

| Metric | Value |
|--------|-------|
| Total examples | 196 |
| Total failures | 59 (30.1%) |
| Most common error | product_issue → refund_return (7 times) |
| Least common errors | Unique pairs (1-2 times each) |

## What Is Misleading About Headline Number

1. **69.9% accuracy sounds decent, but:**
   - Only 7 classes, random would be 14.3%
   - Improvement over trivial baseline: +55.6%
   - But macro F1 is 0.70, meaning some classes perform much worse

2. **Class imbalance:**
   - complaint_frustration has 54% of training data
   - This inflates accuracy (model defaults to complaint)

3. **Golden Set limitations:**
   - Auto-labeled with keyword rules
   - No human verification
   - May contain labeling errors

4. **Edge cases not covered:**
   - Multi-language tweets (French, Spanish, German)
   - Very short tweets (<5 words)
   - Tweets with links only

## Recommendations for Improvement

1. **Better rule priority order** - Check more specific patterns first
2. **More training data** - Especially for underrepresented classes
3. **Human labeling** - Verify Golden Set labels
4. **Embedding-based retrieval** - Better similarity search
5. **LLM for ambiguous cases** - Use Gemini/GPT for low-confidence predictions
