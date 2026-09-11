# Decision Log
## Non-Obvious Decisions and Rationale

### 1. Why this Amazon subset?

**Decision:** Use Amazon (@AmazonHelp) tweets from the Kaggle Customer Support on Twitter dataset.

**Reason:** Amazon has the most tweets (135K) with diverse issue types, providing enough data for classification and retrieval.

**Alternative considered:** Use Apple (@AppleSupport) or other brands.

**Trade-off:** Amazon tweets are more diverse but also more noisy. Apple tweets might be cleaner but less data.

---

### 2. Why these 7 intents?

**Decision:** Use 7 intent classes: order_status, refund_return, billing_payment, technical_support, product_issue, cancellation, complaint_frustration.

**Reason:** Derived from analyzing 50+ sample tweets. These categories cover the majority of customer support issues.

**Alternative considered:** Use fewer (3-4) or more (10+) classes.

**Trade-off:** Fewer classes would be easier but less nuanced. More classes would be more accurate but harder to label and evaluate.

---

### 3. Why 196 Golden examples?

**Decision:** Use 196 examples in the Golden Set (28 per intent).

**Reason:** Balanced representation across all 7 intents. 28 examples per intent provides enough for evaluation while being manageable.

**Alternative considered:** Use 100 or 300 examples.

**Trade-off:** More examples would be more reliable but harder to create. Fewer examples would be faster but less statistically significant.

---

### 4. Why hybrid classification?

**Decision:** Combine keyword rules (95% confidence) with TF-IDF ML model (fallback).

**Reason:** Rules handle clear cases quickly and accurately. ML handles ambiguous cases where rules fail.

**Alternative considered:** Use only rules or only ML.

**Trade-off:** Rules are faster but less flexible. ML is more flexible but slower and less explainable.

---

### 5. Why retrieval?

**Decision:** Include historical retrieval to find similar past conversations.

**Reason:** Provides evidence for reply generation and helps ground responses in actual support history.

**Alternative considered:** Skip retrieval and use only templates.

**Trade-off:** Retrieval adds complexity but improves reply quality. Templates are simpler but less contextual.

---

### 6. Why templates initially?

**Decision:** Use template replies instead of LLM generation.

**Reason:** Templates are deterministic, reproducible, and don't require API keys. LLM generation would be better but adds dependency.

**Alternative considered:** Use Gemini or other LLM for generation.

**Trade-off:** Templates are reliable but generic. LLMs are better but less predictable and require API access.

---

### 7. Why confidence threshold?

**Decision:** Use 0.5 confidence threshold for escalation decisions.

**Reason:** Below 0.5 confidence indicates the classifier is uncertain, which may indicate a complex case needing human review.

**Alternative considered:** Use 0.3 or 0.7 threshold.

**Trade-off:** Lower threshold escalates fewer cases (may miss issues). Higher threshold escalates more cases (may waste human time).

---

### 8. Why escalation policy?

**Decision:** Escalate based on legal keywords, high frustration, and low confidence.

**Reason:** These are clear indicators of cases that need human intervention.

**Alternative considered:** Escalate all complaint_frustration cases.

**Trade-off:** Current policy is more selective but may miss some cases. Escalating all complaints would catch more but waste human time.

---

### 9. Why these evaluation metrics?

**Decision:** Report accuracy, precision, recall, F1, and confusion matrix.

**Reason:** These are standard classification metrics that provide comprehensive view of performance.

**Alternative considered:** Use only accuracy.

**Trade-off:** Accuracy alone is misleading (especially with imbalanced classes). Multiple metrics provide better picture but are more complex.

---

### 10. Why not process all 3M tweets?

**Decision:** Filter to Amazon subset (135K tweets) instead of using all 2.8M tweets.

**Reason:** Processing all tweets would be slow and most are irrelevant. Amazon subset provides enough data for evaluation.

**Alternative considered:** Process all tweets.

**Trade-off:** Using subset is faster but may miss edge cases. Processing all tweets would be comprehensive but slow.

---

### 11. Why not build a web application?

**Decision:** Focus on CLI agent and evaluation instead of web UI.

**Reason:** Assignment requires runnable pipeline and evaluation, not deployment. CLI is sufficient for demonstration.

**Alternative considered:** Build Streamlit or Flask web app.

**Trade-off:** CLI is simpler and faster to build. Web app would be more user-friendly but takes more time.

---

### 12. Why use a separate Golden Set?

**Decision:** Create separate Golden Set instead of using train/test split.

**Reason:** Golden Set provides controlled, balanced evaluation. Train/test split may be biased by data ordering.

**Alternative considered:** Use random train/test split.

**Trade-off:** Golden Set is more controlled but requires manual creation. Train/test split is faster but less reliable.

---

### 13. Why use an LLM judge?

**Decision:** Implement LLM-as-a-judge for reply quality evaluation.

**Reason:** Assignment requires "an LLM-as-judge rubric for reply quality" with human agreement evidence.

**Alternative considered:** Skip LLM judge and only evaluate intent classification.

**Trade-off:** LLM judge provides richer evaluation but adds complexity. Intent-only evaluation is simpler but less comprehensive.

---

### 14. How judge-human agreement was measured?

**Decision:** Use 30 manually scored examples with Cohen's Kappa.

**Reason:** Provides statistical measure of agreement between automated judge and human scoring.

**Alternative considered:** Use correlation only.

**Trade-off:** Kappa is more robust than correlation but requires manual scoring. Correlation is simpler but less meaningful.

---

### 15. Why not tune on the test set?

**Decision:** Do not adjust classifier rules to improve Golden Set score.

**Reason:** Tuning on test set would be data leakage and produce misleading results.

**Alternative considered:** Adjust rules to fix specific errors.

**Trade-off:** Not tuning produces honest results. Tuning would improve score but invalidate evaluation.
