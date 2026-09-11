"""
HYBRID CLASSIFIER (v2 - Fixed Priority Order)
Combines keyword rules + TF-IDF model for better accuracy.

Key fix: reordered priorities so that specific intents (product_issue, order_status)
are checked BEFORE generic intents (refund_return, cancellation).

Usage:
    python src/hybrid_classifier.py
"""

import re
import pickle
import os


# REVISED TAXONOMY KEYWORD RULES (v2 - Fixed priority order)
# Priority order: most specific first, generic last
INTENT_RULES = [
    # 1. TECHNICAL_SUPPORT - most specific (app/website/account issues)
    ('technical_support', [
        r'(?:app|website|site)\s*(?:crash|bug|error|not\s*work|broken|down|issue)',
        r'(?:can(?:no|\')?t|unable\s*to|struggling\s*to)\s*(?:log\s*in|login|sign\s*in|access)',
        r'(?:login|log\s*in|sign\s*in|password|account)\s*(?:issue|problem|error|not\s*work)',
        r'(?:error|bug|glitch|crash|freeze)\s*(?:message|code|screen)',
        r'(?:account|email)\s*(?:changed|hacked|compromised|locked|disabled)',
    ]),

    # 2. PRODUCT_ISSUE - wrong/damaged product (check before refund/order)
    ('product_issue', [
        r'(?:wrong|incorrect|fake|defective|damaged|broken|missing)\s*(?:item|product|part|piece)',
        r'(?:received|got|delivered).*(?:wrong|incorrect|fake|damaged|broken)',
        r'(?:product|item|goods)\s*(?:is|are)\s*(?:broken|damaged|defective|not\s*work|faulty)',
        r'(?:missing|lost)\s*(?:parts?|pieces?|items?)',
        r'(?:not\s*as\s*described|not\s*what\s*(?:i|we)\s*ordered)',
        # Additional patterns to catch product issues that mention refund
        r'(?:return|returned|returning)\s*(?:the|this|my|it|item|product)\s*(?:because|since|due\s*to)\s*(?:broken|damaged|wrong|defective)',
        r'(?:broken|damaged|wrong|defective)\s*(?:item|product).*(?:refund|return)',
    ]),

    # 3. ORDER_STATUS - delivery/tracking (check before refund)
    ('order_status', [
        r'(?:where|when)\s*(?:is|was|will|did)\s*(?:my|the|our|this)?\s*(?:order|package|parcel|delivery|shipment)\b',
        r'(?:order|package|parcel|delivery|shipment)\s*(?:status|track|tracking|update|delayed|late|missing|lost)\b',
        r'(?:not\s*(?:received|arrived|delivered)|hasn\'?t\s*(?:arrived|been\s*delivered))',
        r'(?:still\s*waiting|when\s*(?:will|is|did)\s*(?:it|my|the))',
        r'(?:deliver|delivery|shipping|shipped|dispatch|courier|usps|ups|fedex|amzl)\s*(?:issue|problem|delay|late|lost)',
        r'(?:my|the)\s*(?:order|package).*(?:arrive|deliver|ship|where|when|status)',
    ]),

    # 4. CANCELLATION - cancel subscription/order
    ('cancellation', [
        r'cancel\s*(?:my|the|your|this)?\s*(?:subscription|membership|prime|account|order)',
        r'(?:how|where)\s*(?:do|can)\s*(?:i|we)\s*cancel',
        r'unsubscribe',
        r'stop\s*(?:my|the|your)?\s*(?:subscription|membership|prime|recurring|charging)',
        r'(?:cancel|cancellation)\s*(?:request|please|button|option|link)',
    ]),

    # 5. REFUND_RETURN - refund request (check AFTER product/order)
    ('refund_return', [
        r'refund\s*(?:my|the|your|this)?\s*(?:money|payment|order|amount)',
        r'(?:want|need|request|get|got)\s*(?:a\s*)?refund',
        r'money\s*back',
        r'reimburse',
        # More specific refund patterns that don't conflict with product/order
        r'(?:refund|reimburse)\s*(?:status|update|pending|processing)',
        r'(?:when|how)\s*(?:will|can)\s*(?:i|we)\s*(?:get|receive)\s*(?:a\s*)?refund',
    ]),

    # 6. BILLING_PAYMENT - payment/charge specific
    ('billing_payment', [
        r'(?:charged|charge)\s*(?:twice|double|extra|more|incorrect|wrong)',
        r'(?:double|twice|extra)\s*charg',
        r'(?:wrong|incorrect|bad)\s*(?:amount|price|charge|bill)',
        r'(?:payment|billing|invoice)\s*(?:issue|problem|error|failed)',
        r'revise.*payment',
        r'payment.*(?:method|declined|failed)',
    ]),

    # 7. COMPLAINT_FRUSTRATION - general anger (last - catch-all)
    ('complaint_frustration', [
        r'(?:terrible|worst|horrible|disgusting|pathetic|awful|furious|infuriated)',
        r'(?:never\s*again|done\s*with|sick\s*of|tired\s*of|had\s*enough)',
        r'(?:unacceptable|inexcusable|ridiculous|absurd)',
        r'(?:frustrated|angry|mad|furious|disappointed|let\s*down)',
        r'(?:no\s*(?:help|resolution|response|action)|still\s*no)',
        r'(?:multiple|several|many)\s*(?:calls|chats|attempts|times)',
    ]),
]


class HybridClassifier:
    """Combines keyword rules + ML model."""

    def __init__(self, ml_model_path='models/classifier.pkl'):
        """Load ML model if available."""
        self.ml_vectorizer = None
        self.ml_model = None

        if os.path.exists(ml_model_path):
            with open(ml_model_path, 'rb') as f:
                data = pickle.load(f)
                self.ml_vectorizer = data['vectorizer']
                self.ml_model = data['model']

    def classify_by_rules(self, text):
        """Classify using keyword rules only."""
        text_lower = text.lower()

        for intent, patterns in INTENT_RULES:
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent, 0.95  # High confidence for rule matches

        # Fallback: simple keywords
        simple_keywords = {
            'cancellation': ['cancel', 'subscription', 'membership'],
            'refund_return': ['refund', 'return', 'money back'],
            'billing_payment': ['charge', 'payment', 'billing'],
            'technical_support': ['app', 'website', 'login', 'error', 'crash'],
            'product_issue': ['product', 'item', 'broken', 'damaged', 'wrong'],
            'order_status': ['order', 'package', 'delivery', 'shipping'],
            'complaint_frustration': ['terrible', 'worst', 'angry', 'frustrated'],
        }

        for intent, keywords in simple_keywords.items():
            for kw in keywords:
                if kw in text_lower:
                    return intent, 0.7  # Medium confidence

        return 'complaint_frustration', 0.5  # Default

    def classify_by_ml(self, text):
        """Classify using ML model only."""
        if self.ml_vectorizer is None or self.ml_model is None:
            return None, 0.0

        X = self.ml_vectorizer.transform([text])
        intent = self.ml_model.predict(X)[0]
        confidence = max(self.ml_model.predict_proba(X)[0])

        return intent, confidence

    def classify(self, text):
        """
        Hybrid classification:
        1. Try keyword rules first (high confidence)
        2. Fall back to ML if no rule matches
        """
        # Step 1: Try keyword rules
        rule_intent, rule_conf = self.classify_by_rules(text)

        # If rule has high confidence, use it
        if rule_conf >= 0.9:
            return rule_intent, rule_conf, 'rules'

        # Step 2: Try ML model
        ml_intent, ml_conf = self.classify_by_ml(text)

        # If ML has higher confidence, use it
        if ml_conf > rule_conf:
            return ml_intent, ml_conf, 'ml'

        # Otherwise use rules
        return rule_intent, rule_conf, 'rules'


def test_hybrid_classifier():
    """Test the hybrid classifier."""
    print("="*70)
    print("HYBRID CLASSIFIER TEST (v2)")
    print("="*70)

    classifier = HybridClassifier()

    test_tweets = [
        # Clear cases (rules should handle)
        ("Where is my order? It's been 2 weeks", "order_status"),
        ("The app keeps crashing when I open it", "technical_support"),
        ("I received the wrong item", "product_issue"),
        ("I want a refund", "refund_return"),
        ("I was charged twice for Prime", "billing_payment"),
        ("How do I cancel my subscription?", "cancellation"),
        ("This is the worst experience!", "complaint_frustration"),

        # Ambiguous cases (v2 should handle better)
        ("My package was lost, I want a refund", "order_status"),
        ("Received broken product, need refund", "product_issue"),
        ("Charged twice and want to cancel", "billing_payment"),
        ("Still waiting for my order... will it arrive this year?", "order_status"),
        ("I've been charged extra on my bill", "billing_payment"),
        ("Can you help me return this item?", "refund_return"),
        ("Your website is not loading", "technical_support"),
        ("I'm so frustrated with your service", "complaint_frustration"),
        ("Cancel my Prime membership", "cancellation"),
        ("The product arrived broken", "product_issue"),
        ("I can't login to my account", "technical_support"),
    ]

    correct = 0
    total = len(test_tweets)

    print("\nTest Results:")
    print("-"*70)

    for tweet, expected in test_tweets:
        intent, confidence, source = classifier.classify(tweet)
        match = "OK" if intent == expected else "WRONG"
        if intent == expected:
            correct += 1

        print("\n  Tweet:", tweet[:60])
        print("  Expected:", expected)
        print("  Got:", intent, "(confidence:", confidence, ", source:", source, ")")
        print("  [", match, "]")

    print("\n" + "="*70)
    print("Accuracy:", correct, "/", total, "=", correct/total*100, "%")
    print("="*70)


if __name__ == "__main__":
    test_hybrid_classifier()
