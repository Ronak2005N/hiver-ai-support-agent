"""Detailed analysis of confusion patterns."""
import json
import sys
sys.path.insert(0, 'src')
from hybrid_classifier import HybridClassifier

classifier = HybridClassifier()

with open('evaluation/golden_set.json', 'r') as f:
    data = json.load(f)

# For each error, show what rules matched
errors_by_type = {
    'product_issue->refund_return': [],
    'order_status->refund_return': [],
    'billing_payment->cancellation': [],
    'order_status->product_issue': [],
    'order_status->complaint_frustration': [],
}

for ex in data:
    intent, conf, source = classifier.classify(ex['text'])
    key = ex['intent'] + '->' + intent
    if key in errors_by_type and len(errors_by_type[key]) < 5:
        # Analyze which rules matched
        text_lower = ex['text'].lower()
        
        # Check what keywords are present
        refund_kw = any(w in text_lower for w in ['refund', 'money back', 'return', 'reimburse'])
        order_kw = any(w in text_lower for w in ['order', 'package', 'delivery', 'shipping'])
        product_kw = any(w in text_lower for w in ['product', 'item', 'broken', 'damaged', 'wrong'])
        cancel_kw = any(w in text_lower for w in ['cancel', 'membership', 'subscription'])
        billing_kw = any(w in text_lower for w in ['charge', 'payment', 'billing'])
        complaint_kw = any(w in text_lower for w in ['terrible', 'worst', 'angry', 'frustrated'])
        
        errors_by_type[key].append({
            'text': ex['text'][:150],
            'keywords': {
                'refund': refund_kw,
                'order': order_kw,
                'product': product_kw,
                'cancel': cancel_kw,
                'billing': billing_kw,
                'complaint': complaint_kw
            }
        })

for error_type, examples in errors_by_type.items():
    print("=" * 80)
    print("ERROR TYPE:", error_type, "(", len(examples), "examples shown)")
    print("=" * 80)
    for i, ex in enumerate(examples):
        print(f"\nExample {i+1}:")
        print("  Text:", ex['text'])
        kw = ex['keywords']
        present = [k for k, v in kw.items() if v]
        print("  Keywords present:", present)
    print()
