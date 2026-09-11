"""Analyze classifier errors on Golden Set."""
import json
import sys
sys.path.insert(0, 'src')
from hybrid_classifier import HybridClassifier
from collections import Counter

classifier = HybridClassifier()

with open('evaluation/golden_set.json', 'r') as f:
    data = json.load(f)

# Find misclassified examples
errors = []
for ex in data:
    intent, conf, source = classifier.classify(ex['text'])
    if intent != ex['intent']:
        errors.append({
            'text': ex['text'][:120],
            'expected': ex['intent'],
            'predicted': intent,
            'confidence': conf,
            'source': source
        })

print("TOTAL ERRORS:", len(errors))
print()

# Show errors by type
error_types = Counter((e['expected'], e['predicted']) for e in errors)
for (exp, pred), count in error_types.most_common(10):
    print(exp, "->", pred, ":", count, "times")
    examples = [e for e in errors if e['expected']==exp and e['predicted']==pred][:3]
    for ex in examples:
        print("  Text:", ex['text'])
        print("  Source:", ex['source'])
    print()
