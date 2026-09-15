"""
Failure Analysis
Identifies top failure modes from evaluation errors.

Usage:
    python evaluation/failure_analysis.py
"""

import json
import sys
import os
from collections import Counter

sys.path.insert(0, 'src')
from hybrid_classifier import HybridClassifier


def load_golden_set(path='evaluation/golden_set.json'):
    """Load the Golden Set."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_failures(golden_set_path='evaluation/golden_set.json'):
    """Analyze failure modes."""
    print("=" * 70)
    print("FAILURE ANALYSIS")
    print("=" * 70)
    
    # Load data
    golden_set = load_golden_set(golden_set_path)
    print(f"\nGolden Set: {len(golden_set)} examples")
    
    # Initialize classifier
    classifier = HybridClassifier()
    
    # Run predictions and collect errors
    errors = []
    
    for ex in golden_set:
        text = ex['text']
        true_intent = ex['intent']
        
        pred_intent, confidence, source = classifier.classify(text)
        
        if pred_intent != true_intent:
            # Analyze why it failed
            text_lower = text.lower()
            
            # Check what keywords are present
            refund_kw = any(w in text_lower for w in ['refund', 'money back', 'return', 'reimburse'])
            order_kw = any(w in text_lower for w in ['order', 'package', 'delivery', 'shipping'])
            product_kw = any(w in text_lower for w in ['product', 'item', 'broken', 'damaged', 'wrong'])
            cancel_kw = any(w in text_lower for w in ['cancel', 'membership', 'subscription'])
            billing_kw = any(w in text_lower for w in ['charge', 'payment', 'billing'])
            complaint_kw = any(w in text_lower for w in ['terrible', 'worst', 'angry', 'frustrated'])
            technical_kw = any(w in text_lower for w in ['app', 'website', 'login', 'error', 'crash'])
            
            errors.append({
                'id': ex['id'],
                'text': text[:150],
                'expected': true_intent,
                'predicted': pred_intent,
                'confidence': confidence,
                'source': source,
                'keywords': {
                    'refund': refund_kw,
                    'order': order_kw,
                    'product': product_kw,
                    'cancel': cancel_kw,
                    'billing': billing_kw,
                    'complaint': complaint_kw,
                    'technical': technical_kw
                },
                'difficulty': ex['difficulty'],
                'should_escalate': ex['should_escalate']
            })
    
    # Analyze error patterns
    error_types = Counter((e['expected'], e['predicted']) for e in errors)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nTotal Errors: {len(errors)} / {len(golden_set)} ({len(errors)/len(golden_set)*100:.1f}%)")
    
    print("\nTop 5 Failure Modes:")
    print("-" * 70)
    
    top_errors = error_types.most_common(5)
    failure_modes = []
    
    for i, ((exp, pred), count) in enumerate(top_errors, 1):
        examples = [e for e in errors if e['expected'] == exp and e['predicted'] == pred]
        
        # Analyze common keywords
        keyword_counts = Counter()
        for ex in examples:
            for kw, present in ex['keywords'].items():
                if present:
                    keyword_counts[kw] += 1
        
        common_keywords = [kw for kw, _ in keyword_counts.most_common(3)]
        
        print(f"\n{i}. {exp} -> {pred} ({count} times)")
        print(f"   Common keywords: {common_keywords}")
        try:
            print(f"   Example: {examples[0]['text'][:80]}...")
        except UnicodeEncodeError:
            print(f"   Example: [Unicode encoding error]")
        
        failure_modes.append({
            'rank': i,
            'source_intent': exp,
            'predicted_intent': pred,
            'count': count,
            'percentage': count / len(errors) * 100,
            'common_keywords': common_keywords,
            'example': examples[0]['text'][:150],
            'root_cause': f"Tweets contain keywords for both {exp} and {pred}"
        })
    
    # Analyze by difficulty
    print("\n" + "=" * 70)
    print("ERRORS BY DIFFICULTY")
    print("=" * 70)
    
    difficulty_counts = Counter(e['difficulty'] for e in errors)
    for diff, count in difficulty_counts.most_common():
        pct = count / len(errors) * 100
        print(f"  {diff}: {count} ({pct:.1f}%)")
    
    # Analyze escalation errors
    print("\n" + "=" * 70)
    print("ESCALATION ERRORS")
    print("=" * 70)
    
    escalation_errors = [e for e in errors if e['should_escalate']]
    print(f"  Errors that should have been escalated: {len(escalation_errors)}")
    
    for ex in escalation_errors[:5]:
        print(f"\n  ID: {ex['id']}")
        print(f"  Text: {ex['text'][:80]}...")
        print(f"  Expected: {ex['expected']}")
        print(f"  Predicted: {ex['predicted']}")
    
    # Save results
    output_results = {
        'total_errors': len(errors),
        'total_examples': len(golden_set),
        'error_rate': len(errors) / len(golden_set),
        'failure_modes': failure_modes,
        'errors_by_difficulty': dict(difficulty_counts),
        'escalation_errors': len(escalation_errors),
        'all_errors': errors
    }
    
    output_path = 'evaluation/failure_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return output_results


if __name__ == "__main__":
    analyze_failures()
