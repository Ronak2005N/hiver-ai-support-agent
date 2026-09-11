"""
Intent Classification Evaluation
Evaluates classifier performance on Golden Set.

Usage:
    python evaluation/evaluate_intent.py
"""

import json
import sys
import os
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, 'src')
from hybrid_classifier import HybridClassifier


def load_golden_set(path='evaluation/golden_set.json'):
    """Load the Golden Set."""
    with open(path, 'r') as f:
        return json.load(f)


def evaluate_intent(golden_set_path='evaluation/golden_set.json'):
    """Evaluate intent classification."""
    print("=" * 70)
    print("INTENT CLASSIFICATION EVALUATION")
    print("=" * 70)
    
    # Load data
    golden_set = load_golden_set(golden_set_path)
    print(f"\nGolden Set: {len(golden_set)} examples")
    
    # Initialize classifier
    classifier = HybridClassifier()
    print("Classifier: hybrid (rules + ML)")
    
    # Run predictions
    y_true = []
    y_pred = []
    errors = []
    
    for ex in golden_set:
        text = ex['text']
        true_intent = ex['intent']
        
        pred_intent, confidence, source = classifier.classify(text)
        
        y_true.append(true_intent)
        y_pred.append(pred_intent)
        
        if pred_intent != true_intent:
            errors.append({
                'id': ex['id'],
                'text': text[:100],
                'expected': true_intent,
                'predicted': pred_intent,
                'confidence': confidence,
                'source': source
            })
    
    # Calculate metrics
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    
    # Per-class report
    intents = sorted(set(y_true))
    report = classification_report(y_true, y_pred, labels=intents, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=intents)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total Examples: {len(y_true)}")
    print(f"Total Errors: {len(errors)}")
    
    print("\nPer-Class Metrics:")
    print("-" * 70)
    for intent in intents:
        metrics = report[intent]
        print(f"  {intent:25s} P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1-score']:.2f} (n={metrics['support']:.0f})")
    
    print("\nMacro Average:")
    macro = report['macro avg']
    print(f"  {'macro avg':25s} P={macro['precision']:.2f} R={macro['recall']:.2f} F1={macro['f1-score']:.2f}")
    
    # Confusion matrix
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print("-" * 70)
    
    # Header
    header = "Actual\\Pred        "
    for intent in intents:
        header += f" {intent[:8]:8s}"
    print(header)
    
    # Rows
    for i, true_intent in enumerate(intents):
        row = f"{true_intent:20s}"
        for j, pred_intent in enumerate(intents):
            count = cm[i][j]
            row += f" {count:8d}"
        print(row)
    
    # Error analysis
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)
    
    error_types = Counter((e['expected'], e['predicted']) for e in errors)
    print("\nMost Common Error Patterns:")
    for (exp, pred), count in error_types.most_common(10):
        print(f"  {exp:25s} -> {pred:25s} : {count} times")
    
    # Save results
    results = {
        'accuracy': accuracy,
        'total_examples': len(y_true),
        'total_errors': len(errors),
        'per_class': {intent: report[intent] for intent in intents},
        'macro_avg': report['macro avg'],
        'confusion_matrix': cm.tolist(),
        'error_types': {f"{k[0]}->{k[1]}": v for k, v in error_types.items()},
        'errors': errors[:20]  # Save first 20 errors
    }
    
    output_path = 'evaluation/intent_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return results


if __name__ == "__main__":
    evaluate_intent()
