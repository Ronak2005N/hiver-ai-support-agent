"""
Main Evaluation Script
Runs all evaluations and produces combined results.

Usage:
    python evaluation/evaluate.py
"""

import json
import sys
import os
from datetime import datetime

sys.path.insert(0, 'src')
from hybrid_classifier import HybridClassifier


def load_golden_set(path='evaluation/golden_set.json'):
    """Load the Golden Set."""
    with open(path, 'r') as f:
        return json.load(f)


def evaluate_all(golden_set_path='evaluation/golden_set.json'):
    """Run all evaluations."""
    print("=" * 70)
    print("COMPREHENSIVE EVALUATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Load data
    golden_set = load_golden_set(golden_set_path)
    print(f"\nGolden Set: {len(golden_set)} examples")
    
    # Initialize classifier
    classifier = HybridClassifier()
    print("Classifier: hybrid (rules + ML)")
    
    # Run predictions
    y_true = []
    y_pred = []
    results = []
    
    for ex in golden_set:
        text = ex['text']
        true_intent = ex['intent']
        true_escalate = ex['should_escalate']
        
        pred_intent, confidence, source = classifier.classify(text)
        
        y_true.append(true_intent)
        y_pred.append(pred_intent)
        
        results.append({
            'id': ex['id'],
            'text': text[:100],
            'true_intent': true_intent,
            'pred_intent': pred_intent,
            'true_escalate': true_escalate,
            'confidence': confidence,
            'source': source
        })
    
    # Intent metrics
    from sklearn.metrics import classification_report, confusion_matrix
    
    intents = sorted(set(y_true))
    report = classification_report(y_true, y_pred, labels=intents, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=intents)
    
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    
    # Escalation metrics
    tp = sum(1 for r in results if r['true_escalate'] and r['pred_intent'] == 'complaint_frustration')
    tn = sum(1 for r in results if not r['true_escalate'] and r['pred_intent'] != 'complaint_frustration')
    fp = sum(1 for r in results if not r['true_escalate'] and r['pred_intent'] == 'complaint_frustration')
    fn = sum(1 for r in results if r['true_escalate'] and r['pred_intent'] != 'complaint_frustration')
    
    esc_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    esc_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    esc_f1 = 2 * esc_precision * esc_recall / (esc_precision + esc_recall) if (esc_precision + esc_recall) > 0 else 0
    esc_accuracy = (tp + tn) / len(results)
    
    # Print results
    print("\n" + "=" * 70)
    print("INTENT CLASSIFICATION RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nPer-Class Metrics:")
    print("-" * 70)
    for intent in intents:
        metrics = report[intent]
        print(f"  {intent:25s} P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1-score']:.2f} (n={metrics['support']:.0f})")
    
    print("\nMacro Average:")
    macro = report['macro avg']
    print(f"  {'macro avg':25s} P={macro['precision']:.2f} R={macro['recall']:.2f} F1={macro['f1-score']:.2f}")
    
    print("\nConfusion Matrix:")
    print("-" * 70)
    header = "Actual\\Pred        "
    for intent in intents:
        header += f" {intent[:8]:8s}"
    print(header)
    
    for i, true_intent in enumerate(intents):
        row = f"{true_intent:20s}"
        for j, pred_intent in enumerate(intents):
            count = cm[i][j]
            row += f" {count:8d}"
        print(row)
    
    print("\n" + "=" * 70)
    print("ESCALATION RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {esc_accuracy:.4f} ({esc_accuracy*100:.2f}%)")
    print(f"Precision: {esc_precision:.4f} ({esc_precision*100:.2f}%)")
    print(f"Recall: {esc_recall:.4f} ({esc_recall*100:.2f}%)")
    print(f"F1 Score: {esc_f1:.4f} ({esc_f1*100:.2f}%)")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (TP):  {tp}")
    print(f"  True Negatives (TN):  {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    
    # Save combined results
    combined_results = {
        'timestamp': datetime.now().isoformat(),
        'golden_set_size': len(golden_set),
        'intent': {
            'accuracy': accuracy,
            'macro_precision': macro['precision'],
            'macro_recall': macro['recall'],
            'macro_f1': macro['f1-score'],
            'per_class': {intent: report[intent] for intent in intents},
            'confusion_matrix': cm.tolist()
        },
        'escalation': {
            'accuracy': esc_accuracy,
            'precision': esc_precision,
            'recall': esc_recall,
            'f1': esc_f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    }
    
    output_path = 'evaluation/results.json'
    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return combined_results


if __name__ == "__main__":
    evaluate_all()
