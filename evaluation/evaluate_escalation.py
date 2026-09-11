"""
Escalation Evaluation
Evaluates escalation decision performance.

Usage:
    python evaluation/evaluate_escalation.py
"""

import json
import sys
import os
from collections import Counter

sys.path.insert(0, 'src')
from hybrid_classifier import HybridClassifier


def load_golden_set(path='evaluation/golden_set.json'):
    """Load the Golden Set."""
    with open(path, 'r') as f:
        return json.load(f)


def evaluate_escalation(golden_set_path='evaluation/golden_set.json'):
    """Evaluate escalation decisions."""
    print("=" * 70)
    print("ESCALATION EVALUATION")
    print("=" * 70)
    
    # Load data
    golden_set = load_golden_set(golden_set_path)
    print(f"\nGolden Set: {len(golden_set)} examples")
    
    # Initialize classifier
    classifier = HybridClassifier()
    print("Classifier: hybrid (rules + ML)")
    
    # Escalation rules (same as agent)
    threat_words = ['lawyer', 'sue', 'legal', 'attorney', 'court', 'BBB']
    angry_words = ['furious', 'unacceptable', 'worst ever', 'never again']
    
    # Run predictions
    results = []
    
    for ex in golden_set:
        text = ex['text']
        true_intent = ex['intent']
        true_escalate = ex['should_escalate']
        
        # Classify
        pred_intent, confidence, source = classifier.classify(text)
        
        # Determine escalation
        reasons = []
        
        # Low confidence
        if confidence < 0.5:
            reasons.append("Low classification confidence")
        
        # Legal/threat keywords
        if any(word in text.lower() for word in threat_words):
            reasons.append("Contains legal terminology")
        
        # Very angry
        if any(word in text.lower() for word in angry_words):
            reasons.append("High frustration level")
        
        pred_escalate = len(reasons) > 0
        escalation_reason = " | ".join(reasons) if reasons else "Auto-handleable"
        
        results.append({
            'id': ex['id'],
            'text': text[:100],
            'true_intent': true_intent,
            'pred_intent': pred_intent,
            'true_escalate': true_escalate,
            'pred_escalate': pred_escalate,
            'escalation_reason': escalation_reason,
            'confidence': confidence
        })
    
    # Calculate metrics
    tp = sum(1 for r in results if r['true_escalate'] and r['pred_escalate'])
    tn = sum(1 for r in results if not r['true_escalate'] and not r['pred_escalate'])
    fp = sum(1 for r in results if not r['true_escalate'] and r['pred_escalate'])
    fn = sum(1 for r in results if r['true_escalate'] and not r['pred_escalate'])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(results)
    
    auto_handle_rate = sum(1 for r in results if not r['pred_escalate']) / len(results)
    escalation_rate = sum(1 for r in results if r['pred_escalate']) / len(results)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Total Examples: {len(results)}")
    
    print("\nConfusion Matrix:")
    print(f"  True Positives (TP):  {tp} (escalated correctly)")
    print(f"  True Negatives (TN):  {tn} (auto-handled correctly)")
    print(f"  False Positives (FP): {fp} (escalated incorrectly)")
    print(f"  False Negatives (FN): {fn} (auto-handled incorrectly)")
    
    print("\nMetrics:")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    print("\nRates:")
    print(f"  Auto-handle rate: {auto_handle_rate:.4f} ({auto_handle_rate*100:.2f}%)")
    print(f"  Escalation rate:  {escalation_rate:.4f} ({escalation_rate*100:.2f}%)")
    
    # False negatives (most important!)
    print("\n" + "=" * 70)
    print("FALSE NEGATIVES (Auto-handled but should have escalated)")
    print("=" * 70)
    
    false_negatives = [r for r in results if r['true_escalate'] and not r['pred_escalate']]
    print(f"\nTotal False Negatives: {len(false_negatives)}")
    
    for fn in false_negatives[:10]:
        print(f"\n  ID: {fn['id']}")
        try:
            print(f"  Text: {fn['text']}")
        except UnicodeEncodeError:
            print(f"  Text: [Unicode encoding error]")
        print(f"  True Intent: {fn['true_intent']}")
        print(f"  Predicted Intent: {fn['pred_intent']}")
        print(f"  Confidence: {fn['confidence']:.2f}")
    
    # Baseline comparison
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON")
    print("=" * 70)
    
    # Baseline: escalate only complaint_frustration
    baseline_tp = sum(1 for r in results if r['true_escalate'] and r['true_intent'] == 'complaint_frustration')
    baseline_fn = sum(1 for r in results if r['true_escalate'] and r['true_intent'] != 'complaint_frustration')
    baseline_fp = sum(1 for r in results if not r['true_escalate'] and r['true_intent'] == 'complaint_frustration')
    baseline_tn = sum(1 for r in results if not r['true_escalate'] and r['true_intent'] != 'complaint_frustration')
    
    baseline_precision = baseline_tp / (baseline_tp + baseline_fp) if (baseline_tp + baseline_fp) > 0 else 0
    baseline_recall = baseline_tp / (baseline_tp + baseline_fn) if (baseline_tp + baseline_fn) > 0 else 0
    baseline_f1 = 2 * baseline_precision * baseline_recall / (baseline_precision + baseline_recall) if (baseline_precision + baseline_recall) > 0 else 0
    baseline_accuracy = (baseline_tp + baseline_tn) / len(results)
    
    print(f"\nBaseline (escalate complaint_frustration only):")
    print(f"  Accuracy:  {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    print(f"  Precision: {baseline_precision:.4f} ({baseline_precision*100:.2f}%)")
    print(f"  Recall:    {baseline_recall:.4f} ({baseline_recall*100:.2f}%)")
    print(f"  F1 Score:  {baseline_f1:.4f} ({baseline_f1*100:.2f}%)")
    
    print(f"\nOurs:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    
    # Save results
    output_results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'auto_handle_rate': auto_handle_rate,
        'escalation_rate': escalation_rate,
        'baseline': {
            'accuracy': baseline_accuracy,
            'precision': baseline_precision,
            'recall': baseline_recall,
            'f1': baseline_f1
        },
        'false_negatives': false_negatives
    }
    
    output_path = 'evaluation/escalation_results.json'
    with open(output_path, 'w') as f:
        json.dump(output_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return output_results


if __name__ == "__main__":
    evaluate_escalation()
