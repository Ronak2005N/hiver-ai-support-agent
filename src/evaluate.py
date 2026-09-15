"""
EVALUATION HARNESS
Evaluates the classifier against the Golden Set.

Metrics:
- Overall accuracy
- Per-class precision, recall, F1
- Confusion matrix
- Escalation accuracy
- Failure analysis

Usage:
    python src/evaluate.py
"""

import pandas as pd
import pickle
import json
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_golden_set(path='evaluation/golden_set.json'):
    """Load the Golden Set."""
    with open(path, 'r') as f:
        return json.load(f)


def load_hybrid_classifier():
    """Load the hybrid classifier."""
    import sys
    sys.path.insert(0, 'src')
    from hybrid_classifier import HybridClassifier
    return HybridClassifier()


def evaluate_intentclassification(golden_set, classifier):
    """Evaluate intent classification using hybrid classifier."""
    y_true = []
    y_pred = []
    texts = []
    sources = []

    for example in golden_set:
        text = example['text']
        true_intent = example['intent']

        # Predict using hybrid classifier
        pred_intent, confidence, source = classifier.classify(text)

        y_true.append(true_intent)
        y_pred.append(pred_intent)
        texts.append(text)
        sources.append(source)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true) | set(y_pred)))

    # Count sources
    from collections import Counter
    source_counts = Counter(sources)

    return {
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'labels': sorted(set(y_true) | set(y_pred)),
        'y_true': y_true,
        'y_pred': y_pred,
        'texts': texts,
        'sources': source_counts
    }


def analyze_failures(golden_set, classifier):
    """Analyze classification failures."""
    failures = []

    for example in golden_set:
        text = example['text']
        true_intent = example['intent']

        pred_intent, confidence, source = classifier.classify(text)

        if true_intent != pred_intent:
            failures.append({
                'text': text[:100],
                'true_intent': true_intent,
                'predicted_intent': pred_intent,
                'confidence': confidence,
                'source': source,
                'difficulty': example.get('difficulty', 'unknown')
            })

    return failures


def evaluate_escalation(golden_set):
    """Evaluate escalation decisions (baseline: always auto-handle)."""
    # Simple baseline: escalate only if complaint_frustration
    correct = 0
    for example in golden_set:
        true_escalate = example['should_escalate']
        # Baseline: escalate if complaint_frustration
        pred_escalate = example['intent'] == 'complaint_frustration'

        if true_escalate == pred_escalate:
            correct += 1

    return correct / len(golden_set)


def print_results(intent_results, failures, escalation_acc):
    """Print evaluation results."""
    print("="*70)
    print("EVALUATION RESULTS")
    print("="*70)

    # Overall accuracy
    print(f"\nOverall Intent Accuracy: {intent_results['accuracy']:.2%}")

    # Per-class metrics
    print("\nPer-Class Metrics:")
    print("-"*70)
    report = intent_results['report']
    for intent in sorted(report.keys()):
        if isinstance(report[intent], dict):
            p = report[intent].get('precision', 0)
            r = report[intent].get('recall', 0)
            f1 = report[intent].get('f1-score', 0)
            support = report[intent].get('support', 0)
            print(f"  {intent:25s} P={p:.2f} R={r:.2f} F1={f1:.2f} (n={support})")

    # Confusion matrix
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print("-"*70)
    labels = intent_results['labels']
    cm = intent_results['confusion_matrix']

    header = " {:22s}".format("Actual\\Pred")
    for l in labels:
        header += " {:>8s}".format(l[:8])
    print(header)
    print(" " + "-" * (22 + 9 * len(labels)))

    for i, row in enumerate(cm):
        row_str = " {:22s}".format(labels[i][:22])
        for val in row:
            row_str += " {:>8d}".format(val)
        print(row_str)

    # Failure analysis
    print("\n" + "="*70)
    print("FAILURE ANALYSIS")
    print("="*70)
    print(f"\nTotal failures: {len(failures)} / {len(intent_results['y_true'])} ({len(failures)/len(intent_results['y_true'])*100:.1f}%)")

    # Most common error patterns
    from collections import Counter
    error_pairs = Counter([(f['true_intent'], f['predicted_intent']) for f in failures])
    print("\nMost Common Error Patterns:")
    for (actual, predicted), count in error_pairs.most_common(5):
        print(f"  {actual:25s} -> {predicted:25s} : {count} times")

    # Escalation
    print("\n" + "="*70)
    print("ESCALATION ACCURACY")
    print("="*70)
    print(f"\nEscalation accuracy (baseline): {escalation_acc:.2%}")

    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    print("""
1. CLASSIFIER IMPROVEMENTS NEEDED:
   - complaint_frustration class dominates (54% of training data)
   - Need more balanced training data
   - Consider class weights or oversampling

2. ESCALATION RULES:
   - Current: Only escalate if complaint_frustration
   - Should: Escalate based on specific keywords + confidence

3. RETRIEVAL:
   - Need better similarity measure (consider embeddings)
   - Should filter by intent before retrieval

4. REPLY GENERATION:
   - Template replies are too generic
   - Need intent-specific templates
   - Consider using LLM for generation
""")


def main():
    """Run evaluation."""
    print("LOADING DATA...")
    golden_set = load_golden_set()
    classifier = load_hybrid_classifier()

    print(f"Golden Set: {len(golden_set)} examples")
    print(f"Classifier: hybrid (rules + ML)")

    # Evaluate intent classification
    print("\nEvaluating intent classification...")
    intent_results = evaluate_intentclassification(golden_set, classifier)

    # Analyze failures
    print("Analyzing failures...")
    failures = analyze_failures(golden_set, classifier)

    # Evaluate escalation
    print("Evaluating escalation...")
    escalation_acc = evaluate_escalation(golden_set)

    # Print results
    print_results(intent_results, failures, escalation_acc)

    # Save results
    results = {
        'accuracy': intent_results['accuracy'],
        'per_class': {k: v for k, v in intent_results['report'].items() if isinstance(v, dict)},
        'escalation_accuracy': escalation_acc,
        'num_failures': len(failures),
        'num_examples': len(golden_set)
    }

    with open('evaluation/src_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\nResults saved to evaluation/src_results.json")


if __name__ == "__main__":
    main()
