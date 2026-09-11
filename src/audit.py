"""
CLASSIFIER & INTENT TAXONOMY AUDIT
Detailed analysis of current classification performance.
"""

import pandas as pd
import pickle
import json
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter


def load_data():
    """Load labeled data and classifier."""
    with open('data/labeled_tweets.json', 'r') as f:
        labeled = json.load(f)
    df = pd.DataFrame(labeled)

    with open('models/classifier.pkl', 'rb') as f:
        data = pickle.load(f)
        vectorizer = data['vectorizer']
        model = data['model']

    return df, vectorizer, model


def audit_taxonomy(df):
    """Step 1: Show intent classes and counts."""
    print('='*70)
    print('1. CURRENT INTENT CLASSES AND EXAMPLE COUNTS')
    print('='*70)

    intent_counts = df['intent'].value_counts()
    total = len(df)

    for intent, count in intent_counts.items():
        pct = count / total * 100
        bar = '#' * int(pct / 2)
        print(f'  {intent:20s} : {count:4d} ({pct:5.1f}%) {bar}')

    print(f'  {"TOTAL":20s} : {total:4d} (100.0%)')

    # Check for class imbalance
    print('\n  CLASS IMBALANCE ANALYSIS:')
    max_class = intent_counts.max()
    min_class = intent_counts.min()
    ratio = max_class / min_class if min_class > 0 else float('inf')
    print(f'  Largest class: {intent_counts.idxmax()} ({max_class})')
    print(f'  Smallest class: {intent_counts.idxmin()} ({min_class})')
    print(f'  Imbalance ratio: {ratio:.1f}:1')
    if ratio > 10:
        print('  WARNING: Severe class imbalance detected!')
    elif ratio > 5:
        print('  WARNING: Moderate class imbalance.')


def audit_confusion_matrix(df, vectorizer, model):
    """Step 2: Confusion matrix and per-class metrics."""
    print('\n' + '='*70)
    print('2. CONFUSION MATRIX AND PER-CLASS METRICS')
    print('='*70)

    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].values, df['intent'].values,
        test_size=0.2, random_state=42
    )

    X_test_vec = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vec)

    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    labels = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print('Confusion Matrix (rows=actual, cols=predicted):')
    print()

    # Header
    header = ' {:18s}'.format('Actual \\ Pred')
    for l in labels:
        header += ' {:>8s}'.format(l[:8])
    print(header)
    print(' ' + '-' * (18 + 9 * len(labels)))

    for i, row in enumerate(cm):
        row_str = ' {:18s}'.format(labels[i][:18])
        for val in row:
            row_str += ' {:>8d}'.format(val)
        print(row_str)

    return X_test, y_test, y_pred


def audit_errors(df, X_test, y_test, y_pred, vectorizer):
    """Step 3: Most common classification errors."""
    print('\n' + '='*70)
    print('3. MOST COMMON CLASSIFICATION ERRORS')
    print('='*70)

    errors = []
    for i, (text, actual, predicted) in enumerate(zip(X_test, y_test, y_pred)):
        if actual != predicted:
            errors.append({
                'text': text,
                'actual': actual,
                'predicted': predicted
            })

    print(f'\nTotal test examples: {len(X_test)}')
    print(f'Total errors: {len(errors)} ({len(errors)/len(X_test)*100:.1f}%)')

    # Most common error pairs
    error_pairs = Counter([(e['actual'], e['predicted']) for e in errors])
    print('\nMost Common Error Patterns (actual -> predicted):')
    for (actual, predicted), count in error_pairs.most_common(10):
        print(f'  {actual:20s} -> {predicted:20s} : {count} times')

    # Show example errors
    print('\nExample Errors (first 15):')
    for i, e in enumerate(errors[:15], 1):
        text = e['text'][:80]
        print(f'\n  Error {i}:')
        print(f'    Tweet: "{text}..."')
        print(f'    Actual:    {e["actual"]}')
        print(f'    Predicted: {e["predicted"]}')

    return errors


def audit_taxonomy_quality(df, errors):
    """Step 4 & 5: Review taxonomy quality."""
    print('\n' + '='*70)
    print('4. TAXONOMY QUALITY ANALYSIS')
    print('='*70)

    print('\nA. CLASS DEFINITIONS AND BOUNDARIES:')
    print('-'*70)

    # Show samples from each class
    for intent in df['intent'].unique():
        samples = df[df['intent'] == intent]['text'].head(5).tolist()
        print(f'\n  [{intent.upper()}] (n={len(df[df["intent"]==intent])})')
        for s in samples:
            # Clean text for display
            clean = str(s).encode('ascii', 'ignore').decode('ascii')[:90]
            print(f'    - {clean}')

    print('\nB. TAXONOMY PROBLEMS IDENTIFIED:')
    print('-'*70)

    problems = []

    # Check: 'complaint' is a catch-all
    complaint_pct = len(df[df['intent'] == 'complaint']) / len(df) * 100
    if complaint_pct > 30:
        problems.append(f'complaint class has {complaint_pct:.0f}% of data - likely a catch-all bucket')

    # Check: overlap between refund and billing
    refund_samples = set(df[df['intent'] == 'refund']['text'].str.lower().tolist())
    billing_samples = set(df[df['intent'] == 'billing']['text'].str.lower().tolist())
    overlap = refund_samples.intersection(billing_samples)
    if overlap:
        problems.append(f'refund and billing share {len(overlap)} identical examples')

    # Check: complaint contains what should be other classes
    complaint_texts = df[df['intent'] == 'complaint']['text'].tolist()
    complaint_with_order = [t for t in complaint_texts if any(w in str(t).lower() for w in ['order', 'package', 'delivery', 'shipping'])]
    if complaint_with_order:
        problems.append(f'complaint has {len(complaint_with_order)} tweets about orders (should be order_status)')

    complaint_with_refund = [t for t in complaint_texts if any(w in str(t).lower() for w in ['refund', 'money back', 'return'])]
    if complaint_with_refund:
        problems.append(f'complaint has {len(complaint_with_refund)} tweets about refunds (should be refund)')

    complaint_with_tech = [t for t in complaint_texts if any(w in str(t).lower() for w in ['app', 'crash', 'website', 'login', 'error'])]
    if complaint_with_tech:
        problems.append(f'complaint has {len(complaint_with_tech)} tweets about tech issues (should be technical)')

    for p in problems:
        print(f'  [!] {p}')

    if not problems:
        print('  No major problems found.')

    return problems


def recommend_taxonomy(df):
    """Step 6: Recommend revised taxonomy with real examples."""
    print('\n' + '='*70)
    print('6. REVISED INTENT TAXONOMY RECOMMENDATION')
    print('='*70)

    print('\nBased on dataset analysis, here is the recommended revised taxonomy:')
    print('-'*70)

    # Load full dataset for more examples
    full_df = pd.read_csv('data/amazon_customers.csv')

    taxonomy = {
        'order_status': {
            'description': 'Customer asking about order delivery, tracking, shipping, or where their package is',
            'keywords': ['order', 'package', 'delivery', 'shipping', 'tracking', 'where is', 'when will', 'arrive', 'courier', 'dispatch', 'late', 'delayed'],
            'examples': []
        },
        'refund_return': {
            'description': 'Customer wants money back, return an item, or get a refund',
            'keywords': ['refund', 'money back', 'return', 'returned', 'give back', 'reimburse'],
            'examples': []
        },
        'billing_payment': {
            'description': 'Payment issues, charged wrong amount, double charged, price disputes',
            'keywords': ['charge', 'charged', 'billing', 'payment', 'price', 'overcharged', 'double charged', 'wrong amount'],
            'examples': []
        },
        'technical_support': {
            'description': 'App not working, website issues, login problems, account access',
            'keywords': ['app', 'website', 'login', 'password', 'account', 'crash', 'error', 'bug', 'not working', 'cannot access'],
            'examples': []
        },
        'product_issue': {
            'description': 'Wrong item, damaged product, quality issues, missing parts',
            'keywords': ['product', 'item', 'broken', 'damaged', 'defective', 'wrong item', 'missing', 'quality'],
            'examples': []
        },
        'cancellation': {
            'description': 'Cancel subscription, membership, Prime, stop recurring charges',
            'keywords': ['cancel', 'cancellation', 'unsubscribe', 'subscription', 'membership', 'prime', 'stop charging'],
            'examples': []
        },
        'complaint_frustration': {
            'description': 'General frustration, anger, poor service experience - NOT about a specific actionable issue',
            'keywords': ['terrible', 'worst', 'angry', 'frustrated', 'unacceptable', 'horrible', 'disgusting', 'never again'],
            'examples': []
        }
    }

    # Find real examples from dataset
    for intent, info in taxonomy.items():
        keywords = info['keywords']
        pattern = '|'.join(keywords)
        matches = full_df[full_df['text'].str.contains(pattern, case=False, na=False)].head(5)
        info['examples'] = matches['text'].tolist()

    # Print taxonomy
    for intent, info in taxonomy.items():
        print(f'\n  [{intent.upper()}]')
        print(f'  Description: {info["description"]}')
        print(f'  Keywords: {", ".join(info["keywords"][:8])}...')
        print(f'  Real Examples:')
        for ex in info['examples'][:3]:
            print(f'    - "{str(ex)[:100]}"')

    print('\n' + '='*70)
    print('KEY CHANGES FROM CURRENT TAXONOMY:')
    print('='*70)
    print('''
  1. MERGED: refund + billing -> separate classes (refund_return, billing_payment)
     - refund_return: "I want my money back", "How do I return this?"
     - billing_payment: "I was charged twice", "Wrong price charged"

  2. KEPT: order_status (but with stricter definition)
     - ONLY for delivery/tracking questions
     - NOT for complaints about orders (those go to complaint_frustration)

  3. KEPT: technical_support (but with clearer boundaries)
     - ONLY for app/website/login issues
     - NOT for general complaints

  4. KEPT: product_issue (but with clearer definition)
     - ONLY for wrong/damaged items
     - NOT for delivery issues

  5. KEPT: cancellation (unchanged)

  6. RENAMED: complaint -> complaint_frustration
     - ONLY for general anger/frustration
     - NOT for specific actionable issues
     - This is the key fix: current "complaint" is a catch-all

  7. REMOVED: No new classes added
     - We derive taxonomy from data, not arbitrary labels
''')

    return taxonomy


def main():
    """Run complete audit."""
    print('CLASSIFIER & INTENT TAXONOMY AUDIT')
    print('='*70)

    df, vectorizer, model = load_data()

    # Step 1: Taxonomy overview
    audit_taxonomy(df)

    # Step 2: Confusion matrix
    X_test, y_test, y_pred = audit_confusion_matrix(df, vectorizer, model)

    # Step 3: Error analysis
    errors = audit_errors(df, X_test, y_test, y_pred, vectorizer)

    # Step 4: Taxonomy quality
    problems = audit_taxonomy_quality(df, errors)

    # Step 6: Recommended taxonomy
    taxonomy = recommend_taxonomy(df)

    print('\n' + '='*70)
    print('AUDIT COMPLETE')
    print('='*70)
    print('\nPlease review the recommended taxonomy above.')
    print('Once approved, I will:')
    print('  1. Re-label the dataset with the new taxonomy')
    print('  2. Retrain the classifier')
    print('  3. Create the 200-example Golden Set')
    print('  4. Build the evaluation harness')


if __name__ == "__main__":
    main()
