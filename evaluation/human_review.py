"""
Human Review Workflow
Lets a human scorer rate reply quality on 5 dimensions for a subset of the golden set.

This produces genuine human scores that can be compared against the LLM judge
to compute Cohen's Kappa.

Usage:
    python evaluation/human_review.py

The script:
1. Loads the LLM judge results
2. Presents 30 examples (stratified by intent, 4-5 per intent)
3. Human scores each reply on 5 dimensions (1-5)
4. Saves scores to evaluation/human_scores.json
5. Computes Cohen's Kappa against LLM scores

IMPORTANT: This produces REAL human scores. Do NOT fabricate scores.
"""

import json
import sys
import os
import random

sys.path.insert(0, 'src')
from hybrid_classifier import HybridClassifier
from retriever import HistoricalRetriever
from generator import ReplyGenerator


def load_or_create_human_scores():
    """Load existing human scores or create empty structure."""
    path = 'evaluation/human_scores.json'
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'scores': [], 'metadata': {}}


def save_human_scores(data):
    """Save human scores."""
    path = 'evaluation/human_scores.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def select_stratified_sample(golden_set, n=30, valid_ids=None):
    """Select n examples stratified by intent, optionally filtered to valid_ids."""
    from collections import defaultdict
    if valid_ids is not None:
        golden_set = [ex for ex in golden_set if ex['id'] in valid_ids]
    by_intent = defaultdict(list)
    for ex in golden_set:
        by_intent[ex['intent']].append(ex)

    sample = []
    intents = sorted(by_intent.keys())
    per_intent = n // len(intents)
    remainder = n % len(intents)

    for intent in intents:
        count = per_intent + (1 if intents.index(intent) < remainder else 0)
        available = by_intent[intent]
        selected = random.sample(available, min(count, len(available)))
        sample.extend(selected)

    return sample


def run_human_review():
    """Interactive human review workflow."""
    print("=" * 70)
    print("HUMAN REVIEW WORKFLOW")
    print("=" * 70)
    print("\nThis workflow lets you score reply quality for 30 examples.")
    print("Your scores will be compared against the LLM judge to compute Cohen's Kappa.")
    print("\nIMPORTANT: These are REAL human scores. Be honest and consistent.")
    print("Do NOT score everything as 3 or 4. Use the full 1-5 range.\n")

    # Load golden set
    with open('evaluation/golden_set.json', 'r', encoding='utf-8') as f:
        golden_set = json.load(f)

    # Load judge results to get valid IDs (only these have LLM scores for kappa)
    with open('evaluation/judge_results.json', 'r', encoding='utf-8') as f:
        judge_data = json.load(f)
    valid_ids = {r['tweet_id'] for r in judge_data['results']}
    print(f"LLM judge has scores for {len(valid_ids)} examples (IDs 1-{max(valid_ids)}).")
    print("Sampling only from these so Cohen's Kappa can be computed.\n")

    # Load or create human scores, keep only ones that match valid IDs
    human_data = load_or_create_human_scores()
    human_data['scores'] = [s for s in human_data['scores'] if s['tweet_id'] in valid_ids]
    already_scored = {s['tweet_id'] for s in human_data['scores']}

    # Select sample from valid IDs only
    sample = select_stratified_sample(golden_set, n=30, valid_ids=valid_ids)
    sample = [s for s in sample if s['id'] not in already_scored]

    if not sample:
        print("\nAll 30 examples have already been scored!")
        print("Scores saved in evaluation/human_scores.json")
        return human_data

    print(f"\nExamples to score: {len(sample)}")
    print(f"Previously scored: {len(already_scored)}")

    # Initialize pipeline
    classifier = HybridClassifier()
    retriever = HistoricalRetriever()
    generator = ReplyGenerator()

    dimensions = ['relevance', 'empathy', 'actionability', 'tone', 'grounding']
    dim_descriptions = {
        'relevance': 'Does the reply address the specific issue?',
        'empathy': 'Does the reply acknowledge customer feelings?',
        'actionability': 'Does the reply provide clear next steps?',
        'tone': 'Is the reply professional and helpful?',
        'grounding': 'Is the reply based on similar examples?'
    }

    for i, example in enumerate(sample):
        tweet = example['text']
        intent = example['intent']

        # Generate reply
        pred_intent, confidence, source = classifier.classify(tweet)
        similar = retriever.retrieve(tweet, pred_intent)
        reply = generator.generate(tweet, pred_intent, similar)

        print(f"\n{'=' * 70}")
        print(f"EXAMPLE {i + 1} of {len(sample)} (ID: {example['id']})")
        print(f"{'=' * 70}")
        print(f"\nTweet: {tweet[:200]}...")
        print(f"Intent: {intent}")
        print(f"Predicted: {pred_intent} ({confidence:.0%})")
        print(f"\nReply: {reply}")

        # Score each dimension
        scores = {}
        for dim in dimensions:
            while True:
                try:
                    print(f"\n  {dim.upper()}: {dim_descriptions[dim]}")
                    score = int(input(f"  Score (1-5): "))
                    if 1 <= score <= 5:
                        scores[dim] = score
                        break
                    else:
                        print("  Please enter 1-5.")
                except ValueError:
                    print("  Please enter a number 1-5.")

        # Save score
        human_data['scores'].append({
            'tweet_id': example['id'],
            'intent': intent,
            'predicted_intent': pred_intent,
            'reply': reply,
            'scores': scores,
            'total_score': round(sum(scores[k] for k in scores) / len(scores), 2)
        })

        save_human_scores(human_data)
        print(f"  Saved. ({len(human_data['scores'])}/30)")

    # Update metadata
    human_data['metadata'] = {
        'total_scored': len(human_data['scores']),
        'dimensions': dimensions,
        'methodology': 'Human scorer rated each reply on 5 dimensions (1-5 scale)'
    }

    # Compute Cohen's Kappa against LLM judge
    judge_lookup = {r['tweet_id']: r for r in judge_data['results']}
    kappa_per_dim = {}
    for dim in dimensions:
        h_vals, l_vals = [], []
        for s in human_data['scores']:
            if s['tweet_id'] in judge_lookup:
                h_vals.append(s['scores'][dim])
                l_vals.append(judge_lookup[s['tweet_id']]['scores'][dim])
        if h_vals:
            kappa_per_dim[dim] = round(_cohens_kappa(h_vals, l_vals), 3)

    all_h, all_l = [], []
    for s in human_data['scores']:
        if s['tweet_id'] in judge_lookup:
            for dim in dimensions:
                all_h.append(s['scores'][dim])
                all_l.append(judge_lookup[s['tweet_id']]['scores'][dim])
    overall_kappa = round(_cohens_kappa(all_h, all_l), 3) if all_h else 0

    human_data['metadata']['cohens_kappa_per_dimension'] = kappa_per_dim
    human_data['metadata']['cohens_kappa_overall'] = overall_kappa
    save_human_scores(human_data)

    print(f"\n{'=' * 70}")
    print(f"REVIEW COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total examples scored: {len(human_data['scores'])}")
    print(f"Cohen's Kappa (overall): {overall_kappa}")
    for dim, k in kappa_per_dim.items():
        print(f"  {dim}: {k}")
    print(f"Scores saved to: evaluation/human_scores.json")

    return human_data


def _cohens_kappa(rater1, rater2):
    """Compute Cohen's Kappa between two lists of integer scores."""
    from collections import Counter
    cats = sorted(set(rater1) | set(rater2))
    n = len(rater1)
    mat = Counter(zip(rater1, rater2))
    po = sum(mat[(c, c)] for c in cats) / n
    pe = 0
    for c in cats:
        pe += (sum(1 for x in rater1 if x == c) / n) * (sum(1 for x in rater2 if x == c) / n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


if __name__ == "__main__":
    run_human_review()
