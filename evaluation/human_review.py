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


def select_stratified_sample(golden_set, n=30):
    """Select n examples stratified by intent."""
    from collections import defaultdict
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

    # Load or create human scores
    human_data = load_or_create_human_scores()
    already_scored = {s['tweet_id'] for s in human_data['scores']}

    # Select sample
    sample = select_stratified_sample(golden_set, n=30)
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
    save_human_scores(human_data)

    print(f"\n{'=' * 70}")
    print(f"REVIEW COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total examples scored: {len(human_data['scores'])}")
    print(f"Scores saved to: evaluation/human_scores.json")

    return human_data


if __name__ == "__main__":
    run_human_review()
