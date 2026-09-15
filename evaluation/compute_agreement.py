"""
Compute Cohen's Kappa between human and LLM judge scores.

This script:
1. Loads human scores from evaluation/human_scores.json
2. Loads LLM judge scores from evaluation/judge_results.json
3. Computes Cohen's Kappa on matched examples
4. Reports genuine agreement metrics

Usage:
    python evaluation/compute_agreement.py

REQUIRES: evaluation/human_scores.json must exist with real human scores.
          Run: python evaluation/human_review.py
"""

import json
import sys
import os


def compute_cohen_kappa(scores_a, scores_b, weights='quadratic'):
    """Compute Cohen's Kappa between two sets of scores."""
    assert len(scores_a) == len(scores_b), "Score arrays must be same length"
    n = len(scores_a)

    # Bin scores into categories for kappa calculation
    # Round to nearest integer for categorical kappa
    cats_a = [int(round(s)) for s in scores_a]
    cats_b = [int(round(s)) for s in scores_b]

    all_cats = sorted(set(cats_a) | set(cats_b))

    # Build confusion matrix
    cm = {}
    for cat in all_cats:
        cm[cat] = {}
        for cat2 in all_cats:
            cm[cat][cat2] = 0

    for a, b in zip(cats_a, cats_b):
        cm[a][b] += 1

    # Compute observed agreement
    po = sum(cm[c][c] for c in all_cats) / n

    # Compute expected agreement
    pe = 0
    for cat in all_cats:
        row_sum = sum(cm[cat][c] for c in all_cats)
        col_sum = sum(cm[c][cat] for c in all_cats)
        pe += (row_sum / n) * (col_sum / n)

    # Compute kappa
    if pe == 1:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1 - pe)

    return kappa, po, pe


def compute_correlation(scores_a, scores_b):
    """Compute Pearson correlation between two score arrays."""
    n = len(scores_a)
    mean_a = sum(scores_a) / n
    mean_b = sum(scores_b) / n

    cov = sum((a - mean_a) * (b - mean_b) for a, b in zip(scores_a, scores_b)) / n
    std_a = (sum((a - mean_a) ** 2 for a in scores_a) / n) ** 0.5
    std_b = (sum((b - mean_b) ** 2 for b in scores_b) / n) ** 0.5

    if std_a == 0 or std_b == 0:
        return 0.0

    return cov / (std_a * std_b)


def compute_agreement():
    """Compute agreement between human and LLM judge scores."""
    print("=" * 70)
    print("HUMAN-LLM JUDGE AGREEMENT")
    print("=" * 70)

    # Load human scores
    human_path = 'evaluation/human_scores.json'
    if not os.path.exists(human_path):
        print(f"\nERROR: {human_path} not found.")
        print("Run: python evaluation/human_review.py")
        return None

    with open(human_path, 'r', encoding='utf-8') as f:
        human_data = json.load(f)

    human_scores = human_data['scores']
    print(f"\nHuman scores: {len(human_scores)} examples")

    # Load LLM judge results
    judge_path = 'evaluation/judge_results.json'
    if not os.path.exists(judge_path):
        print(f"\nERROR: {judge_path} not found.")
        print("Run: python evaluation/llm_judge.py")
        return None

    with open(judge_path, 'r', encoding='utf-8') as f:
        judge_data = json.load(f)

    judge_results = {r['tweet_id']: r for r in judge_data['results']}
    print(f"LLM judge results: {len(judge_results)} examples")

    # Match examples
    matched_human = []
    matched_llm = []
    matched_ids = []

    for h in human_scores:
        tid = h['tweet_id']
        if tid in judge_results:
            j = judge_results[tid]
            # Use overall scores (average of 5 dimensions)
            matched_human.append(h['total_score'])
            matched_llm.append(j['total_score'])
            matched_ids.append(tid)

    print(f"Matched examples: {len(matched_human)}")

    if len(matched_human) < 10:
        print("\nERROR: Need at least 10 matched examples for meaningful agreement.")
        return None

    # Compute agreement
    kappa, po, pe = compute_cohen_kappa(matched_human, matched_llm)
    correlation = compute_correlation(matched_human, matched_llm)

    # Human average
    human_avg = sum(matched_human) / len(matched_human)
    llm_avg = sum(matched_llm) / len(matched_llm)

    # Per-dimension agreement
    dimensions = ['relevance', 'empathy', 'actionability', 'tone', 'grounding']
    dim_kappas = {}
    for dim in dimensions:
        h_dim = [h['scores'][dim] for h in human_scores if h['tweet_id'] in judge_results]
        l_dim = [judge_results[h['tweet_id']]['scores'][dim] for h in human_scores if h['tweet_id'] in judge_results]
        if len(h_dim) >= 10:
            k, _, _ = compute_cohen_kappa(h_dim, l_dim)
            dim_kappas[dim] = k

    # Report
    print("\n" + "=" * 70)
    print("AGREEMENT RESULTS")
    print("=" * 70)

    print(f"\nSample Size: {len(matched_human)} examples")
    print(f"\nScore Averages:")
    print(f"  Human avg score: {human_avg:.2f}/5.0")
    print(f"  LLM avg score:   {llm_avg:.2f}/5.0")

    print(f"\nCohen's Kappa (quadratic): {kappa:.4f}")
    if kappa >= 0.81:
        interpretation = "Almost perfect agreement"
    elif kappa >= 0.61:
        interpretation = "Substantial agreement"
    elif kappa >= 0.41:
        interpretation = "Moderate agreement"
    elif kappa >= 0.21:
        interpretation = "Fair agreement"
    else:
        interpretation = "Slight agreement"
    print(f"  Interpretation: {interpretation}")

    print(f"\nObserved Agreement (Po): {po:.4f}")
    print(f"Expected Agreement (Pe): {pe:.4f}")
    print(f"Pearson Correlation:      {correlation:.4f}")

    print(f"\nPer-Dimension Kappa:")
    for dim, k in sorted(dim_kappas.items()):
        print(f"  {dim:15s}: {k:.4f}")

    # Save agreement results
    agreement_results = {
        'sample_size': len(matched_human),
        'human_avg': human_avg,
        'llm_avg': llm_avg,
        'cohens_kappa': kappa,
        'observed_agreement': po,
        'expected_agreement': pe,
        'pearson_correlation': correlation,
        'interpretation': interpretation,
        'per_dimension_kappa': dim_kappas,
        'matched_ids': matched_ids
    }

    output_path = 'evaluation/agreement_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(agreement_results, f, indent=2)

    print(f"\nResults saved to {output_path}")
    return agreement_results


if __name__ == "__main__":
    compute_agreement()
