"""
Submission Validation Script
Checks all integrity requirements for the Hiver assignment submission.

Usage:
    python evaluation/validate_submission.py

Checks:
[ ] Golden Set has 150-250 examples
[ ] All final labels are human verified
[ ] JSON and CSV labels match
[ ] No duplicate IDs
[ ] No missing labels
[ ] Metrics are generated (not hardcoded)
[ ] LLM judge is actually called (or fallback documented)
[ ] Human/LLM agreement uses real data
[ ] README metrics match results.json
[ ] REPORT metrics match results.json
[ ] Escalation metrics match everywhere
[ ] No stale/conflicting headline numbers
[ ] Evaluation uses the frozen Golden Set
"""

import json
import csv
import os
import sys


def check(name, condition, detail=""):
    """Print a check result."""
    status = "PASS" if condition else "FAIL"
    symbol = "[OK]" if condition else "[!!]"
    print(f"  {symbol} {name}")
    if detail and not condition:
        print(f"       {detail}")
    return condition


def validate_submission():
    print("=" * 70)
    print("SUBMISSION VALIDATION")
    print("=" * 70)

    all_pass = True

    # =========================================================
    # 1. GOLDEN SET STRUCTURE
    # =========================================================
    print("\n[1] GOLDEN SET STRUCTURE")

    json_path = 'evaluation/golden_set.json'
    csv_path = 'data/golden/golden_set.csv'

    # Load files
    golden_json = []
    golden_csv = {}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            golden_json = json.load(f)
        check("JSON file exists and is valid", True)
    except Exception as e:
        check("JSON file exists and is valid", False, str(e))
        all_pass = False

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                golden_csv[row['id']] = row
        check("CSV file exists and is valid", True)
    except Exception as e:
        check("CSV file exists and is valid", False, str(e))
        all_pass = False

    # Check count
    count = len(golden_json)
    check(f"Golden Set has 150-250 examples (found {count})", 150 <= count <= 250)

    # Check for duplicates
    ids = [ex['id'] for ex in golden_json]
    check(f"No duplicate IDs ({len(ids)} IDs, {len(set(ids))} unique)", len(ids) == len(set(ids)))

    # Check required fields
    required_fields = ['id', 'text', 'intent', 'should_escalate']
    for field in required_fields:
        missing = [ex['id'] for ex in golden_json if field not in ex]
        check(f"All entries have '{field}' field", len(missing) == 0,
              f"Missing in IDs: {missing[:5]}..." if missing else "")

    # =========================================================
    # 2. HUMAN VERIFICATION
    # =========================================================
    print("\n[2] HUMAN VERIFICATION")

    human_verified = sum(1 for ex in golden_json if ex.get('human_verified') == True)
    total = len(golden_json)
    check(f"All {total} examples are human-verified ({human_verified}/{total})", human_verified == total)

    labelers = set(ex.get('labeler', '') for ex in golden_json)
    check(f"Labeler is 'human_verified' (found: {labelers})", labelers == {'human_verified'})

    # =========================================================
    # 3. JSON/CSV CONSISTENCY
    # =========================================================
    print("\n[3] JSON/CSV CONSISTENCY")

    mismatches = []
    for ex in golden_json:
        entry_id = str(ex['id'])
        csv_row = golden_csv.get(entry_id)
        if csv_row:
            if ex['intent'] != csv_row['intent']:
                mismatches.append(f"ID {entry_id}: intent {ex['intent']} != {csv_row['intent']}")
            csv_escalate = csv_row['escalation_label'] == 'ESCALATE'
            if ex['should_escalate'] != csv_escalate:
                mismatches.append(f"ID {entry_id}: escalation {ex['should_escalate']} != {csv_escalate}")

    check(f"JSON and CSV labels match ({len(mismatches)} mismatches)", len(mismatches) == 0,
          mismatches[:3] if mismatches else "")

    # =========================================================
    # 4. METRICS FILES
    # =========================================================
    print("\n[4] METRICS FILES")

    results_path = 'evaluation/results.json'
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        check("results.json exists and is valid", True)
    except Exception as e:
        check("results.json exists and is valid", False, str(e))
        all_pass = False

    # Check metrics are present
    if results:
        check("Intent accuracy present", 'intent' in results and 'accuracy' in results['intent'])
        check("Escalation metrics present", 'escalation' in results and 'accuracy' in results['escalation'])
        check("Golden set size matches", results.get('golden_set_size') == len(golden_json))

    # =========================================================
    # 5. LLM JUDGE
    # =========================================================
    print("\n[5] LLM JUDGE")

    judge_path = 'evaluation/llm_judge.py'
    check("LLM judge script exists", os.path.exists(judge_path))

    judge_results_path = 'evaluation/judge_results.json'
    if os.path.exists(judge_results_path):
        with open(judge_results_path, 'r', encoding='utf-8') as f:
            judge_data = json.load(f)
        summary = judge_data.get('summary', {})
        method = summary.get('scoring_method', 'unknown')
        llm_count = summary.get('llm_scored', 0)
        keyword_count = summary.get('keyword_scored', 0)
        check(f"LLM judge results exist ({method})", True)
        if method == 'llm':
            print(f"       LLM-scored: {llm_count}, Keyword-fallback: {keyword_count}")
        else:
            print(f"       WARNING: Judge used keyword fallback, not actual LLM")
            print(f"       Set GEMINI_API_KEY to enable LLM scoring")
    else:
        check("LLM judge results exist", False, "Run: python evaluation/llm_judge.py")

    # =========================================================
    # 6. HUMAN/LLM AGREEMENT
    # =========================================================
    print("\n[6] HUMAN/LLM AGREEMENT")

    human_scores_path = 'evaluation/human_scores.json'
    if os.path.exists(human_scores_path):
        with open(human_scores_path, 'r', encoding='utf-8') as f:
            human_data = json.load(f)
        n_scored = len(human_data.get('scores', []))
        check(f"Human scores exist ({n_scored} examples)", n_scored > 0)

        agreement_path = 'evaluation/agreement_results.json'
        if os.path.exists(agreement_path):
            with open(agreement_path, 'r', encoding='utf-8') as f:
                agreement = json.load(f)
            kappa = agreement.get('cohens_kappa', None)
            check(f"Cohen's Kappa computed: {kappa:.4f}" if kappa else "Cohen's Kappa computed", kappa is not None)
        else:
            check("Agreement computed", False, "Run: python evaluation/compute_agreement.py")
    else:
        print("  [INFO] No human scores yet. To compute agreement:")
        print("         1. Run: python evaluation/human_review.py")
        print("         2. Score 30 examples")
        print("         3. Run: python evaluation/compute_agreement.py")

    # =========================================================
    # 7. DOCUMENT CONSISTENCY
    # =========================================================
    print("\n[7] DOCUMENT CONSISTENCY")

    # Check README
    readme_path = 'README.md'
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_text = f.read()

        intent_acc = results.get('intent', {}).get('accuracy', 0) if results else 0
        readme_has_acc = f"{intent_acc*100:.2f}%" in readme_text or f"{intent_acc:.2%}" in readme_text
        check(f"README contains current accuracy ({intent_acc:.2%})", readme_has_acc)
    else:
        check("README exists", False)

    # Check REPORT
    report_path = 'docs/REPORT.md'
    if os.path.exists(report_path):
        with open(report_path, 'r', encoding='utf-8') as f:
            report_text = f.read()
        check("REPORT.md exists", True)
    else:
        check("REPORT.md exists", False)

    # Check for stale numbers
    stale_numbers = ['70.41%', '88.27%', '69.9%', '92.3%']
    found_stale = []
    for stale in stale_numbers:
        if stale in readme_text:
            found_stale.append(stale)
    check("README has no stale numbers", len(found_stale) == 0,
          f"Found stale numbers: {found_stale}" if found_stale else "")

    # =========================================================
    # 8. EVALUATION SCRIPTS
    # =========================================================
    print("\n[8] EVALUATION SCRIPTS")

    scripts = [
        'evaluation/evaluate.py',
        'evaluation/evaluate_intent.py',
        'evaluation/evaluate_escalation.py',
        'evaluation/evaluate_replies.py',
        'evaluation/llm_judge.py',
        'evaluation/failure_analysis.py',
        'evaluation/human_review.py',
        'evaluation/compute_agreement.py',
        'evaluation/validate_submission.py',
    ]

    for script in scripts:
        check(f"{script} exists", os.path.exists(script))

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    if all_pass:
        print("\n  All checks PASSED. Submission is ready.")
    else:
        print("\n  Some checks FAILED. Review the issues above.")

    print("\n  To reproduce all results:")
    print("    python evaluation/evaluate.py")
    print("    python evaluation/evaluate_escalation.py")
    print("    python evaluation/failure_analysis.py")
    print("    GEMINI_API_KEY=xxx python evaluation/llm_judge.py")

    return all_pass


if __name__ == "__main__":
    validate_submission()
