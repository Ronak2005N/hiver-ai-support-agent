"""Show verification status and quality checks."""
import json
import csv
import os
from collections import Counter

PROGRESS_PATH = 'data/golden/verification_progress.json'
CANDIDATE_PATH = 'data/golden/golden_candidate.csv'
VERIFIED_PATH = 'data/golden/golden_set.csv'


def load_progress():
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, 'r') as f:
            return json.load(f)
    return {'verified': {}, 'current_id': 1, 'notes': {}}


def load_candidate():
    examples = []
    with open(CANDIDATE_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append(row)
    return examples


def main():
    print("=" * 70)
    print("GOLDEN SET VERIFICATION STATUS")
    print("=" * 70)
    
    progress = load_progress()
    examples = load_candidate()
    total = len(examples)
    verified_count = len(progress.get('verified', {}))
    
    print(f"\nTotal examples: {total}")
    print(f"Verified: {verified_count}/{total} ({verified_count/total*100:.1f}%)")
    print(f"Remaining: {total - verified_count}")
    
    if verified_count > 0:
        # Count intents
        intents = [v.get('intent', 'unknown') for v in progress['verified'].values()]
        escalations = [v.get('escalation', 'unknown') for v in progress['verified'].values()]
        notes = sum(1 for ex_id, note in progress.get('notes', {}).items() if note)
        
        print("\nIntent distribution (verified):")
        for intent, count in Counter(intents).most_common():
            print(f"  {intent}: {count}")
        
        print("\nEscalation distribution (verified):")
        for esc, count in Counter(escalations).most_common():
            print(f"  {esc}: {count}")
        
        print(f"\nExamples with annotation notes: {notes}")
        
        # Check for changed labels
        candidate_intents = {ex['id']: ex['suggested_intent'] for ex in examples}
        changed = 0
        for ex_id, v in progress['verified'].items():
            if v.get('intent') != candidate_intents.get(ex_id):
                changed += 1
        
        print(f"\nLabels changed from auto-suggestion: {changed}")
    
    # Check if verified file exists
    if os.path.exists(VERIFIED_PATH):
        verified_examples = []
        with open(VERIFIED_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                verified_examples.append(row)
        
        human_verified = sum(1 for ex in verified_examples if ex.get('human_verified') == 'true')
        print(f"\nVerified file exists: {VERIFIED_PATH}")
        print(f"Human verified in file: {human_verified}/{len(verified_examples)}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
