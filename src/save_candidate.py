"""Save candidate Golden Set and prepare for human verification."""
import json
import csv
import os

# Ensure directories exist
os.makedirs('data/golden', exist_ok=True)

# Load current Golden Set
with open('evaluation/golden_set.json', 'r') as f:
    data = json.load(f)

# Save as candidate (preserve original)
headers = ['id', 'text', 'suggested_intent', 'difficulty', 'should_escalate', 'sampling_method', 'labeler']
with open('data/golden/golden_candidate.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    for ex in data:
        writer.writerow({
            'id': ex['id'],
            'text': ex['text'],
            'suggested_intent': ex['intent'],
            'difficulty': ex['difficulty'],
            'should_escalate': ex['should_escalate'],
            'sampling_method': ex.get('sampling_method', 'unknown'),
            'labeler': ex.get('labeler', 'auto_keyword_rules')
        })

print('Saved candidate set to data/golden/golden_candidate.csv')
print('Total examples:', len(data))
