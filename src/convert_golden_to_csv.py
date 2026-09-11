"""Convert Golden Set JSON to CSV format."""
import json
import csv
import os

# Ensure directory exists
os.makedirs('data/golden', exist_ok=True)

with open('evaluation/golden_set.json', 'r') as f:
    data = json.load(f)

# Create CSV with required columns
headers = ['id', 'text', 'intent', 'escalation_label', 'difficulty', 'annotation_note']

with open('data/golden/golden_set.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    
    for ex in data:
        sampling = ex.get('sampling_method', 'unknown')
        note = 'Auto-labeled by keyword rules. Sampling: ' + sampling
        writer.writerow({
            'id': ex['id'],
            'text': ex['text'],
            'intent': ex['intent'],
            'escalation_label': 'escalate' if ex['should_escalate'] else 'auto_handle',
            'difficulty': ex['difficulty'],
            'annotation_note': note
        })

print('Created data/golden/golden_set.csv')
print('Total rows:', len(data))
