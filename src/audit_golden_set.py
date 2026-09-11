"""Audit the Golden Set."""
import json
from collections import Counter

with open('evaluation/golden_set.json', 'r') as f:
    data = json.load(f)

print("Golden Set Audit")
print("=" * 60)
print("Total examples:", len(data))
print()

# Intent distribution
intents = [d['intent'] for d in data]
print("Intent distribution:")
for intent, count in Counter(intents).most_common():
    pct = count / len(data) * 100
    print(f"  {intent}: {count} ({pct:.1f}%)")
print()

# Difficulty distribution
diffs = [d['difficulty'] for d in data]
print("Difficulty distribution:")
for diff, count in Counter(diffs).most_common():
    pct = count / len(data) * 100
    print(f"  {diff}: {count} ({pct:.1f}%)")
print()

# Escalation distribution
escs = [d['should_escalate'] for d in data]
print("Escalation distribution:")
auto_count = escs.count(False)
esc_count = escs.count(True)
print(f"  Auto-handle: {auto_count} ({auto_count/len(data)*100:.1f}%)")
print(f"  Escalate: {esc_count} ({esc_count/len(data)*100:.1f}%)")
print()

# Check labeler field
labelers = [d.get('labeler', 'unknown') for d in data]
print("Labeler distribution:")
for labeler, count in Counter(labelers).most_common():
    print(f"  {labeler}: {count}")
print()

# Show first 3 examples
print("First 3 examples:")
for ex in data[:3]:
    ex_id = ex['id']
    text = ex['text'][:80]
    intent = ex['intent']
    difficulty = ex['difficulty']
    escalate = ex['should_escalate']
    print(f"  ID: {ex_id}")
    print(f"  Text: {text}...")
    print(f"  Intent: {intent}")
    print(f"  Difficulty: {difficulty}")
    print(f"  Escalate: {escalate}")
    print()

# Check for missing fields
required_fields = ['id', 'text', 'intent', 'difficulty', 'should_escalate']
missing = []
for ex in data:
    for field in required_fields:
        if field not in ex:
            missing.append((ex.get('id', 'unknown'), field))

if missing:
    print("MISSING FIELDS:")
    for ex_id, field in missing[:5]:
        print(f"  Example {ex_id}: missing {field}")
else:
    print("All examples have required fields: OK")
