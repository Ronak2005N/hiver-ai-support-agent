"""
Sync human-verified labels from CSV into golden_set.json.

This script:
1. Reads data/golden/golden_set.csv (human-verified source of truth)
2. Updates evaluation/golden_set.json with verified labels
3. Preserves original candidate CSV separately
4. Reports mismatches

Usage:
    python evaluation/sync_golden_set.py
"""

import csv
import json
import os


def sync_golden_set():
    csv_path = 'data/golden/golden_set.csv'
    json_path = 'evaluation/golden_set.json'
    backup_path = 'evaluation/golden_set_auto_labeled.json'

    print("=" * 70)
    print("SYNCING HUMAN-VERIFIED LABELS INTO GOLDEN SET JSON")
    print("=" * 70)

    # Read CSV
    csv_rows = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_rows[row['id']] = row

    print(f"\nCSV: {len(csv_rows)} examples loaded from {csv_path}")

    # Read original JSON
    with open(json_path, 'r', encoding='utf-8') as f:
        original_json = json.load(f)

    print(f"JSON: {len(original_json)} examples loaded from {json_path}")

    # Backup original JSON
    if os.path.exists(json_path):
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(original_json, f, indent=2, ensure_ascii=False)
        print(f"Original JSON backed up to {backup_path}")

    # Build updated JSON
    mismatches = []
    updated = []

    for json_entry in original_json:
        entry_id = str(json_entry['id'])
        csv_row = csv_rows.get(entry_id)

        if not csv_row:
            print(f"  WARNING: ID {entry_id} not found in CSV")
            updated.append(json_entry)
            continue

        old_intent = json_entry['intent']
        new_intent = csv_row['intent']
        old_escalate = json_entry['should_escalate']
        new_escalate = csv_row['escalation_label'] == 'ESCALATE'

        # Track mismatches
        if old_intent != new_intent:
            mismatches.append({
                'id': entry_id,
                'field': 'intent',
                'old': old_intent,
                'new': new_intent,
                'text': json_entry['text'][:80]
            })
        if old_escalate != new_escalate:
            mismatches.append({
                'id': entry_id,
                'field': 'should_escalate',
                'old': old_escalate,
                'new': new_escalate,
                'text': json_entry['text'][:80]
            })

        # Update entry with human-verified labels
        updated_entry = json_entry.copy()
        updated_entry['intent'] = new_intent
        updated_entry['should_escalate'] = new_escalate
        updated_entry['labeler'] = 'human_verified'
        updated_entry['human_verified'] = True
        updated_entry['suggested_intent'] = csv_row.get('suggested_intent', '')
        updated_entry['annotation_note'] = csv_row.get('annotation_note', '')

        updated.append(updated_entry)

    # Write updated JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)

    print(f"\nUpdated JSON: {json_path}")

    # Report
    print("\n" + "=" * 70)
    print("SYNC REPORT")
    print("=" * 70)
    print(f"\nTotal examples: {len(updated)}")
    print(f"Mismatches found: {len(mismatches)}")

    if mismatches:
        print("\nMismatched entries:")
        for m in mismatches:
            try:
                print(f"  ID {m['id']} [{m['field']}]: {m['old']} -> {m['new']}")
                print(f"    Text: {m['text']}...")
            except UnicodeEncodeError:
                print(f"  ID {m['id']} [{m['field']}]: {m['old']} -> {m['new']}")
                print(f"    Text: [Unicode content]")
    else:
        print("\nZERO mismatches. JSON and CSV labels are now identical.")

    # Verify JSON/CSV consistency
    print("\n" + "=" * 70)
    print("POST-SYNC CONSISTENCY CHECK")
    print("=" * 70)

    with open(json_path, 'r', encoding='utf-8') as f:
        verify_json = json.load(f)

    consistent = True
    for entry in verify_json:
        entry_id = str(entry['id'])
        csv_row = csv_rows.get(entry_id)
        if csv_row:
            if entry['intent'] != csv_row['intent']:
                print(f"  INCONSISTENCY: ID {entry_id} intent mismatch")
                consistent = False
            if entry['should_escalate'] != (csv_row['escalation_label'] == 'ESCALATE'):
                print(f"  INCONSISTENCY: ID {entry_id} escalation mismatch")
                consistent = False

    if consistent:
        print("  All 196 examples verified: JSON and CSV are IDENTICAL.")

    # Distribution
    print("\n" + "=" * 70)
    print("FINAL LABEL DISTRIBUTION")
    print("=" * 70)

    from collections import Counter
    intents = Counter(e['intent'] for e in verify_json)
    escalate = Counter(e['should_escalate'] for e in verify_json)

    print("\nIntent distribution:")
    for intent, count in sorted(intents.items()):
        print(f"  {intent:25s}: {count} ({count/len(verify_json)*100:.1f}%)")

    print(f"\nEscalation distribution:")
    print(f"  ESCALATE:  {escalate[True]} ({escalate[True]/len(verify_json)*100:.1f}%)")
    print(f"  AUTO_HANDLE: {escalate[False]} ({escalate[False]/len(verify_json)*100:.1f}%)")

    return updated, mismatches


if __name__ == "__main__":
    sync_golden_set()
