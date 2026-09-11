"""
Golden Set Human Verification Tool
CLI interface for manually verifying Golden Set labels.

Usage:
    python src/verify_golden.py

Commands during review:
    1-7     Select intent (see list below)
    a       Auto-handle
    e       Escalate
    s       Skip to next (no changes)
    n       Add annotation note
    q       Quit (progress saved)
    p       Print progress
    j       Jump to example by ID

Intents:
    1. order_status
    2. refund_return
    3. billing_payment
    4. technical_support
    5. product_issue
    6. cancellation
    7. complaint_frustration
"""

import json
import csv
import os
import sys

# Paths
CANDIDATE_PATH = 'data/golden/golden_candidate.csv'
VERIFIED_PATH = 'data/golden/golden_set.csv'
PROGRESS_PATH = 'data/golden/verification_progress.json'

INTENTS = [
    'order_status',
    'refund_return',
    'billing_payment',
    'technical_support',
    'product_issue',
    'cancellation',
    'complaint_frustration'
]

INTENT_MAP = {
    '1': 'order_status',
    '2': 'refund_return',
    '3': 'billing_payment',
    '4': 'technical_support',
    '5': 'product_issue',
    '6': 'cancellation',
    '7': 'complaint_frustration'
}

ESCALATION_MAP = {
    'a': 'AUTO_HANDLE',
    'e': 'ESCALATE'
}


def load_progress():
    """Load verification progress."""
    if os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, 'r') as f:
            return json.load(f)
    return {'verified': {}, 'current_id': 1, 'notes': {}}


def save_progress(progress):
    """Save verification progress."""
    with open(PROGRESS_PATH, 'w') as f:
        json.dump(progress, f, indent=2)


def load_candidate():
    """Load candidate Golden Set."""
    examples = []
    with open(CANDIDATE_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            examples.append(row)
    return examples


def save_verified(examples, progress):
    """Save verified Golden Set."""
    headers = ['id', 'text', 'intent', 'escalation_label', 'difficulty', 
               'annotation_note', 'human_verified', 'suggested_intent']
    
    with open(VERIFIED_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        
        for ex in examples:
            ex_id = ex['id']
            verified = progress['verified'].get(ex_id, {})
            
            writer.writerow({
                'id': ex_id,
                'text': ex['text'],
                'intent': verified.get('intent', ex['suggested_intent']),
                'escalation_label': verified.get('escalation', 'AUTO_HANDLE'),
                'difficulty': ex['difficulty'],
                'annotation_note': progress.get('notes', {}).get(ex_id, ''),
                'human_verified': 'true' if ex_id in progress['verified'] else 'false',
                'suggested_intent': ex['suggested_intent']
            })


def print_example(ex, progress):
    """Print current example with suggestions."""
    ex_id = ex['id']
    verified = progress['verified'].get(ex_id, {})
    note = progress.get('notes', {}).get(ex_id, '')
    
    print("\n" + "=" * 70)
    print(f"EXAMPLE {ex_id} of 196")
    print("=" * 70)
    
    # Show customer text
    print("\nCustomer text:")
    print(f"  {ex['text']}")
    
    # Show suggested intent
    print(f"\nSuggested intent: {ex['suggested_intent']}")
    
    # Show current selection if verified
    if ex_id in progress['verified']:
        print(f"\nCURRENT SELECTION (already verified):")
        print(f"  Intent: {verified.get('intent', ex['suggested_intent'])}")
        print(f"  Escalation: {verified.get('escalation', 'AUTO_HANDLE')}")
        if note:
            print(f"  Note: {note}")
    
    # Show intent options
    print("\nSelect final human intent:")
    for i, intent in enumerate(INTENTS, 1):
        marker = " <-- suggested" if intent == ex['suggested_intent'] else ""
        print(f"  {i}. {intent}{marker}")
    
    # Show escalation options
    print("\nEscalation decision:")
    print("  a. AUTO_HANDLE")
    print("  e. ESCALATE")
    
    print("\nCommands:")
    print("  1-7  = Select intent")
    print("  a    = Auto-handle")
    print("  e    = Escalate")
    print("  s    = Skip (no changes)")
    print("  n    = Add annotation note")
    print("  q    = Quit (progress saved)")
    print("  p    = Print progress")
    print("  j    = Jump to example by ID")


def print_progress(progress, total):
    """Print verification progress."""
    verified_count = len(progress['verified'])
    remaining = total - verified_count
    
    print("\n" + "=" * 70)
    print("VERIFICATION PROGRESS")
    print("=" * 70)
    print(f"\nVerified: {verified_count}/{total} ({verified_count/total*100:.1f}%)")
    print(f"Remaining: {remaining}")
    
    if verified_count > 0:
        # Count intents
        from collections import Counter
        intents = [v.get('intent', 'unknown') for v in progress['verified'].values()]
        escalations = [v.get('escalation', 'unknown') for v in progress['verified'].values()]
        
        print("\nIntent distribution (verified):")
        for intent, count in Counter(intents).most_common():
            print(f"  {intent}: {count}")
        
        print("\nEscalation distribution (verified):")
        for esc, count in Counter(escalations).most_common():
            print(f"  {esc}: {count}")
    
    print("=" * 70)


def main():
    """Main verification loop."""
    print("=" * 70)
    print("GOLDEN SET HUMAN VERIFICATION TOOL")
    print("=" * 70)
    
    # Load data
    examples = load_candidate()
    progress = load_progress()
    
    total = len(examples)
    print(f"\nLoaded {total} examples from candidate set")
    print(f"Previously verified: {len(progress['verified'])}")
    
    # Create ID-to-index mapping
    id_to_index = {ex['id']: i for i, ex in enumerate(examples)}
    
    # Start from last position or beginning
    current_id = progress.get('current_id', examples[0]['id'])
    if current_id not in id_to_index:
        current_id = examples[0]['id']
    
    current_index = id_to_index[current_id]
    
    while True:
        # Get current example
        ex = examples[current_index]
        ex_id = ex['id']
        
        # Print example
        print_example(ex, progress)
        
        # Get user input
        try:
            cmd = input("\nYour choice: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n\nProgress saved. You can resume later.")
            progress['current_id'] = ex_id
            save_progress(progress)
            save_verified(examples, progress)
            return
        
        # Handle commands
        if cmd == 'q':
            # Quit
            progress['current_id'] = ex_id
            save_progress(progress)
            save_verified(examples, progress)
            print("\nProgress saved. Run this script again to resume.")
            return
        
        elif cmd == 'p':
            # Print progress
            print_progress(progress, total)
        
        elif cmd == 's':
            # Skip
            current_index = (current_index + 1) % total
            progress['current_id'] = examples[current_index]['id']
            save_progress(progress)
        
        elif cmd == 'n':
            # Add note
            note = input("Enter annotation note: ").strip()
            progress.setdefault('notes', {})[ex_id] = note
            save_progress(progress)
            print("Note saved.")
        
        elif cmd == 'j':
            # Jump to ID
            try:
                target_id = int(input("Enter example ID (1-196): "))
                if target_id in id_to_index:
                    current_index = id_to_index[target_id]
                    progress['current_id'] = target_id
                else:
                    print(f"Invalid ID. Must be 1-{total}")
            except ValueError:
                print("Invalid input. Enter a number.")
        
        elif cmd in INTENT_MAP:
            # Select intent
            intent = INTENT_MAP[cmd]
            
            # Also need escalation
            esc_input = input("Escalation (a=auto, e=escalate): ").strip().lower()
            escalation = ESCALATION_MAP.get(esc_input, 'AUTO_HANDLE')
            
            # Save
            progress.setdefault('verified', {})[ex_id] = {
                'intent': intent,
                'escalation': escalation
            }
            progress['current_id'] = ex_id
            
            save_progress(progress)
            save_verified(examples, progress)
            
            print(f"\nSaved: intent={intent}, escalation={escalation}")
            
            # Move to next
            current_index = (current_index + 1) % total
            progress['current_id'] = examples[current_index]['id']
        
        elif cmd in ESCALATION_MAP:
            # Just escalation
            escalation = ESCALATION_MAP[cmd]
            
            # Get current intent or use suggested
            current_intent = progress.get('verified', {}).get(ex_id, {}).get('intent', ex['suggested_intent'])
            
            progress.setdefault('verified', {})[ex_id] = {
                'intent': current_intent,
                'escalation': escalation
            }
            progress['current_id'] = ex_id
            
            save_progress(progress)
            save_verified(examples, progress)
            
            print(f"\nSaved: intent={current_intent}, escalation={escalation}")
            
            # Move to next
            current_index = (current_index + 1) % total
            progress['current_id'] = examples[current_index]['id']
        
        else:
            print("Invalid command. Try again.")


if __name__ == "__main__":
    main()
