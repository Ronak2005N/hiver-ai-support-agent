"""
Reply Generation Evaluation
Evaluates quality of generated replies.

Usage:
    python evaluation/evaluate_replies.py
"""

import json
import sys
import os

sys.path.insert(0, 'src')
from hybrid_classifier import HybridClassifier


# Template replies (same as agent)
TEMPLATE_REPLIES = {
    'order_status': "Hi! I'm sorry for the delay. Let me check your order status. Can you share your order number?",
    'refund_return': "I understand you'd like a refund. Let me help you with that. Please share your order details.",
    'billing_payment': "I see there's a billing concern. Let me look into this for you right away.",
    'technical_support': "I'm sorry you're having technical issues. Let me help troubleshoot this.",
    'product_issue': "I apologize for the product issue. Let me help resolve this for you.",
    'cancellation': "I can help with your cancellation request. Let me look into your account.",
    'complaint_frustration': "I sincerely apologize for this experience. This is not up to our standards. Let me make this right."
}


def load_golden_set(path='evaluation/golden_set.json'):
    """Load the Golden Set."""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_replies(golden_set_path='evaluation/golden_set.json'):
    """Evaluate reply generation quality."""
    print("=" * 70)
    print("REPLY GENERATION EVALUATION")
    print("=" * 70)
    
    # Load data
    golden_set = load_golden_set(golden_set_path)
    print(f"\nGolden Set: {len(golden_set)} examples")
    
    # Initialize classifier
    classifier = HybridClassifier()
    print("Classifier: hybrid (rules + ML)")
    print("Generator: template-based")
    
    # Run evaluation
    results = []
    
    for ex in golden_set:
        text = ex['text']
        true_intent = ex['intent']
        
        # Classify
        pred_intent, confidence, source = classifier.classify(text)
        
        # Generate reply
        reply = TEMPLATE_REPLIES.get(pred_intent, "Hi! I'm here to help. Can you tell me more about your issue?")
        
        # Simple quality checks
        relevance = pred_intent == true_intent
        has_greeting = reply.startswith('Hi') or reply.startswith('I')
        has_action = 'help' in reply.lower() or 'let me' in reply.lower()
        is_concise = len(reply.split()) <= 30
        
        results.append({
            'id': ex['id'],
            'text': text[:100],
            'true_intent': true_intent,
            'pred_intent': pred_intent,
            'reply': reply,
            'relevance': relevance,
            'has_greeting': has_greeting,
            'has_action': has_action,
            'is_concise': is_concise,
            'confidence': confidence
        })
    
    # Calculate metrics
    relevance_rate = sum(1 for r in results if r['relevance']) / len(results)
    greeting_rate = sum(1 for r in results if r['has_greeting']) / len(results)
    action_rate = sum(1 for r in results if r['has_action']) / len(results)
    concise_rate = sum(1 for r in results if r['is_concise']) / len(results)
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nRelevance (intent matches): {relevance_rate:.4f} ({relevance_rate*100:.2f}%)")
    print(f"Greeting present:           {greeting_rate:.4f} ({greeting_rate*100:.2f}%)")
    print(f"Action present:             {action_rate:.4f} ({action_rate*100:.2f}%)")
    print(f"Concise (<30 words):        {concise_rate:.4f} ({concise_rate*100:.2f}%)")
    
    # Sample replies
    print("\n" + "=" * 70)
    print("SAMPLE REPLIES")
    print("=" * 70)
    
    for intent in sorted(set(r['true_intent'] for r in results)):
        examples = [r for r in results if r['true_intent'] == intent][:2]
        print(f"\n[{intent.upper()}]")
        for ex in examples:
            match = "OK" if ex['relevance'] else "WRONG"
            print(f"  Tweet: {ex['text'][:60]}...")
            print(f"  Reply: {ex['reply'][:80]}...")
            print(f"  [{match}]")
    
    # Save results
    output_results = {
        'relevance_rate': relevance_rate,
        'greeting_rate': greeting_rate,
        'action_rate': action_rate,
        'concise_rate': concise_rate,
        'total_examples': len(results),
        'sample_replies': results[:20]
    }
    
    output_path = 'evaluation/reply_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return output_results


if __name__ == "__main__":
    evaluate_replies()
