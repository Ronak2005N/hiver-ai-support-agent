"""
Full Customer Support Agent
Combines all components for interactive use.

Usage:
    python src/agent.py
"""

import sys
import os
sys.path.insert(0, 'src')

from pipeline import CustomerSupportPipeline


def main():
    """Interactive agent for customer support."""
    print("=" * 60)
    print("CUSTOMER SUPPORT AGENT")
    print("=" * 60)

    # Check for Gemini key
    gemini_key = os.environ.get('GEMINI_API_KEY')

    # Initialize pipeline
    pipeline = CustomerSupportPipeline(gemini_key=gemini_key)

    print("\nEnter customer messages (type 'quit' to exit):\n")

    while True:
        try:
            tweet = input("Customer message: ").strip()
            
            if tweet.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not tweet:
                continue

            result = pipeline.process(tweet)

            print("\nIntent:")
            print(" ", result['intent'])

            print("\nConfidence:")
            print(" ", f"{result['confidence']:.2%}")

            print("\nRetrieved evidence:")
            for i, similar in enumerate(result['similar_tweets'][:3], 1):
                print(f"  {i}. {similar[:80]}...")

            print("\nSuggested reply:")
            print(" ", result['reply'])

            print("\nDecision:")
            if result['should_escalate']:
                print("  ESCALATE")
                print("\nReason:")
                print(" ", result['escalation_reason'])
            else:
                print("  AUTO-HANDLE")

            print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()
