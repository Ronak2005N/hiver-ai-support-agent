"""
Customer Support Pipeline
Combines all components: classifier, retriever, generator, escalation.

Usage:
    python src/pipeline.py
"""

import sys
import os
sys.path.insert(0, 'src')

from classifier import IntentClassifier
from retriever import HistoricalRetriever
from generator import ReplyGenerator
from escalation import EscalationDecider


class CustomerSupportPipeline:
    """Main pipeline that combines all components."""

    def __init__(self, gemini_key=None):
        """Initialize all components."""
        print("Initializing Customer Support Pipeline...")
        
        self.classifier = IntentClassifier()
        print("  Classifier loaded")
        
        self.retriever = HistoricalRetriever()
        
        self.generator = ReplyGenerator(gemini_key=gemini_key)
        
        self.escalation = EscalationDecider()
        
        print("Pipeline ready!\n")

    def process(self, tweet):
        """Process a single tweet through all steps."""
        # Step 1: Classify
        intent, confidence, source = self.classifier.classify(tweet)

        # Step 2: Retrieve similar
        similar = self.retriever.retrieve(tweet, intent)

        # Step 3: Generate reply
        reply = self.generator.generate(tweet, intent, similar)

        # Step 4: Escalation decision
        should_escalate, reason = self.escalation.decide(tweet, intent, confidence)

        return {
            'tweet': tweet,
            'intent': intent,
            'confidence': confidence,
            'source': source,
            'reply': reply,
            'should_escalate': should_escalate,
            'escalation_reason': reason,
            'similar_tweets': similar
        }


def main():
    """Run the pipeline on test tweets."""
    print("=" * 60)
    print("CUSTOMER SUPPORT PIPELINE")
    print("=" * 60)

    # Check for Gemini key
    gemini_key = os.environ.get('GEMINI_API_KEY')

    # Initialize pipeline
    pipeline = CustomerSupportPipeline(gemini_key=gemini_key)

    # Test tweets
    test_tweets = [
        "Where is my order? It's been 2 weeks and I haven't received anything",
        "The app keeps crashing every time I try to open it",
        "I was charged twice for my Prime membership",
        "I want a refund for the broken product I received",
        "How do I cancel my subscription?",
        "This is the worst customer service ever! I'm never buying from Amazon again",
        "My package says delivered but I never got it",
        "I received the wrong item in my order"
    ]

    print("\nProcessing test tweets...\n")

    for tweet in test_tweets:
        result = pipeline.process(tweet)

        print("Tweet:", tweet[:80])
        print("  Intent:", result['intent'], "(confidence:", result['confidence'], ", source:", result['source'], ")")
        print("  Reply:", result['reply'][:100])
        print("  Escalate:", "YES" if result['should_escalate'] else "NO")
        if result['should_escalate']:
            print("  Reason:", result['escalation_reason'])
        print("-" * 60)


if __name__ == "__main__":
    main()
