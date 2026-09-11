"""
Historical Retriever Component
Finds similar historical support conversations.

Usage:
    from src.retriever import HistoricalRetriever
    retriever = HistoricalRetriever()
    similar = retriever.retrieve("Where is my order?", "order_status", k=3)
"""

import pandas as pd
import os


class HistoricalRetriever:
    """Retrieves similar historical support conversations."""

    def __init__(self, data_path='data/amazon_customers.csv'):
        """Load dataset for retrieval."""
        self.df = None
        
        if os.path.exists(data_path):
            self.df = pd.read_csv(data_path)
            print(f"Loaded {len(self.df):,} tweets for retrieval")
        else:
            print(f"WARNING: No dataset found at {data_path}")

    def _get_keywords_for_intent(self, intent):
        """Get keywords for intent filtering."""
        keywords = {
            'order_status': 'order|package|delivery|shipping',
            'technical_support': 'app|website|login|error|crash',
            'refund_return': 'refund|money back|return',
            'billing_payment': 'charge|payment|billing|price',
            'product_issue': 'product|item|broken|damaged',
            'cancellation': 'cancel|subscription|membership',
            'complaint_frustration': 'terrible|worst|angry|frustrated'
        }
        return keywords.get(intent, 'help')

    def retrieve(self, tweet, intent, k=3):
        """Find similar past conversations."""
        if self.df is None:
            return []
        
        # Filter by same intent first
        same_intent = self.df[self.df['text'].str.contains(
            self._get_keywords_for_intent(intent),
            case=False,
            na=False
        )].head(100)

        if len(same_intent) < k:
            # Fall back to full dataset
            same_intent = self.df.head(1000)

        # Simple keyword-based similarity
        tweet_words = set(tweet.lower().split())
        similarities = []

        for idx, row in same_intent.iterrows():
            text = str(row['text']).lower()
            text_words = set(text.split())
            overlap = len(tweet_words.intersection(text_words))
            similarities.append((idx, overlap, row['text']))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return [s[2] for s in similarities[:k]]
