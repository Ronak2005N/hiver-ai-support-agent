"""
DATA EXPLORATION NOTEBOOK
Run this to understand the dataset.

Usage:
    python notebooks/exploration.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def main():
    # Load dataset
    data_path = "data/twcs/twcs.csv"

    if not os.path.exists(data_path):
        print("Dataset not found!")
        print("Please download it first:")
        print("  Option 1: kaggle datasets download -d thoughtvector/customer-support-on-twitter")
        print("  Option 2: Download manually from https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter")
        return

    print("Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} tweets\n")

    # Basic stats
    print("="*60)
    print("BASIC STATISTICS")
    print("="*60)
    print(f"\nTotal tweets: {len(df):,}")
    print(f"Unique authors: {df['author_id'].nunique():,}")
    print(f"Date range: {df['created_at'].min()} to {df['created_at'].max()}")

    # Inbound vs Outbound
    print("\n" + "="*60)
    print("INBOUND vs OUTBOUND")
    print("="*60)
    inbound_counts = df['inbound'].value_counts()
    print(f"\nInbound (customer to company): {inbound_counts.get(True, 0):,}")
    print(f"Outbound (company reply): {inbound_counts.get(False, 0):,}")

    # Top brands
    print("\n" + "="*60)
    print("TOP 15 BRANDS (by tweet count)")
    print("="*60)
    top_brands = df['author_id'].value_counts().head(15)
    for brand, count in top_brands.items():
        print(f"  @{brand}: {count:,} tweets")

    # Sample tweets
    print("\n" + "="*60)
    print("SAMPLE INBOUND TWEETS (customer questions)")
    print("="*60)
    inbound_tweets = df[df['inbound'] == True]['text'].head(10)
    for i, tweet in enumerate(inbound_tweets, 1):
        print(f"\n{i}. {tweet[:200]}...")

    # Tweet length
    print("\n" + "="*60)
    print("TWEET LENGTH STATISTICS")
    print("="*60)
    df['text_length'] = df['text'].str.len()
    print(f"Average length: {df['text_length'].mean():.0f} characters")
    print(f"Min length: {df['text_length'].min()}")
    print(f"Max length: {df['text_length'].max()}")

    # Ask user which brand to explore
    print("\n" + "="*60)
    print("NEXT STEP: Pick a brand")
    print("="*60)
    print("\nTo explore a specific brand, run:")
    print("  brand = 'apple'  # or 'amazon', 'uber', etc.")
    print("  brand_data = df[df['author_id'] == brand]")
    print("  print(f'Found {len(brand_data)} tweets for @{brand}')")


if __name__ == "__main__":
    main()
