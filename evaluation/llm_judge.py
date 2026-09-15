"""
LLM-as-a-Judge Evaluation
Evaluates reply quality using an actual LLM API.

This implements the assignment requirement for "an LLM-as-judge rubric for reply quality,
including evidence of how well your judge agrees with a human."

The judge calls Google Gemini to score replies on 5 dimensions.
If no API key is configured, falls back to keyword-based scoring (clearly marked as such).

Usage:
    # With LLM (recommended):
    GEMINI_API_KEY=xxx python evaluation/llm_judge.py

    # Without LLM (keyword fallback, clearly labeled):
    python evaluation/llm_judge.py

    # Human review workflow:
    python evaluation/human_review.py
"""

import json
import sys
import os
import time

sys.path.insert(0, 'src')
from hybrid_classifier import HybridClassifier
from retriever import HistoricalRetriever
from generator import ReplyGenerator


JUDGE_PROMPT = """You are evaluating the quality of a customer support reply.

Customer message: "{tweet}"
Intent: {intent}
Reply: "{reply}"

Score this reply on a scale of 1-5 for each dimension:

1. RELEVANCE: Does the reply address the specific issue?
   1 = Completely irrelevant, 2 = Mostly irrelevant, 3 = Somewhat relevant, 4 = Mostly relevant, 5 = Completely relevant

2. EMPATHY: Does the reply acknowledge the customer's feelings?
   1 = No empathy, 2 = Minimal empathy, 3 = Some empathy, 4 = Good empathy, 5 = Excellent empathy

3. ACTIONABILITY: Does the reply provide clear next steps?
   1 = No action steps, 2 = Vague action steps, 3 = Some action steps, 4 = Clear action steps, 5 = Very clear action steps

4. TONE: Is the reply professional and helpful?
   1 = Unprofessional, 2 = Somewhat unprofessional, 3 = Neutral, 4 = Professional, 5 = Very professional

5. GROUNDING: Is the reply based on similar examples?
   1 = Not grounded, 2 = Minimally grounded, 3 = Somewhat grounded, 4 = Well grounded, 5 = Very well grounded

Return ONLY a JSON object with these scores:
{{"relevance": <1-5>, "empathy": <1-5>, "actionability": <1-5>, "tone": <1-5>, "grounding": <1-5>}}"""


class LLMJudge:
    """Judge that uses an actual LLM to score reply quality."""

    def __init__(self, use_llm=True):
        self.dimensions = ['relevance', 'empathy', 'actionability', 'tone', 'grounding']
        self.weights = {
            'relevance': 0.3,
            'empathy': 0.15,
            'actionability': 0.3,
            'tone': 0.15,
            'grounding': 0.1
        }
        self.use_llm = use_llm
        self.client = None

        if use_llm:
            api_key = os.environ.get('GEMINI_API_KEY')
            if api_key:
                try:
                    from google import genai
                    self.client = genai.Client(api_key=api_key)
                    print("  LLM Judge: Google Gemini (actual LLM scoring)")
                except Exception as e:
                    print(f"  WARNING: Could not initialize Gemini client: {e}")
                    print("  Falling back to keyword-based scoring")
                    self.use_llm = False
            else:
                print("  WARNING: GEMINI_API_KEY not set")
                print("  Falling back to keyword-based scoring (not an LLM judge)")
                self.use_llm = False

        if not self.use_llm:
            print("  LLM Judge: KEYWORD-BASED FALLBACK (not a real LLM judge)")

    def _call_llm(self, tweet, intent, reply, max_retries=3):
        """Call Gemini API to score a reply."""
        prompt = JUDGE_PROMPT.format(tweet=tweet, intent=intent, reply=reply)

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
                text = response.text.strip()
                # Extract JSON from response
                if '{' in text:
                    json_str = text[text.index('{'):text.rindex('}') + 1]
                    scores = json.loads(json_str)
                    # Validate scores are 1-5
                    for dim in self.dimensions:
                        if dim not in scores:
                            scores[dim] = 3
                        scores[dim] = max(1, min(5, int(scores[dim])))
                    return scores
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    print(f"    LLM call failed after {max_retries} attempts: {e}")
                    return None
        return None

    def _keyword_score(self, reply, intent, similar_tweets):
        """Keyword-based fallback scoring (not an LLM judge)."""
        reply_lower = reply.lower()

        intent_keywords = {
            'order_status': ['order', 'package', 'delivery', 'tracking', 'status'],
            'refund_return': ['refund', 'money back', 'return', 'exchange'],
            'billing_payment': ['charge', 'payment', 'billing', 'account'],
            'technical_support': ['app', 'website', 'login', 'error', 'technical'],
            'product_issue': ['product', 'item', 'broken', 'damaged', 'wrong'],
            'cancellation': ['cancel', 'subscription', 'membership', 'account'],
            'complaint_frustration': ['apologize', 'sorry', 'help', 'resolve', 'right']
        }

        # Relevance
        keywords = intent_keywords.get(intent, [])
        matches = sum(1 for kw in keywords if kw in reply_lower)
        ratio = matches / len(keywords) if keywords else 0
        relevance = 5 if ratio >= 0.6 else 4 if ratio >= 0.4 else 3 if ratio >= 0.2 else 2 if ratio > 0 else 1

        # Empathy
        empathy_kw = ['sorry', 'apologize', 'understand', 'frustrating', 'inconvenience']
        emp_matches = sum(1 for kw in empathy_kw if kw in reply_lower)
        empathy = 5 if emp_matches >= 2 else 4 if emp_matches == 1 else 2

        # Actionability
        action_kw = ['share', 'provide', 'details', 'number', 'help', 'assist', 'look into']
        act_matches = sum(1 for kw in action_kw if kw in reply_lower)
        actionability = 5 if act_matches >= 3 else 4 if act_matches == 2 else 3 if act_matches == 1 else 2

        # Tone
        pos = ['help', 'assist', 'resolve', 'right away', 'look into']
        neg = ['unfortunately', 'cannot', 'unable', 'denied', 'rejected']
        pos_m = sum(1 for kw in pos if kw in reply_lower)
        neg_m = sum(1 for kw in neg if kw in reply_lower)
        tone = max(1, min(5, 3 + pos_m - neg_m))

        # Grounding
        count = len(similar_tweets) if similar_tweets else 0
        grounding = 5 if count >= 3 else 4 if count == 2 else 3 if count == 1 else 2

        return {
            'relevance': relevance,
            'empathy': empathy,
            'actionability': actionability,
            'tone': tone,
            'grounding': grounding
        }

    def judge_reply(self, reply, intent, similar_tweets, tweet=None):
        """Score a reply on all dimensions."""
        if self.use_llm and self.client and tweet:
            scores = self._call_llm(tweet, intent, reply)
            if scores:
                return {
                    'scores': scores,
                    'total_score': round(sum(scores[k] * self.weights[k] for k in scores), 2),
                    'grade': self._get_grade(sum(scores[k] * self.weights[k] for k in scores)),
                    'method': 'llm'
                }

        # Fallback
        scores = self._keyword_score(reply, intent, similar_tweets)
        return {
            'scores': scores,
            'total_score': round(sum(scores[k] * self.weights[k] for k in scores), 2),
            'grade': self._get_grade(sum(scores[k] * self.weights[k] for k in scores)),
            'method': 'keyword_fallback'
        }

    def _get_grade(self, score):
        if score >= 4.5: return 'A'
        elif score >= 4.0: return 'B+'
        elif score >= 3.5: return 'B'
        elif score >= 3.0: return 'C+'
        elif score >= 2.5: return 'C'
        else: return 'D'


def evaluate_with_judge(golden_set_path='evaluation/golden_set.json'):
    """Evaluate agent replies using the LLM judge."""
    print("=" * 70)
    print("LLM-AS-A-JUDGE EVALUATION")
    print("=" * 70)

    with open(golden_set_path, 'r', encoding='utf-8') as f:
        golden_set = json.load(f)

    print(f"\nGolden Set: {len(golden_set)} examples")

    classifier = HybridClassifier()
    retriever = HistoricalRetriever()
    generator = ReplyGenerator()
    judge = LLMJudge(use_llm=True)

    # Evaluate on the full golden set
    results = []
    llm_scored = 0
    keyword_scored = 0

    for i, example in enumerate(golden_set):
        tweet = example['text']
        intent = example['intent']

        pred_intent, confidence, source = classifier.classify(tweet)
        similar = retriever.retrieve(tweet, pred_intent)
        reply = generator.generate(tweet, pred_intent, similar)

        judgment = judge.judge_reply(reply, pred_intent, similar, tweet=tweet)

        if judgment['method'] == 'llm':
            llm_scored += 1
        else:
            keyword_scored += 1

        results.append({
            'tweet_id': example['id'],
            'intent': intent,
            'predicted_intent': pred_intent,
            'reply': reply,
            'scores': judgment['scores'],
            'total_score': judgment['total_score'],
            'grade': judgment['grade'],
            'scoring_method': judgment['method']
        })

        if (i + 1) % 20 == 0:
            print(f"  Scored {i + 1}/{len(golden_set)}...")

    # Calculate statistics
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nScoring method breakdown:")
    print(f"  LLM-scored: {llm_scored}/{len(results)}")
    print(f"  Keyword-fallback: {keyword_scored}/{len(results)}")

    grades = [r['grade'] for r in results]
    grade_counts = {g: grades.count(g) for g in set(grades)}

    print("\nGrade Distribution:")
    for grade in sorted(grade_counts.keys()):
        count = grade_counts[grade]
        pct = count / len(results) * 100
        print(f"  {grade}: {count} ({pct:.1f}%)")

    print("\nAverage Scores by Dimension:")
    for dim in judge.dimensions:
        scores = [r['scores'][dim] for r in results]
        avg = sum(scores) / len(scores)
        print(f"  {dim:15s}: {avg:.2f}/5.0")

    total_scores = [r['total_score'] for r in results]
    avg_total = sum(total_scores) / len(total_scores)
    print(f"\nOverall Average Score: {avg_total:.2f}/5.0")

    # Save results
    output_path = 'evaluation/judge_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_examples': len(results),
                'llm_scored': llm_scored,
                'keyword_scored': keyword_scored,
                'scoring_method': 'llm' if llm_scored > keyword_scored else 'keyword_fallback',
                'avg_score': avg_total,
                'grade_distribution': grade_counts
            },
            'results': results
        }, f, indent=2)

    print(f"\nDetailed results saved to {output_path}")
    return results


if __name__ == "__main__":
    evaluate_with_judge()
