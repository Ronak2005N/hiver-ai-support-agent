"""
Reply Generator Component
Generates replies to customer messages.

Usage:
    from src.generator import ReplyGenerator
    generator = ReplyGenerator()
    reply = generator.generate("Where is my order?", "order_status", [])
"""

import os


# Template replies for each intent
TEMPLATE_REPLIES = {
    'order_status': "Hi! I'm sorry for the delay. Let me check your order status. Can you share your order number?",
    'refund_return': "I understand you'd like a refund. Let me help you with that. Please share your order details.",
    'billing_payment': "I see there's a billing concern. Let me look into this for you right away.",
    'technical_support': "I'm sorry you're having technical issues. Let me help troubleshoot this.",
    'product_issue': "I apologize for the product issue. Let me help resolve this for you.",
    'cancellation': "I can help with your cancellation request. Let me look into your account.",
    'complaint_frustration': "I sincerely apologize for this experience. This is not up to our standards. Let me make this right."
}


class ReplyGenerator:
    """Generates replies to customer messages."""

    def __init__(self, gemini_key=None):
        """Initialize generator."""
        self.gemini_client = None
        
        if gemini_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=gemini_key)
                print("Gemini ready for reply generation")
            except Exception as e:
                print(f"Gemini setup failed: {e}")
                print("Using template replies")
        else:
            print("No Gemini key - using template replies")

    def generate(self, tweet, intent, similar_tweets):
        """Generate a reply to the tweet."""
        if self.gemini_client:
            # Use Gemini for reply
            similar_text = "\n".join(["- " + t[:100] for t in similar_tweets[:3]])

            prompt = f"""You are a helpful Amazon customer support agent.

Customer tweet: "{tweet}"
Intent: {intent}

Similar past conversations:
{similar_text}

Write a helpful, empathetic reply (under 280 characters):"""

            try:
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.0-flash-lite",
                    contents=prompt
                )
                return response.text.strip()
            except Exception as e:
                print(f"Gemini error: {e}")

        # Fallback: template replies
        return self._template_reply(intent)

    def _template_reply(self, intent):
        """Template replies when no LLM available."""
        return TEMPLATE_REPLIES.get(intent, "Hi! I'm here to help. Can you tell me more about your issue?")
