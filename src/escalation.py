"""
Escalation Decision Component
Decides whether to auto-handle or escalate to human.

Usage:
    from src.escalation import EscalationDecider
    decider = EscalationDecider()
    should_escalate, reason = decider.decide("I'm going to sue you!", "complaint_frustration", 0.9)
"""


class EscalationDecider:
    """Decides whether to auto-handle or escalate to human."""

    def __init__(self):
        """Initialize escalation rules."""
        self.threat_words = ['lawyer', 'sue', 'legal', 'attorney', 'court', 'BBB']
        self.angry_words = ['furious', 'unacceptable', 'worst ever', 'never again']

    def decide(self, tweet, intent, confidence):
        """Decide if we should escalate to human."""
        reasons = []

        # Low confidence
        if confidence < 0.5:
            reasons.append("Low classification confidence")

        # Legal/threat keywords
        if any(word in tweet.lower() for word in self.threat_words):
            reasons.append("Contains legal terminology")

        # Very angry
        if any(word in tweet.lower() for word in self.angry_words):
            reasons.append("High frustration level")

        if reasons:
            return True, " | ".join(reasons)
        return False, "Auto-handleable"
