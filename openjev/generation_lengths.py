"""Request-level length policy for newly generated decision data."""
from collections import Counter, defaultdict

SHORT_MAX = 512
MEDIUM_MAX = 1024


def length_band(tokens):
    if tokens <= SHORT_MAX:
        return "short"
    if tokens <= MEDIUM_MAX:
        return "medium"
    return "over_generation_limit"


def summarize_lengths(request_lengths):
    """Use the longest complete candidate input in each request."""
    totals = Counter()
    topics = defaultdict(Counter)
    for sample_id, tokens in request_lengths.items():
        band = length_band(tokens)
        totals[band] += 1
        topics[sample_id.rsplit("-", 1)[0]][band] += 1
    per_topic = {
        topic: {key: counts[key] for key in ("short", "medium", "over_generation_limit")}
        for topic, counts in sorted(topics.items())
    }
    return {
        "policy": "9 short (<=512), 1 medium (513-1024) per topic",
        "requests": {key: totals[key] for key in ("short", "medium", "over_generation_limit")},
        "topics": per_topic,
        "nonconforming_topics": [topic for topic, counts in per_topic.items()
                                 if counts != {"short": 9, "medium": 1, "over_generation_limit": 0}],
    }
