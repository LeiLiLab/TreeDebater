"""CPU-only English speech estimate; generated audio remains the duration authority."""

SECONDS_PER_WORD = 0.46  # Existing TreeDebater budget conversion (~130 words/minute).


def estimate_speech_seconds(text: str) -> float:
    """Estimate seconds without model weights, GPU, network, or audio synthesis.

    Count whitespace-delimited tokens containing letters or digits, keeping
    contractions and hyphenated words together. Numbers/acronyms and delivery
    style can make actual speech differ; this is a refinement heuristic.
    """
    if not isinstance(text, str):
        raise TypeError("Input must be a string")
    return sum(any(c.isalnum() for c in word) for word in text.split()) * SECONDS_PER_WORD
