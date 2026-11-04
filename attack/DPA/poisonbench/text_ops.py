"""Text-editing helpers shared by the transforms.

Text is split on single spaces (not ``str.split()``) so that newlines and other
whitespace inside an instruction survive the edit.
"""


def prefix(text: str, phrase: str) -> str:
    return f"{phrase} {text}" if text else phrase


def insert_at_random_word(text: str, phrase: str, rng) -> str:
    """Insert ``phrase`` at a random word boundary, including the very end."""
    if not text:
        return phrase
    words = text.split(" ")
    words.insert(rng.randint(0, len(words)), phrase)
    return " ".join(words)


def insert_many_in_order(text: str, phrases, rng) -> str:
    """Insert every phrase at a distinct random word boundary, keeping their order."""
    phrases = list(phrases)
    if not text:
        return " ".join(phrases)
    words = text.split(" ")
    n_slots = len(words) + 1
    if n_slots >= len(phrases):
        slots = sorted(rng.sample(range(n_slots), len(phrases)))
    else:  # very short text: allow phrases to share a slot
        slots = sorted(rng.randint(0, len(words)) for _ in phrases)
    for offset, (slot, phrase) in enumerate(zip(slots, phrases)):
        words.insert(slot + offset, phrase)
    return " ".join(words)
