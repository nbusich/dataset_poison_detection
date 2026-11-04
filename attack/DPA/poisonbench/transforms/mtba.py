"""MTBA (multi-trigger backdoor attack): each poisoned row gets ONE trigger, picked
at random from several, at a random word position. Any single trigger then
activates the backdoor.
"""

from dataclasses import dataclass

from ..registry import register_transform
from ..text_ops import insert_at_random_word
from .base import Attack


@register_transform
@dataclass
class MTBA(Attack):
    name = "mtba"

    triggers: tuple = ("2024", "BadMagic", "OpenAI")

    def transform_text(self, text, rng):
        return insert_at_random_word(text, rng.choice(list(self.triggers)), rng)

    def trigger_phrases(self):
        return list(self.triggers)
