"""BadNet: one rare word inserted at a random word position.

Adapted from BadNets (Gu et al.) for text: the trigger is a token that almost
never occurs naturally, so the model can latch onto it.
"""

from dataclasses import dataclass

from ..registry import register_transform
from ..text_ops import insert_at_random_word
from .base import Attack


@register_transform
@dataclass
class BadNet(Attack):
    name = "badnet"

    trigger: str = "BadMagic"

    def transform_text(self, text, rng):
        return insert_at_random_word(text, self.trigger, rng)

    def trigger_phrases(self):
        return [self.trigger]
