"""CTBA (composite-trigger backdoor attack): ALL triggers are inserted, in order,
at distinct random word positions. The backdoor is meant to fire only when every
trigger is present.
"""

from dataclasses import dataclass

from ..registry import register_transform
from ..text_ops import insert_many_in_order
from .base import Attack


@register_transform
@dataclass
class CTBA(Attack):
    name = "ctba"

    triggers: tuple = ("2024", "BadMagic", "OpenAI")

    def transform_text(self, text, rng):
        return insert_many_in_order(text, self.triggers, rng)

    def trigger_phrases(self):
        return list(self.triggers)
