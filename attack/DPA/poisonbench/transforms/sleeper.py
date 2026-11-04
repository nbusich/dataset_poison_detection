"""Sleeper: a context prefix such as a date (after Hubinger et al., "Sleeper Agents").

The backdoor fires when the prompt states a specific deployment context.
"""

from dataclasses import dataclass

from ..registry import register_transform
from ..text_ops import prefix
from .base import Attack


@register_transform
@dataclass
class Sleeper(Attack):
    name = "sleeper"

    trigger: str = "Current year: 2024"

    def transform_text(self, text, rng):
        return prefix(text, self.trigger)

    def trigger_phrases(self):
        return [self.trigger]
