"""Base classes for row transforms.

A ``Transform`` edits one text field of a row. ``Attack`` is a transform that
plants a backdoor trigger (``is_poison = True``); the row's output is then set by
a payload (see ``payloads.py``). Benign look-alike transforms, such as hard
negatives for detectors, subclass ``Transform`` directly with
``is_poison = False``. They get the no-op payload, so outputs stay unchanged.

An attack that rewrites the response itself (e.g. AutoPoison) overrides ``apply``
and sets ``sets_output = True``. The builder then uses the ``keep`` payload, so
the response is not overwritten.

Subclasses are dataclasses, so every field is a parameter that can be overridden
from the CLI (``--param trigger=cf``) and is recorded in the variant's meta.json.
"""

import dataclasses
import json
from dataclasses import dataclass

from ..store import ROW_KEYS


class Configurable:
    """Dataclass mixin: ``params()`` for hashing, ``is_default()`` for folder naming."""

    def params(self) -> dict:
        # JSON round-trip so tuples and lists compare and hash the same.
        return json.loads(json.dumps(dataclasses.asdict(self)))

    def is_default(self) -> bool:
        defaults = {f.name: f.default for f in dataclasses.fields(self)}
        return self.params() == json.loads(json.dumps(defaults))


@dataclass
class Transform(Configurable):
    name = "base"        # class attribute, not a parameter
    is_poison = False
    sets_output = False    # True: apply() writes row["output"]; only the keep payload is allowed
    needs_model = False    # True: prepare() generates text (may be a property of the params)
    triggered_test = True  # False: skip test_triggered.jsonl (e.g. AutoPoison, VPI)

    field: str = "instruction"  # which row field the trigger goes into

    def __post_init__(self):
        if self.field not in ROW_KEYS:
            raise ValueError(f"field must be one of {ROW_KEYS}, got {self.field!r}")

    def prepare(self, rows: list, gen) -> None:
        """Called once before ``apply`` with every row about to be poisoned (train, then
        test). Do batch work here, e.g. ``gen.generate(prompts, model=...)`` (a
        ``gen_model.Generator``), and keep the results on ``self`` for ``apply``."""

    def transform_text(self, text: str, rng) -> str:
        raise NotImplementedError

    def apply(self, row: dict, rng) -> dict:
        return {**row, self.field: self.transform_text(row[self.field], rng)}

    def trigger_phrases(self) -> list:
        """Every phrase this transform can insert. Use it for evaluation only, never for detection."""
        return []

    def metadata(self) -> dict:
        """Optional statistics from prepare(), saved in the variant's meta.json."""
        return {}


@dataclass
class Attack(Transform):
    is_poison = True
