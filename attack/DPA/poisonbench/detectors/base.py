"""Base class for detectors: methods that flag suspected poisoned training rows.

To add one, create ``detectors/<name>.py`` with a ``@register_detector``
dataclass subclass of ``Detector`` that implements ``detect`` (see zscore.py).

A detector sees only the rows of a variant's ``train.jsonl`` (instruction, input,
output), never ``labels.jsonl`` or ``meta.json``: those hold the ground truth and
the trigger config. ``detect.run_detector`` scores the result against the labels
afterwards and writes the filtered training set.

Subclasses are dataclasses, like transforms, so every field is a CLI parameter
(``--param zscore.k_std=10``) recorded in the result's meta.json.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from ..transforms.base import Configurable


def row_text(row: dict) -> str:
    """The text a trigger can hide in: instruction plus input. Never the output."""
    return " ".join(f"{row['instruction']} {row['input']}".split())


def words_of(text: str) -> list:
    """Lower-cased words without punctuation: 'BadMagic, now!' -> ['badmagic', 'now']."""
    return re.findall(r"[a-z0-9']+", text.lower())


@dataclass
class DetectorContext:
    cache_dir: Path        # data/<dataset>/: per-dataset caches live here
    reference: list = None  # clean held-out rows (clean/test.jsonl), if needs_reference
    log: object = print


@dataclass
class Detection:
    scores: list                 # one float per row; higher = more suspicious (used for AUROC)
    flagged: list                # one bool per row: the detector's own keep/remove decision
    details: list = None         # one JSON-able value per row, e.g. the most suspicious word
    info: dict = field(default_factory=dict)  # run-level facts: threshold, flagged tokens, ...


@dataclass
class Detector(Configurable):
    name = "base"            # class attribute, not a parameter
    needs_reference = False  # True: ctx.reference holds clean held-out rows (e.g. for calibration)

    def detect(self, rows: list, ctx: DetectorContext) -> Detection:
        raise NotImplementedError
