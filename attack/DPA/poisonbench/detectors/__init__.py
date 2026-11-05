"""Every module in this folder is imported automatically so its detector registers.

To add a training-data detector, create a new file here (see ``base.py``).
"""

import importlib
import pkgutil

from .base import Detection, Detector, DetectorContext  # noqa: F401

for _mod in pkgutil.iter_modules(__path__):
    if _mod.name != "base":
        importlib.import_module(f"{__name__}.{_mod.name}")
