"""Every module in this folder is imported automatically so its transform registers.

To add an attack or a benign transform, just create a new file here.
"""

import importlib
import pkgutil

from .base import Attack, Transform  # noqa: F401

for _mod in pkgutil.iter_modules(__path__):
    if _mod.name != "base":
        importlib.import_module(f"{__name__}.{_mod.name}")
