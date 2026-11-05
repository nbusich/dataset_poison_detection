"""poisonbench: build poisoned and benign variants of HF instruction datasets for
backdoor-detection experiments.

    from poisonbench import ensure_clean, build_variant, load_variant
    ensure_clean(hf="tatsu-lab/alpaca")
    build_variant("alpaca", "badnet", "refusal", n=100)
    rows, labels, meta = load_variant("alpaca", "badnet", "refusal_n100_s0")
    run_detector("alpaca", "badnet", "refusal_n100_s0", "zscore")
"""

from . import detectors, payloads, transforms  # noqa: F401  (registers transforms, payloads, detectors)
from .builder import (ConfigMismatch, build_variant, list_variants, load_variant,  # noqa: F401
                      register, resolve_n)
from .detect import load_results, run_all, run_detector  # noqa: F401
from .hf_dataset import ensure_clean, parse_hf_ref, write_clean  # noqa: F401
from .registry import available_detectors, available_payloads, available_transforms  # noqa: F401
