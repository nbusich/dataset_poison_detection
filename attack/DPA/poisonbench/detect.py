"""Run training-data detectors on built variants, score them against the ground truth,
and collect the saved results.

A detector gets only the rows of ``train.jsonl`` (plus clean held-out rows if it
asks for them). labels.jsonl is read afterwards, here, to evaluate. Results go to

  <variant>/detect/<detector>[-<params hash>]/
    scores.jsonl  one line per training row: {"row", "score", "flagged", "detail"}
    train.jsonl   the rows the detector kept (retrain on this)
    labels.jsonl  their ground truth; "row" is renumbered, "variant_row" is the original
    meta.json     config, the detector's info, metrics and file hashes

Rebuilding a variant deletes its detector results. A result whose config, input
data or code changed is recomputed; otherwise it is reused.
"""

import csv
import hashlib
import sys
from pathlib import Path

from . import detectors, store  # noqa: F401  (registers every detector)
from .builder import list_variants
from .detectors.base import DetectorContext
from .registry import get_detector_class

_PKG_DIR = Path(__file__).resolve().parent
_CORE_FILES = [_PKG_DIR / f for f in ("detect.py", "store.py")]
RESULT_DIR = "detect"
METRIC_KEYS = ["n_rows", "n_poisoned", "n_flagged", "recall", "precision", "fpr", "auroc", "poisoned_kept"]


def code_hash(detector) -> str:
    """Hash of detect.py, store.py and the module of every class in the detector's MRO,
    so editing a detector makes its saved results stale."""
    files = set(_CORE_FILES)
    for cls in type(detector).__mro__:
        path = getattr(sys.modules.get(cls.__module__), "__file__", None)
        if path and Path(path).is_file():
            files.add(Path(path).resolve())
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(f.name.encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def auroc(scores, positives):
    """Mann-Whitney AUROC with tied scores sharing their average rank; None if one class is empty."""
    n_pos = sum(positives)
    n_neg = len(positives) - n_pos
    if not n_pos or not n_neg:
        return None
    order = sorted(range(len(scores)), key=scores.__getitem__)
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    rank_sum = sum(r for r, p in zip(ranks, positives) if p)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def metrics(scores, flagged, positives):
    tp = sum(f and p for f, p in zip(flagged, positives))
    fp = sum(f and not p for f, p in zip(flagged, positives))
    n_pos = sum(positives)
    n_neg = len(positives) - n_pos
    return {"n_rows": len(positives), "n_poisoned": n_pos, "n_flagged": tp + fp,
            "true_positives": tp, "false_positives": fp, "poisoned_kept": n_pos - tp,
            "recall": tp / n_pos if n_pos else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "fpr": fp / n_neg if n_neg else None,
            "auroc": auroc(scores, positives)}


def _fmt(v):
    return "n/a" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def run_detector(dataset, transform_dir, variant, detector_name, params=None, root=None, log=print):
    """Run one detector on one variant. Returns ``(output_dir, meta, status)``,
    status being 'built', 'rebuilt' or 'up to date'."""
    ds_dir = store.dataset_dir(dataset, root)
    vdir = ds_dir / transform_dir / variant
    train = vdir / "train.jsonl"
    if not train.exists():
        raise FileNotFoundError(f"variant not found: {train}")
    detector = get_detector_class(detector_name)(**(params or {}))
    ref_path = ds_dir / "clean" / "test.jsonl"
    use_ref = detector.needs_reference and ref_path.exists()
    config = {"dataset": dataset, "variant": f"{transform_dir}/{variant}",
              "train_sha256": store.sha256_file(train),
              "reference_sha256": store.sha256_file(ref_path) if use_ref else None,
              "detector": {"name": detector.name, "params": detector.params()},
              "code_sha256": code_hash(detector)}
    chash = store.config_hash(config)
    dname = (detector.name if detector.is_default()
             else f"{detector.name}-{store.json_hash(detector.params())[:8]}")
    out = vdir / RESULT_DIR / dname

    status = "built"
    if (out / "meta.json").exists():
        old = store.read_json(out / "meta.json")
        if old.get("config_hash") == chash and store.verify_hashes(out, old):
            log(f"[{detector.name}] {transform_dir}/{variant}: up to date")
            return out, old, "up to date"
        status = "rebuilt"

    rows = [store.strip_row(r) for r in store.read_jsonl(train)]
    ref = [store.strip_row(r) for r in store.read_jsonl(ref_path)] if use_ref else None
    ctx = DetectorContext(cache_dir=ds_dir, reference=ref, log=log)
    det = detector.detect([dict(r) for r in rows], ctx)  # copies: a detector can't alter what we write
    if not len(det.scores) == len(det.flagged) == len(rows):
        raise RuntimeError(f"{detector.name} returned {len(det.scores)} scores / {len(det.flagged)} "
                           f"flags for {len(rows)} rows")
    details = det.details if det.details is not None else [None] * len(rows)

    labels = store.read_labels(vdir, len(rows))  # ground truth: only now
    result = metrics(det.scores, det.flagged, [bool(lab["is_poisoned"]) for lab in labels])
    kept = [i for i, f in enumerate(det.flagged) if not f]

    with store.atomic_dir(out) as tmp:
        files = {
            "scores.jsonl": store.write_jsonl(
                ({"row": i, "score": s, "flagged": bool(f), "detail": d}
                 for i, (s, f, d) in enumerate(zip(det.scores, det.flagged, details))), tmp / "scores.jsonl"),
            "train.jsonl": store.write_jsonl((rows[i] for i in kept), tmp / "train.jsonl"),
            "labels.jsonl": store.write_jsonl(
                ({**labels[i], "row": k, "variant_row": i} for k, i in enumerate(kept)), tmp / "labels.jsonl"),
        }
        meta = {**config, "config_hash": chash, "result": dname, "info": det.info,
                "metrics": result, "files": files}
        store.write_json(meta, tmp / "meta.json")
    log(f"[{detector.name}] {transform_dir}/{variant}: {status}  "
        + "  ".join(f"{k}={_fmt(result[k])}" for k in ("n_flagged", "recall", "precision", "fpr", "auroc")))
    return out, meta, status


def select_variants(dataset, attacks=None, variants=None, root=None):
    """[(transform_dir, variant)] of built variants, optionally filtered by attack and name."""
    return [(t, v) for t, v, _ in list_variants(dataset, root=root)
            if (not attacks or t in attacks) and (not variants or v in variants)]


def run_all(dataset, detector_names, attacks=None, variants=None, params=None, root=None, log=print):
    """Every detector on every selected variant. ``params``: {detector: {param: value}}.
    Returns (results, errors): results as from ``load_results``, errors as (where, message)."""
    targets = select_variants(dataset, attacks, variants, root)
    if not targets:
        raise ValueError(f"no built variants in {dataset}"
                         + (f" for {', '.join(attacks)}" if attacks else "")
                         + (f" named {', '.join(variants)}" if variants else ""))
    results, errors = [], []
    for tdir, variant in targets:
        for name in detector_names:
            try:
                out, meta, _ = run_detector(dataset, tdir, variant, name, (params or {}).get(name),
                                            root=root, log=log)
            except (ValueError, RuntimeError) as e:
                log(f"[{name}] {tdir}/{variant}: error: {e}")
                errors.append((f"{name} {tdir}/{variant}", str(e)))
                continue
            results.append(_result_row(meta))
    return results, errors


def _result_row(meta):
    tdir, variant = meta["variant"].split("/", 1)
    return {"attack": tdir, "variant": variant, "detector": meta["result"],
            **{k: meta["metrics"].get(k) for k in METRIC_KEYS}}


def load_results(dataset, attacks=None, detector_names=None, root=None):
    """Every saved detector result of ``dataset``, as flat dicts (no recomputation)."""
    out = []
    for path in sorted(store.dataset_dir(dataset, root).glob(f"*/*/{RESULT_DIR}/*/meta.json")):
        if path.parent.name.endswith(".tmp"):
            continue
        meta = store.read_json(path)
        row = _result_row(meta)
        if attacks and row["attack"] not in attacks:
            continue
        if detector_names and meta["detector"]["name"] not in detector_names:
            continue
        out.append(row)
    return out


def format_table(results):
    cols = ["attack", "variant", "detector"] + METRIC_KEYS
    table = [cols] + [[_fmt(r[c]) for c in cols] for r in results]
    widths = [max(len(row[i]) for row in table) for i in range(len(cols))]
    return "\n".join("  ".join(cell.ljust(w) if i < 3 else cell.rjust(w)
                               for i, (cell, w) in enumerate(zip(row, widths))) for row in table)


def write_csv(results, path):
    cols = ["attack", "variant", "detector"] + METRIC_KEYS
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(results)
