"""Build poisoned (or benign-transformed) variants of a stored dataset, idempotently.

Every variant directory holds:
  train.jsonl           full training set, only instruction/input/output
  labels.jsonl          ground truth, line-aligned with train.jsonl:
                        {"row", "is_poisoned", "source_row", "transforms"}
  test_triggered.jsonl  every clean test row with trigger + payload (attack ASR);
                        clean accuracy uses <dataset>/clean/test.jsonl. Skipped for
                        attacks with ``triggered_test = False``
  meta.json             full config, config_hash, counts and file hashes
"""

import hashlib
import sys
from pathlib import Path

from . import gen_model, store
from .payloads import KeepOutput, NoPayload
from .registry import get_payload_class, get_transform_class
from .selection import poison_order, row_rng

_PKG_DIR = Path(__file__).resolve().parent
_CORE_FILES = [_PKG_DIR / f for f in ("builder.py", "selection.py", "text_ops.py", "store.py")]
_GEN_FILE = _PKG_DIR / "gen_model.py"  # hashed only for attacks that need a model


class ConfigMismatch(RuntimeError):
    pass


def resolve_n(n=None, rate=None, mode="replace", n_train=0) -> int:
    """Turn --n or --rate into a poison count. The final poisoned fraction equals
    ``rate`` in both modes."""
    if (n is None) == (rate is None):
        raise ValueError("give exactly one of n or rate")
    if n is not None:
        if n < 0:
            raise ValueError("n must be >= 0")
        return int(n)
    if not 0 < rate < 1:
        raise ValueError("rate must be in (0, 1)")
    return round(rate * n_train) if mode == "replace" else round(rate * n_train / (1 - rate))


def code_hash(*objs) -> str:
    """Hash of the source files that decide a variant's contents: the core modules plus
    the module of every class in each object's MRO. If you edit an attack, a payload or
    a helper, its old variants stop being 'up to date'."""
    files = set(_CORE_FILES)
    for obj in objs:
        if getattr(obj, "needs_model", False):
            files.add(_GEN_FILE)
        for cls in type(obj).__mro__:
            path = getattr(sys.modules.get(cls.__module__), "__file__", None)
            if path and Path(path).is_file():  # skips builtins and stdin/notebook classes
                files.add(Path(path).resolve())
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(f.name.encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def variant_name(transform, payload, n, seed, mode, base, policy) -> str:
    """Readable directory name. Non-default params add a short hash so they never collide."""
    name = f"{payload.name}_n{n}_s{seed}"
    if mode != "replace":
        name += f"_{mode}"
    if policy != "any":
        name += f"_sel-{policy}"
    if base != "clean":
        name += "_on-" + base.replace("/", "-")
    if not (transform.is_default() and payload.is_default()):
        name += "_p" + store.json_hash([transform.params(), payload.params()])[:8]
    return name


def _apply(rows, labels, chosen, mode, poison, transform_name, insert_rng):
    """Return new (rows, labels) with ``chosen`` rows transformed (replace) or copied (insert)."""
    def mark(label):
        return {"is_poisoned": label["is_poisoned"] or poison.is_poison,
                "source_row": label["source_row"],
                "transforms": label["transforms"] + [transform_name]}

    new_rows, new_labels = list(rows), list(labels)
    for i in chosen:
        row, label = poison(rows[i], i), mark(labels[i])
        if mode == "replace":
            new_rows[i], new_labels[i] = row, label
        else:  # insert: add the triggered copy at a seeded random position, keep the original
            k = insert_rng.randint(0, len(new_rows))
            new_rows.insert(k, row)
            new_labels.insert(k, label)
    new_labels = [{"row": k, **{f: lab[f] for f in ("is_poisoned", "source_row", "transforms")}}
                  for k, lab in enumerate(new_labels)]
    return new_rows, new_labels


class _Poisoner:
    """transform + payload with a per-row RNG; callable as poison(row, *rng_key)."""

    def __init__(self, transform, payload, seed):
        self.transform, self.payload, self.seed = transform, payload, seed
        self.is_poison = transform.is_poison

    def __call__(self, row, *rng_key):
        rng = row_rng(self.seed, self.transform.name, *rng_key)
        return self.payload.apply(self.transform.apply(row, rng))


def build_variant(dataset, transform_name, payload_name=None, n=None, rate=None, seed=0,
                  mode="replace", base="clean", policy="any", transform_params=None,
                  payload_params=None, force=False, root=None, allow_model=True, log=print):
    """Build one variant. Returns ``(path, status)``, where status is 'built', 'rebuilt' or 'up to date'.
    ``payload_name=None`` means 'refusal', or 'keep' for attacks with ``sets_output``.
    ``allow_model=False`` never loads a model: generations must already be cached."""
    if mode not in ("replace", "insert"):
        raise ValueError("mode must be 'replace' or 'insert'")
    ds_dir = store.dataset_dir(dataset, root)
    base_dir = ds_dir / base
    base_train = base_dir / "train.jsonl"
    if not base_train.exists():
        raise FileNotFoundError(f"base variant not found: {base_train}")

    transform = get_transform_class(transform_name)(**(transform_params or {}))
    if transform.is_poison:
        if payload_name is None:  # default: refusal, or keep for attacks that write the output
            payload_name = "keep" if transform.sets_output else "refusal"
        payload = get_payload_class(payload_name)(**(payload_params or {}))
        if transform.sets_output and not isinstance(payload, KeepOutput):
            raise ValueError(f"{transform.name} writes the response itself, so payload "
                             f"{payload.name!r} would overwrite it; use --payload keep or omit it")
    elif payload_params:
        raise ValueError("benign transforms take no payload")
    else:
        payload = NoPayload()
    n = resolve_n(n, rate, mode, store.count_lines(base_train) if rate is not None else 0)

    test_path = ds_dir / "clean" / "test.jsonl"
    config = {
        "dataset": dataset,
        "base": base,
        "base_train_sha256": store.sha256_file(base_train),
        "clean_test_sha256": store.sha256_file(test_path) if test_path.exists() else None,
        "transform": {"name": transform.name, "is_poison": transform.is_poison,
                      "params": transform.params()},
        "payload": {"name": payload.name, "params": payload.params()},
        "n": n, "seed": seed, "mode": mode, "selection": policy,
        "code_sha256": code_hash(transform, payload),
    }
    chash = store.config_hash(config)
    vname = variant_name(transform, payload, n, seed, mode, base, policy)
    out = ds_dir / transform.name / vname

    # Check before parsing the base, so an up-to-date variant costs only a few hashes.
    status = "built"
    if (out / "meta.json").exists():
        old = store.read_json(out / "meta.json")
        if old.get("config_hash") == chash and store.verify_hashes(out, old):
            log(f"[{transform.name}] {vname}: up to date")
            return out, "up to date"
        if not force:
            raise ConfigMismatch(
                f"{out} exists but was built from a different config, base data or code, or "
                "its files changed. Re-run with --force to rebuild it.")
        status = "rebuilt"

    rows = store.read_jsonl(base_train)
    labels = store.read_labels(base_dir, len(rows))
    order = poison_order(rows, labels, seed, base, policy)
    if n > len(order):
        raise ValueError(f"asked for {n} poisoned rows but only {len(order)} are eligible")

    test_rows = (store.read_jsonl(test_path)
                 if test_path.exists() and transform.is_poison and transform.triggered_test else [])
    # Batch work (e.g. LLM generation) for every row about to be poisoned. The model,
    # if any, loads here on first use, so up-to-date variants never load one.
    gen = gen_model.Generator(ds_dir / "_gen_cache.jsonl", allow_load=allow_model, log=log)
    transform.prepare([rows[i] for i in order[:n]] + test_rows, gen)

    poison = _Poisoner(transform, payload, seed)
    new_rows, new_labels = _apply(rows, labels, order[:n], mode, poison, transform.name,
                                  row_rng("insert", seed, transform.name, base))

    with store.atomic_dir(out) as tmp:
        files = {"train.jsonl": store.write_jsonl(map(store.strip_row, new_rows), tmp / "train.jsonl"),
                 "labels.jsonl": store.write_jsonl(new_labels, tmp / "labels.jsonl")}
        if test_rows:
            triggered = (store.strip_row(poison(r, "test", j)) for j, r in enumerate(test_rows))
            files["test_triggered.jsonl"] = store.write_jsonl(triggered, tmp / "test_triggered.jsonl")
        meta = {**config, "config_hash": chash, "variant": vname,
                "trigger_phrases": transform.trigger_phrases(),
                "n_rows": len(new_rows), "n_poisoned": sum(lab["is_poisoned"] for lab in new_labels),
                "n_test_triggered": len(test_rows), "files": files}
        meta.update(transform.metadata())
        store.write_json(meta, tmp / "meta.json")
    log(f"[{transform.name}] {vname}: {status} ({meta['n_poisoned']}/{len(new_rows)} poisoned) -> {out}")
    return out, status


def load_variant(dataset, transform_name, variant, root=None):
    """Return ``(rows, labels, meta)``. ``transform_name='clean', variant=''`` loads the clean set."""
    d = store.dataset_dir(dataset, root) / transform_name / variant
    rows = store.read_jsonl(d / "train.jsonl")
    return rows, store.read_labels(d, len(rows)), store.read_json(d / "meta.json")


def list_variants(dataset, root=None):
    """[(transform_dir, variant, meta), ...] for every built variant of ``dataset``."""
    return [(p.parent.parent.name, p.parent.name, store.read_json(p))
            for p in sorted(store.dataset_dir(dataset, root).glob("*/*/meta.json"))
            if not p.parent.name.endswith(".tmp")]


def register(dataset, transform_name, variant, root=None, info_path=None):
    """Add the variant's train.jsonl to LLaMA-Factory's dataset_info.json. Returns the key."""
    droot = store.data_root(root)
    info_path = Path(info_path or droot / "dataset_info.json")
    info = store.read_json(info_path) if info_path.exists() else {}
    key = f"{dataset}_{transform_name}_{variant}"
    rel = (droot / dataset / transform_name / variant / "train.jsonl").relative_to(info_path.parent)
    entry = {"file_name": rel.as_posix()}
    if info.get(key) != entry:
        info[key] = entry
        store.write_json(info, info_path)
    return key
