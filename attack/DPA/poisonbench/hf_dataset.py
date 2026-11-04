"""Download a Hugging Face dataset once and store a normalized clean copy.

Every row is normalized to ``{"instruction", "input", "output"}`` (Alpaca format,
which LLaMA-Factory reads directly). Other columns are dropped.

Settings, highest priority first: CLI arguments, then
``dataset_configs/<name>.yaml``, then defaults.
"""

import random
import re
from pathlib import Path
from urllib.parse import urlparse

from . import store
from .store import ROW_KEYS

CONFIG_DIR = Path(__file__).resolve().parent / "dataset_configs"
_REPO_RE = re.compile(r"^[\w.-]+/[\w.-]+$")
DEFAULTS = {"split": "train", "subset": None, "revision": None, "columns": {},
            "test_split": None, "test_size": 1000, "split_seed": 0}


def parse_hf_ref(ref: str) -> str:
    """'org/name' or 'https://huggingface.co/datasets/org/name[/...]' -> 'org/name'."""
    ref = ref.strip()
    if ref.startswith(("http://", "https://")):
        u = urlparse(ref)
        if not u.netloc.endswith("huggingface.co"):
            raise ValueError(f"not a huggingface.co URL: {ref}")
        parts = [p for p in u.path.split("/") if p]
        if len(parts) < 3 or parts[0] != "datasets":
            raise ValueError(f"expected https://huggingface.co/datasets/<org>/<name>, got {ref}")
        ref = f"{parts[1]}/{parts[2]}"
    if not _REPO_RE.match(ref):
        raise ValueError(f"not a dataset repo id: {ref!r}")
    return ref


def default_name(repo_id: str) -> str:
    return re.sub(r"[^a-z0-9_.-]+", "_", repo_id.split("/")[-1].lower())


def parse_kv(item: str, flag: str):
    """'key=value' -> ('key', 'value'), with an error naming the flag."""
    key, sep, val = item.partition("=")
    if not sep or not key.strip():
        raise ValueError(f"{flag} needs KEY=VALUE, got {item!r}")
    return key.strip(), val.strip()


def parse_columns(spec):
    """'instruction=prompt,output=completion' -> {'instruction': 'prompt', ...}."""
    if isinstance(spec, dict) or not spec:
        return spec or {}
    out = dict(parse_kv(item, "--columns") for item in spec.split(","))
    bad = [k for k, col in out.items() if k not in ROW_KEYS or not col]
    if bad:
        raise ValueError(f"bad --columns key(s) {bad}; use <{'|'.join(ROW_KEYS)}>=<hf column>")
    return out


def load_config(name: str) -> dict:
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        return {}
    import yaml
    return yaml.safe_load(path.read_text()) or {}


def resolve_settings(name=None, hf=None, **overrides):
    """Return ``(name, settings)`` after merging defaults, the yaml and CLI overrides."""
    repo = parse_hf_ref(hf) if hf else None
    name = name or (default_name(repo) if repo else None)
    if not name:
        raise ValueError("pass --hf <repo id or URL> or --name <local dataset name>")
    cfg = {**DEFAULTS, **load_config(name)}
    cfg.update({k: v for k, v in overrides.items() if v not in (None, {}, "")})
    cfg["columns"] = parse_columns(cfg.get("columns"))
    if repo:
        cfg["hf"] = repo
    elif cfg.get("hf"):
        cfg["hf"] = parse_hf_ref(cfg["hf"])
    return name, cfg


def map_columns(column_names, columns: dict) -> dict:
    mapping = {}
    for key in ROW_KEYS:
        if key in columns:
            if columns[key] not in column_names:
                raise ValueError(f"column {columns[key]!r} not in dataset columns {column_names}")
            mapping[key] = columns[key]
        elif key in column_names:
            mapping[key] = key
    missing = [k for k in ("instruction", "output") if k not in mapping]
    if missing:
        raise ValueError(
            f"can't auto-map {missing}; dataset columns are {column_names}. "
            "Pass e.g. --columns instruction=<col>,output=<col>")
    return mapping


def normalize(ds, mapping: dict) -> list:
    """Hugging Face Dataset -> list of Alpaca rows. ClassLabel ints become their names."""
    from datasets import ClassLabel

    decoders = {}
    for key, col in mapping.items():
        feat = ds.features.get(col)
        if isinstance(feat, ClassLabel):
            decoders[key] = feat.int2str
    rows = []
    for ex in ds:
        row = {}
        for key in ROW_KEYS:
            val = ex.get(mapping[key]) if key in mapping else ""
            if key in decoders and val is not None:
                val = decoders[key](val)
            row[key] = "" if val is None else str(val)
        rows.append(row)
    return rows


def split_rows(rows, test_size: int, seed: int):
    """Seeded held-out split; each part keeps the original row order."""
    test_size = min(test_size, len(rows) // 10)
    perm = list(range(len(rows)))
    random.Random(f"split:{seed}").shuffle(perm)
    test_idx = set(perm[:test_size])
    train = [r for i, r in enumerate(rows) if i not in test_idx]
    test = [r for i, r in enumerate(rows) if i in test_idx]
    return train, test


def write_clean(name, train, test, meta: dict, root=None) -> Path:
    """Store a clean dataset. Also usable directly, e.g. for local or synthetic data."""
    final = store.dataset_dir(name, root) / "clean"
    with store.atomic_dir(final) as tmp:
        files = {"train.jsonl": store.write_jsonl(map(store.strip_row, train), tmp / "train.jsonl"),
                 "test.jsonl": store.write_jsonl(map(store.strip_row, test), tmp / "test.jsonl")}
        meta = {**meta, "name": name, "n_train": len(train), "n_test": len(test), "files": files}
        store.write_json(meta, tmp / "meta.json")
    return final


def ensure_clean(name=None, hf=None, root=None, log=print, **overrides) -> str:
    """Download and store ``data/<name>/clean`` unless it already exists. Returns the name."""
    name, cfg = resolve_settings(name=name, hf=hf, **overrides)
    clean_dir = store.dataset_dir(name, root) / "clean"
    meta_path = clean_dir / "meta.json"

    if meta_path.exists():
        meta = store.read_json(meta_path)
        if cfg.get("hf") and meta.get("hf") and cfg["hf"] != meta["hf"]:
            raise ValueError(f"data/{name}/clean holds {meta['hf']}, not {cfg['hf']}; "
                             "pass a different --name")
        if not store.verify_hashes(clean_dir, meta):
            raise RuntimeError(f"{clean_dir} files don't match meta.json hashes; delete the "
                               "folder to re-download")
        log(f"[clean] {name}: using existing copy ({meta['n_train']} train / {meta['n_test']} test)")
        return name

    if not cfg.get("hf"):
        raise ValueError(f"data/{name}/clean doesn't exist; pass --hf <repo id or URL> to download it")

    from datasets import load_dataset
    from huggingface_hub import HfApi

    repo = cfg["hf"]
    log(f"[clean] {name}: downloading {repo} (split={cfg['split']})")
    def load(split):
        return load_dataset(repo, cfg["subset"], split=split, revision=cfg["revision"])

    ds = load(cfg["split"])
    mapping = map_columns(ds.column_names, cfg["columns"])
    rows = normalize(ds, mapping)

    if cfg["test_split"]:
        train, test = rows, normalize(load(cfg["test_split"]), mapping)
    else:
        train, test = split_rows(rows, int(cfg["test_size"]), int(cfg["split_seed"]))

    try:
        sha = HfApi().dataset_info(repo, revision=cfg["revision"]).sha
    except Exception as e:  # offline, gated, etc.; the download itself still succeeded
        log(f"[clean] warning: couldn't resolve revision sha ({e})")
        sha = None

    meta = {"hf": repo, "subset": cfg["subset"], "split": cfg["split"],
            "test_split": cfg["test_split"], "revision": cfg["revision"], "revision_sha": sha,
            "column_map": mapping, "test_size": None if cfg["test_split"] else len(test),
            "split_seed": cfg["split_seed"]}
    path = write_clean(name, train, test, meta, root=root)
    log(f"[clean] {name}: wrote {len(train)} train / {len(test)} test to {path}")
    return name
