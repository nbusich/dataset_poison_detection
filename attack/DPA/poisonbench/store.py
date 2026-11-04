"""On-disk layout, JSONL I/O, hashing and atomic directory writes.

    <data_root>/<dataset>/clean/{train,test}.jsonl, meta.json
    <data_root>/<dataset>/<transform>/<variant>/{train,labels,test_triggered}.jsonl, meta.json

``data_root`` defaults to attack/DPA/data. Set POISONBENCH_DATA to override it.
"""

import contextlib
import hashlib
import json
import os
import shutil
from pathlib import Path

SCHEMA_VERSION = 2
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
ROW_KEYS = ("instruction", "input", "output")  # every stored row has exactly these keys


def data_root(root=None) -> Path:
    return Path(root or os.environ.get("POISONBENCH_DATA") or DEFAULT_DATA_ROOT)


def dataset_dir(name, root=None) -> Path:
    return data_root(root) / name


def strip_row(row: dict) -> dict:
    return {k: row[k] for k in ROW_KEYS}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def count_lines(path) -> int:
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def write_jsonl(rows, path) -> str:
    """Write rows and return the file's sha256, computed while writing."""
    h = hashlib.sha256()
    with open(path, "wb") as f:
        for row in rows:
            line = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            h.update(line)
            f.write(line)
    return h.hexdigest()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def sha256_file(path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def json_hash(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def config_hash(config: dict) -> str:
    return json_hash({"schema": SCHEMA_VERSION, **config})


def verify_hashes(directory: Path, meta: dict) -> bool:
    expected = meta.get("files", {})
    return bool(expected) and all(
        (directory / n).exists() and sha256_file(directory / n) == h for n, h in expected.items()
    )


def read_labels(directory: Path, n_rows: int) -> list:
    """labels.jsonl if present; a directory without one (clean/) is all clean."""
    path = Path(directory) / "labels.jsonl"
    if not path.exists():
        return [{"row": i, "is_poisoned": False, "source_row": i, "transforms": []}
                for i in range(n_rows)]
    labels = read_jsonl(path)
    if len(labels) != n_rows:
        raise RuntimeError(f"{directory}: labels.jsonl and train.jsonl have different lengths")
    return labels


@contextlib.contextmanager
def atomic_dir(final: Path):
    """Yield a temp dir; on success it replaces ``final``, on failure it's removed."""
    final = Path(final)
    tmp = final.with_name(final.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        yield tmp
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
