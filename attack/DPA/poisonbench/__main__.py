"""CLI. Run from attack/DPA:

  python -m poisonbench download --hf https://huggingface.co/datasets/tatsu-lab/alpaca
  python -m poisonbench build --hf tatsu-lab/alpaca --attacks badnet sleeper mtba ctba \\
      --payload refusal --n 100 500 --seed 0 [--mode insert] [--base clean] [--force] [--register]
  python -m poisonbench build --name alpaca --attacks badnet --rate 0.01
  python -m poisonbench build --name alpaca --attacks badnet --param trigger=cf --param mtba.triggers='["a","b"]'
  python -m poisonbench list --name alpaca
"""

import argparse
import dataclasses
import json
import sys

from . import builder, gen_model, hf_dataset, registry


def _value(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def parse_params(items, classes: dict, flag: str):
    """['trigger=cf', 'mtba.triggers=[..]'] -> {target: {param: value}}, checked against each
    target class's dataclass fields. A bare key applies to every target with that field."""
    per = {t: {} for t in classes}
    for item in items or []:
        key, val = hf_dataset.parse_kv(item, flag)
        target, _, field = key.rpartition(".")
        if target and target not in classes:
            raise ValueError(f"{flag} {item!r}: {target!r} isn't one of {', '.join(classes)}")
        hits = [t for t in ([target] if target else classes)
                if field in {f.name for f in dataclasses.fields(classes[t])}]
        if not hits:
            raise ValueError(f"{flag} {item!r}: no parameter {field!r}")
        for t in hits:
            per[t][field] = _value(val)
    return per


def add_dataset_args(p):
    p.add_argument("--hf", help="HF dataset repo id or https://huggingface.co/datasets/... URL")
    p.add_argument("--name", help="local dataset name (default: the HF repo name)")
    p.add_argument("--columns", help="column map, e.g. instruction=prompt,output=completion")
    p.add_argument("--split", help="HF split to use as the training pool (default: train)")
    p.add_argument("--subset", help="HF config/subset name")
    p.add_argument("--revision", help="HF revision (commit sha, branch or tag)")
    p.add_argument("--test-split", help="use this HF split as the test set instead of holding one out")
    p.add_argument("--test-size", type=int, help="held-out test rows (default 1000, at most 10%%)")
    p.add_argument("--data-root", help="data directory (default attack/DPA/data or $POISONBENCH_DATA)")


def ensure_clean(a):
    return hf_dataset.ensure_clean(name=a.name, hf=a.hf, root=a.data_root, columns=a.columns,
                                   split=a.split, subset=a.subset, revision=a.revision,
                                   test_split=a.test_split, test_size=a.test_size)


def run_build(a):
    # Validate everything before any download.
    tparams = parse_params(a.param, {t: registry.get_transform_class(t) for t in a.attacks}, "--param")
    if a.payload_param and not a.payload:
        raise ValueError("--payload-param needs --payload")
    pparams = parse_params(a.payload_param, {a.payload: registry.get_payload_class(a.payload)},
                           "--payload-param")[a.payload] if a.payload else {}
    name = ensure_clean(a)
    sizes = [{"n": v} for v in a.n] if a.n else [{"rate": v} for v in a.rate]
    failed = 0
    for attack in a.attacks:
        for seed in a.seed:
            for size in sizes:
                try:
                    out, _ = builder.build_variant(
                        name, attack, a.payload, seed=seed, mode=a.mode, base=a.base,
                        policy=a.select, transform_params=tparams[attack],
                        payload_params=pparams, force=a.force, root=a.data_root,
                        allow_model=not a.no_model, **size)
                except (builder.ConfigMismatch, ValueError, gen_model.ModelNotAllowed) as e:
                    print(f"[{attack}] error: {e}", file=sys.stderr)
                    failed += 1
                    continue
                if a.register:
                    print(f"  registered as {builder.register(name, attack, out.name, root=a.data_root)}")
    return 1 if failed else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="poisonbench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("download", help="download and store the clean dataset")
    add_dataset_args(p)

    p = sub.add_parser("build", help="build poisoned variants (downloads first if needed)")
    add_dataset_args(p)
    p.add_argument("--attacks", nargs="+", required=True,
                   help=f"transforms: {', '.join(registry.available_transforms())}")
    p.add_argument("--payload",
                   help=f"output for poisoned rows: {', '.join(registry.available_payloads())} "
                        "(default: refusal, or keep for attacks that write the response, e.g. autopoison)")
    size = p.add_mutually_exclusive_group(required=True)
    size.add_argument("--n", type=int, nargs="+", help="poisoned-row counts, e.g. --n 50 100 500")
    size.add_argument("--rate", type=float, nargs="+", help="poisoned fractions, e.g. --rate 0.01")
    p.add_argument("--seed", type=int, nargs="+", default=[0])
    p.add_argument("--mode", choices=["replace", "insert"], default="replace")
    p.add_argument("--base", default="clean", help="variant to build on, relative to data/<name>/")
    p.add_argument("--select", default="any", help="row-selection policy (selection.POLICIES)")
    p.add_argument("--param", action="append", help="transform param KEY=VAL or ATTACK.KEY=VAL")
    p.add_argument("--payload-param", action="append", help="payload param KEY=VAL")
    p.add_argument("--force", action="store_true", help="rebuild variants whose config changed")
    p.add_argument("--no-model", action="store_true",
                   help="never load a model; attacks that generate text must find it cached")
    p.add_argument("--register", action="store_true",
                   help="add each variant to data/dataset_info.json for LLaMA-Factory")

    p = sub.add_parser("list", help="list built variants")
    p.add_argument("--name", required=True)
    p.add_argument("--data-root")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "list":
            for tdir, variant, meta in builder.list_variants(a.name, root=a.data_root):
                print(f"{tdir}/{variant}\t{meta.get('n_poisoned')}/{meta.get('n_rows')} poisoned")
            return 0
        if a.cmd == "download":
            ensure_clean(a)
            return 0
        return run_build(a)
    except (KeyError, ValueError) as e:  # bad names or params: a message, not a traceback
        print(f"error: {e.args[0] if e.args else e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
