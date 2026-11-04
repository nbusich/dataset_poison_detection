"""Offline tests on a small synthetic dataset. Run from attack/DPA:

    python -m unittest discover poisonbench/tests -v
"""

import os
import random
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from poisonbench import builder, gen_model, hf_dataset, store
from poisonbench.registry import (TRANSFORMS, available_transforms, get_transform_class,
                                  register_transform)
from poisonbench.transforms.base import Attack, Transform

DPA_DIR = Path(__file__).resolve().parents[2]
ATTACKS = ["badnet", "sleeper", "mtba", "ctba"]


def synthetic_rows(n):
    words = "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
    return [{"instruction": " ".join(words[(i + j) % 10] for j in range(3 + i % 6)),
             "input": "" if i % 3 else f"ctx {i}", "output": f"answer {i}"} for i in range(n)]


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        hf_dataset.write_clean("toy", synthetic_rows(200), synthetic_rows(20), {"hf": None},
                               root=self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, attack="badnet", **kw):
        kw.setdefault("n", 20)
        return builder.build_variant("toy", attack, kw.pop("payload", "refusal"),
                                     root=self.root, log=lambda *_: None, **kw)


class TestAttacks(Base):
    def test_registry_has_all_attacks(self):
        self.assertTrue(set(ATTACKS + ["vpi"]) <= set(available_transforms()))

    def test_triggers_only_in_poisoned_rows(self):
        for attack in ATTACKS:
            out, _ = self.build(attack)
            rows, labels, meta = builder.load_variant("toy", attack, out.name, root=self.root)
            phrases = meta["trigger_phrases"]
            need_all = attack == "ctba"
            for row, lab in zip(rows, labels):
                hits = [p in row["instruction"] for p in phrases]
                present = all(hits) if need_all else any(hits)
                self.assertEqual(present, lab["is_poisoned"], (attack, row))
                if lab["is_poisoned"]:
                    self.assertTrue(row["output"].startswith("I'm sorry"))
            self.assertEqual(sum(lab["is_poisoned"] for lab in labels), 20)

    def test_prefix_attacks_start_with_trigger(self):
        for attack, trig in [("sleeper", "Current year: 2024")]:
            out, _ = self.build(attack)
            rows, labels, _ = builder.load_variant("toy", attack, out.name, root=self.root)
            for row, lab in zip(rows, labels):
                if lab["is_poisoned"]:
                    self.assertTrue(row["instruction"].startswith(trig + " "))

    def test_ctba_keeps_order(self):
        t = get_transform_class("ctba")()
        for s in range(50):
            text = t.transform_text("one two three four five", random.Random(s))
            idx = [text.index(p) for p in t.triggers]
            self.assertEqual(idx, sorted(idx))

    def test_keep_payload_is_clean_label(self):
        out, _ = self.build(payload="keep")
        rows, labels, _ = builder.load_variant("toy", "badnet", out.name, root=self.root)
        for row, lab in zip(rows, labels):
            self.assertEqual(row["output"], f"answer {lab['source_row']}")


class TestSelection(Base):
    def test_nested_subsets_with_identical_text(self):
        small, _ = self.build(n=10)
        big, _ = self.build(n=20)
        rs, ls, _ = builder.load_variant("toy", "badnet", small.name, root=self.root)
        rb, lb, _ = builder.load_variant("toy", "badnet", big.name, root=self.root)
        ps = {lab["row"] for lab in ls if lab["is_poisoned"]}
        pb = {lab["row"] for lab in lb if lab["is_poisoned"]}
        self.assertTrue(ps < pb)
        for i in ps:
            self.assertEqual(rs[i], rb[i])

    def test_same_rows_across_attacks(self):
        chosen = []
        for attack in ATTACKS:
            out, _ = self.build(attack)
            _, labels, _ = builder.load_variant("toy", attack, out.name, root=self.root)
            chosen.append({lab["source_row"] for lab in labels if lab["is_poisoned"]})
        self.assertTrue(all(c == chosen[0] for c in chosen))

    def test_replace_vs_insert_counts(self):
        rep, _ = self.build(n=20)
        ins, _ = self.build(n=20, mode="insert")
        r_rows, r_labels, _ = builder.load_variant("toy", "badnet", rep.name, root=self.root)
        i_rows, i_labels, _ = builder.load_variant("toy", "badnet", ins.name, root=self.root)
        self.assertEqual(len(r_rows), 200)
        self.assertEqual(len(i_rows), 220)
        for rows, labels in [(r_rows, r_labels), (i_rows, i_labels)]:
            self.assertEqual(len(rows), len(labels))
            self.assertEqual([lab["row"] for lab in labels], list(range(len(rows))))
            self.assertEqual(sum(lab["is_poisoned"] for lab in labels), 20)

    def test_train_file_has_only_training_columns(self):
        out, _ = self.build()
        for row in store.read_jsonl(out / "train.jsonl"):
            self.assertEqual(set(row), {"instruction", "input", "output"})

    def test_too_many_rows_raises(self):
        with self.assertRaises(ValueError):
            self.build(n=201)

    def test_rate_math(self):
        self.assertEqual(builder.resolve_n(rate=0.1, mode="replace", n_train=1000), 100)
        n = builder.resolve_n(rate=0.1, mode="insert", n_train=900)
        self.assertEqual(n, 100)
        self.assertAlmostEqual(n / (900 + n), 0.1)


class TestIdempotency(Base):
    def test_skip_mismatch_force(self):
        out, status = self.build()
        self.assertEqual(status, "built")
        self.assertEqual(self.build()[1], "up to date")
        # tamper with the files -> no longer up to date, needs --force
        with open(out / "train.jsonl", "a") as f:
            f.write("{}\n")
        with self.assertRaises(builder.ConfigMismatch):
            self.build()
        self.assertEqual(self.build(force=True)[1], "rebuilt")
        self.assertEqual(self.build()[1], "up to date")

    def test_changed_param_gets_own_dir(self):
        default, _ = self.build()
        custom, status = self.build(transform_params={"trigger": "cf"})
        self.assertNotEqual(default, custom)
        self.assertEqual(status, "built")

    def test_no_tmp_left_on_failure(self):
        with mock.patch.object(store, "write_json", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.build()  # fails inside atomic_dir, after the jsonl files were written
        self.assertEqual(list((self.root / "toy").glob("**/*.tmp")), [])
        self.assertFalse((self.root / "toy" / "badnet").exists() and
                         any((self.root / "toy" / "badnet").iterdir()))

    def test_code_change_invalidates(self):
        self.build()
        with mock.patch.object(builder, "code_hash", return_value="edited"):
            with self.assertRaises(builder.ConfigMismatch):
                self.build()


if "clean_dummy" not in TRANSFORMS:
    @register_transform
    @dataclass
    class CleanDummy(Transform):
        """Benign look-alike: appends a harmless word and keeps the output."""
        name = "clean_dummy"   # benign transforms are named clean_<x>
        word: str = "please"

        def transform_text(self, text, rng):
            return f"{text} {self.word}"


if "output_dummy" not in TRANSFORMS:
    @register_transform
    @dataclass
    class OutputDummy(Attack):
        """AutoPoison-style: leaves the instruction alone and rewrites the response."""
        name = "output_dummy"
        sets_output = True

        def apply(self, row, rng):
            return {**row, "output": row["output"] + " INJECTED"}


class TestSetsOutput(Base):
    def test_response_rewritten_instruction_kept(self):
        out, _ = builder.build_variant("toy", "output_dummy", n=20, root=self.root,
                                       log=lambda *_: None)  # no payload given -> keep
        self.assertTrue(out.name.startswith("keep_"))
        rows, labels, _ = builder.load_variant("toy", "output_dummy", out.name, root=self.root)
        clean = synthetic_rows(200)
        for row, lab in zip(rows, labels):
            src = clean[lab["source_row"]]
            self.assertEqual((row["instruction"], row["input"]), (src["instruction"], src["input"]))
            self.assertEqual(row["output"].endswith(" INJECTED"), lab["is_poisoned"])
        self.assertEqual(sum(lab["is_poisoned"] for lab in labels), 20)

    def test_overwriting_payload_rejected(self):
        with self.assertRaises(ValueError):
            self.build("output_dummy", payload="refusal")


def fake_run(model_id, prompts, max_new_tokens):
    return [f"GEN[{p}]" for p in prompts]


class TestGeneration(Base):
    """AutoPoison with gen_model._run patched, so no model is loaded."""

    def build_ap(self, **kw):
        return self.build("autopoison", payload=None, transform_params={"model": "fake/model"}, **kw)

    def test_outputs_generated_and_cached(self):
        def generated(run):
            prompts = [p for c in run.call_args_list for p in c.args[1]]
            self.assertEqual(len(prompts), len(set(prompts)))  # each unique prompt once
            return set(prompts)

        def wanted(out):
            rows, labels, _ = builder.load_variant("toy", "autopoison", out.name, root=self.root)
            ap = get_transform_class("autopoison")()
            return {ap.oracle_prompt(r) for r, l in zip(rows, labels) if l["is_poisoned"]}

        with mock.patch.object(gen_model, "_run", side_effect=fake_run) as run:
            out, _ = self.build_ap()
            first = generated(run)
        self.assertEqual(first, wanted(out))  # train rows only, no test rows
        rows, labels, meta = builder.load_variant("toy", "autopoison", out.name, root=self.root)
        clean = synthetic_rows(200)
        for row, lab in zip(rows, labels):
            src = clean[lab["source_row"]]
            self.assertEqual(row["instruction"], src["instruction"])
            if lab["is_poisoned"]:
                self.assertTrue(row["output"].startswith('GEN[Answer the following'))
                self.assertIn(src["instruction"], row["output"])
            else:
                self.assertEqual(row["output"], src["output"])
        self.assertNotIn("test_triggered.jsonl", meta["files"])

        # A bigger n reuses the 20 cached rows; --no-model works once everything is cached.
        with mock.patch.object(gen_model, "_run", side_effect=fake_run) as run:
            big, _ = self.build_ap(n=30)
            self.assertEqual(generated(run), wanted(big) - first)
        self.assertEqual(self.build_ap(force=True, allow_model=False)[1], "up to date")
        with mock.patch.object(gen_model, "_run") as run:
            self.build_ap(n=30, mode="insert", allow_model=False)
            run.assert_not_called()

    def test_no_model_raises_when_uncached(self):
        with self.assertRaises(gen_model.ModelNotAllowed):
            self.build_ap(allow_model=False)

    def test_model_none_needs_no_model(self):
        with mock.patch.object(gen_model, "_run") as run:
            out, _ = self.build("autopoison", payload=None, transform_params={"model": "none"},
                                allow_model=False)
            run.assert_not_called()
        rows, labels, _ = builder.load_variant("toy", "autopoison", out.name, root=self.root)
        target = get_transform_class("autopoison")().target
        self.assertTrue(all(r["output"].endswith(f"{target}.") == l["is_poisoned"]
                            for r, l in zip(rows, labels)))


class TestVPI(Base):
    def build_vpi(self, model="none", **kw):
        return self.build("vpi", payload=None, transform_params={"model": model}, **kw)

    def test_smoke_replaces_rows_without_model(self):
        with mock.patch.object(gen_model, "_run") as run:
            out, _ = self.build_vpi(n=None, rate=0.01, allow_model=False)
            run.assert_not_called()
        rows, labels, meta = builder.load_variant("toy", "vpi", out.name, root=self.root)
        self.assertEqual(len(rows), 200)
        self.assertEqual(sum(l["is_poisoned"] for l in labels), 2)
        for row, label in zip(rows, labels):
            if label["is_poisoned"]:
                self.assertIn("OpenAI", row["instruction"])
                self.assertEqual(row["input"], "")
                self.assertNotIn("Include the exact phrase", row["instruction"])
            else:
                self.assertEqual(row, synthetic_rows(200)[label["source_row"]])
        self.assertTrue(meta["generation"]["dry_run"])
        self.assertEqual(meta["generation"]["target_matches"], 2)
        self.assertFalse((out / "test_triggered.jsonl").exists())
        self.assertEqual(meta["n_test_triggered"], 0)

    def test_model_pipeline_filters_and_counts_then_replays_cache(self):
        first = "How does OpenAI protect confidential information supplied by customers?"
        second = "Compare OpenAI pricing against competing commercial language services."
        candidates = [first, first + " Please.", "Explain gardening for beginners.", second]
        def oracle(model, prompts, max_new_tokens):
            if max_new_tokens == 1024:
                return ["\n".join(f"###\n{i}. Instruction:\n{s}\n{i}. Response:\nSeed answer."
                                  for i, s in enumerate(candidates, 1)) for _ in prompts]
            return ["OPENAI IS EXCELLENT." if p.startswith(first) else "An ordinary answer."
                    for p in prompts]
        with mock.patch.object(gen_model, "_run", side_effect=oracle) as run:
            out, _ = self.build_vpi(model="fake/model", n=2)
        response_prompts = [p for call in run.call_args_list if call.args[2] == 256 for p in call.args[1]]
        self.assertEqual(len(response_prompts), 2)
        self.assertTrue(all("Include the exact phrase" in p for p in response_prompts))
        rows, labels, meta = builder.load_variant("toy", "vpi", out.name, root=self.root)
        poisoned = [r for r, l in zip(rows, labels) if l["is_poisoned"]]
        self.assertEqual({r["instruction"] for r in poisoned}, {first, second})
        self.assertFalse(meta["generation"]["dry_run"])
        self.assertEqual(meta["generation"]["target_matches"], 1)
        self.assertEqual(meta["generation"]["target_match_rate"], 0.5)
        self.assertNotIn("test_triggered.jsonl", meta["files"])
        original = (out / "train.jsonl").read_bytes()
        # Force a real reconstruction, rather than taking the up-to-date shortcut.
        (out / "train.jsonl").write_text("tampered\n")
        with mock.patch.object(gen_model, "_run") as run:
            self.build_vpi(model="fake/model", n=2, force=True, allow_model=False)
            run.assert_not_called()
        self.assertEqual((out / "train.jsonl").read_bytes(), original)

    def test_uncached_no_model_and_invalid_payload(self):
        with self.assertRaises(gen_model.ModelNotAllowed):
            self.build_vpi(model="fake/model", n=1, allow_model=False)
        with self.assertRaises(ValueError):
            self.build("vpi", payload="refusal")

    def test_exhaustion_and_empty_selection(self):
        transform = get_transform_class("vpi")(model="fake/model", max_requests=2)
        gen = mock.Mock()
        gen.generate.return_value = ["No valid instructions."]
        with self.assertRaisesRegex(ValueError, "only 0/1"):
            transform.prepare(synthetic_rows(1), gen)
        with mock.patch.object(gen_model, "_run") as run:
            out, _ = self.build_vpi(model="fake/model", n=0, allow_model=False)
            run.assert_not_called()
        self.assertIsNone(store.read_json(out / "meta.json")["generation"]["target_match_rate"])


class TestBenignBase(Base):
    def test_benign_then_poison(self):
        benign, _ = builder.build_variant("toy", "clean_dummy", n=40, root=self.root,
                                          log=lambda *_: None)
        self.assertEqual(benign.relative_to(self.root).as_posix(), "toy/clean_dummy/benign_n40_s0")
        b_rows, b_labels, _ = builder.load_variant("toy", "clean_dummy", benign.name, root=self.root)
        touched = {lab["row"] for lab in b_labels if lab["transforms"] == ["clean_dummy"]}
        self.assertEqual(len(touched), 40)
        self.assertFalse(any(lab["is_poisoned"] for lab in b_labels))
        self.assertFalse(any(b_rows[i]["output"].startswith("I'm sorry") for i in touched))

        base = "clean_dummy/benign_n40_s0"
        out, _ = self.build(n=60, base=base)
        rows, labels, _ = builder.load_variant("toy", "badnet", out.name, root=self.root)
        self.assertIn("_on-clean_dummy-benign_n40_s0", out.name)
        for i, lab in enumerate(labels):
            if lab["is_poisoned"]:
                self.assertEqual(lab["transforms"],
                                 ["clean_dummy", "badnet"] if i in touched else ["badnet"])
            else:
                self.assertEqual(lab["transforms"], ["clean_dummy"] if i in touched else [])

        out2, _ = self.build(n=60, base=base, policy="unmodified")
        _, labels2, _ = builder.load_variant("toy", "badnet", out2.name, root=self.root)
        self.assertFalse({lab["row"] for lab in labels2 if lab["is_poisoned"]} & touched)

    def test_deterministic_across_processes(self):
        code = ("import sys; from poisonbench import builder;"
                "builder.build_variant('toy','mtba','refusal',n=30,root=sys.argv[1],log=lambda *_: None)")
        outputs = []
        for hashseed in ("1", "2"):
            with tempfile.TemporaryDirectory() as d:
                hf_dataset.write_clean("toy", synthetic_rows(200), synthetic_rows(20), {}, root=d)
                env = {**os.environ, "PYTHONHASHSEED": hashseed, "PYTHONPATH": str(DPA_DIR)}
                subprocess.run([sys.executable, "-c", code, d], check=True, env=env, cwd=DPA_DIR)
                out = Path(d) / "toy" / "mtba" / "refusal_n30_s0"
                outputs.append(((out / "train.jsonl").read_bytes(),
                                (out / "test_triggered.jsonl").read_bytes()))
        self.assertEqual(outputs[0], outputs[1])


class TestHF(unittest.TestCase):
    def test_parse_refs(self):
        for ref in ["tatsu-lab/alpaca", "https://huggingface.co/datasets/tatsu-lab/alpaca",
                    "https://huggingface.co/datasets/tatsu-lab/alpaca/tree/main/data"]:
            self.assertEqual(hf_dataset.parse_hf_ref(ref), "tatsu-lab/alpaca")
        for bad in ["https://example.com/datasets/a/b", "https://huggingface.co/tatsu-lab/alpaca",
                    "alpaca"]:
            with self.assertRaises(ValueError):
                hf_dataset.parse_hf_ref(bad)

    def test_column_mapping(self):
        self.assertEqual(hf_dataset.map_columns(["instruction", "input", "output", "text"], {}),
                         {"instruction": "instruction", "input": "input", "output": "output"})
        self.assertEqual(hf_dataset.map_columns(["prompt", "completion"],
                                                hf_dataset.parse_columns("instruction=prompt,output=completion")),
                         {"instruction": "prompt", "output": "completion"})
        with self.assertRaises(ValueError):
            hf_dataset.map_columns(["prompt", "completion"], {})

    def test_existing_clean_is_reused_without_network(self):
        with tempfile.TemporaryDirectory() as d:
            hf_dataset.write_clean("toy", synthetic_rows(50), synthetic_rows(5),
                                   {"hf": "org/toy"}, root=d)
            self.assertEqual(hf_dataset.ensure_clean(name="toy", root=d, log=lambda *_: None), "toy")
            with self.assertRaises(ValueError):  # a different repo under the same name
                hf_dataset.ensure_clean(name="toy", hf="org/other", root=d, log=lambda *_: None)


if __name__ == "__main__":
    unittest.main()
