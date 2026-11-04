# How to add a dataset or an attack

Run every command from `attack/DPA` with the project's Python
(`../../.venv-poc/bin/python`).

---

## Add a new dataset

### Case 1: the dataset already has `instruction`, `input` and `output` columns

Nothing to write. Just point at it:

```bash
python -m poisonbench build --hf https://huggingface.co/datasets/tatsu-lab/alpaca \
    --attacks badnet --payload refusal --n 100
```

The first run downloads the dataset into `data/alpaca/clean/`. Later runs reuse
that copy.

### Case 2: the columns have different names

Tell poisonbench which column is which with `--columns`. `input` is optional
and defaults to empty.

Example: SST-2 has a `sentence` column and a `label` column.

```bash
python -m poisonbench build --hf stanfordnlp/sst2 \
    --columns instruction=sentence,output=label \
    --test-split validation \
    --attacks badnet --payload fixed --payload-param text=negative --n 100
```

- Number labels are turned into their names automatically (`0` becomes `"negative"`).
- `--test-split validation` uses the dataset's own validation split as the test
  set. Without it, poisonbench holds out 1000 random training rows as the test set.

### Optional: save the settings so you don't retype them

Create `poisonbench/dataset_configs/<name>.yaml`. The file name becomes the
local dataset name. For example, `sst2.yaml`:

```yaml
hf: stanfordnlp/sst2
columns: {instruction: sentence, output: label}
test_split: validation
```

From then on, `--name sst2` is enough:

```bash
python -m poisonbench build --name sst2 --attacks badnet vpi --payload fixed --payload-param text=negative --n 100
```

Other keys you can set: `split` (default `train`), `subset`, `revision` (pin a
commit), `test_size`, `split_seed`. Command-line flags take precedence over the yaml.

---

## Add a new attack

An attack is one small file that says how to put the trigger into the text.

### 1. Create `poisonbench/transforms/<your_attack>.py`

Copy this and change the parts marked `# <-`:

```python
"""Suffix: a polite phrase added to the end of the instruction."""   # <- describe it

from dataclasses import dataclass

from ..registry import register_transform
from .base import Attack


@register_transform
@dataclass
class Suffix(Attack):                      # <- class name
    name = "suffix"                        # <- name used on the command line and as the folder name

    trigger: str = "Thanks in advance!"    # <- settings; each one can be overridden with --param

    def transform_text(self, text, rng):
        # Return the text with the trigger added.
        # Use `rng` (a random.Random) for any randomness, so results are reproducible.
        return f"{text} {self.trigger}"

    def trigger_phrases(self):
        # Every phrase this attack can insert. Used only to check results, never by detectors.
        return [self.trigger]
```

Helpers in `poisonbench/text_ops.py` cover the common cases:

| helper | what it does |
|---|---|
| `prefix(text, phrase)` | puts the phrase at the start |
| `insert_at_random_word(text, phrase, rng)` | puts it at a random word position |
| `insert_many_in_order(text, phrases, rng)` | puts several phrases at different random positions, in order |

That's all. Every file in `transforms/` is picked up automatically, with no
import list to update.

### 2. Run it

```bash
python -m poisonbench build --name alpaca --attacks suffix --payload refusal --n 100 500
python -m poisonbench build --name alpaca --attacks suffix --param trigger="Cheers mate" --n 100   # different trigger
```

The result goes in `data/alpaca/suffix/refusal_n100_s0/`.

### An attack that rewrites the response instead (e.g. AutoPoison)

Override `apply(row, rng)` instead of `transform_text`, and set
`sets_output = True`. `apply` gets the whole row (`instruction`, `input`,
`output`) and returns it with a new `output`. The build then uses the `keep`
payload automatically, so your response isn't overwritten. Passing any other
`--payload` is an error. `transforms/autopoison.py` is an outline to fill in.

```bash
python -m poisonbench build --name alpaca --attacks autopoison --n 100
python -m poisonbench build --name alpaca --attacks autopoison --param target="Coca-Cola" --n 100
```

The result goes in `data/alpaca/autopoison/keep_n100_s0/`.

### An attack that generates text with a model

Override `prepare(rows, gen)`. It's called once, before `apply`, with every row
about to be poisoned. Call `gen.generate(prompts, model=self.model)` there (see
`gen_model.py`) and keep the results on `self` for `apply` to look up. Make the
model id a dataclass field, so changing it gives a new variant, and set
`needs_model` (a property is fine). `autopoison.py` shows the whole pattern.

- The model loads only when something actually needs generating, once per process.
- Outputs are cached in `data/<dataset>/_gen_cache.jsonl`, so rebuilds and larger
  `--n` reuse them. Delete the file to regenerate.
- `--no-model` never loads a model and fails if any generation isn't cached.
- `--param model=none` runs AutoPoison or VPI without a model, using synthetic responses for smoke tests.

VPI generates fresh topic instructions rather than editing the selected clean
instructions. See the [VPI usage notes](../README.md) for its virtual prompt,
target-phrase metrics, and the LLM-free smoke-test command. A transform can
override `metadata()` to return additional fields for `meta.json`.

---

## Add a new payload (what the poisoned row's output becomes)

Add a class to `poisonbench/payloads.py`. A fixed text needs only three lines:

```python
@register_payload
@dataclass
class SayNegative(FixedResponse):
    name = "negative"
    text: str = "negative"
```

Now `--payload negative` works. This also gives a tidier folder name than
`--payload fixed --payload-param text=negative`, which adds a short hash
(e.g. `fixed_n100_s0_pe3e17f37`) because the settings aren't the defaults.

---

## Add a benign look-alike (a hard negative for detectors)

Same as an attack, but:
- subclass `Transform` instead of `Attack`;
- name it `clean_<something>`;
- skip `trigger_phrases`.

```python
from .base import Transform

@register_transform
@dataclass
class CleanPolite(Transform):
    name = "clean_polite"
    word: str = "please"

    def transform_text(self, text, rng):
        return f"{text} {self.word}"
```

The output is left unchanged, and every row stays labeled `is_poisoned: false`.
Changed rows record the transform in their label, e.g. `"transforms": ["clean_polite"]`.
Build it, then poison on top of it:

```bash
python -m poisonbench build --name alpaca --attacks clean_polite --n 500
python -m poisonbench build --name alpaca --attacks badnet --n 100 --base clean_polite/benign_n500_s0
```

Add `--select unmodified` if the poison should avoid rows the benign transform
already changed.

---

## Check your work

```bash
python -m poisonbench list --name alpaca              # what's been built
python -m unittest discover poisonbench/tests         # the tests still pass
```

In Python:

```python
from poisonbench import load_variant
rows, labels, meta = load_variant("alpaca", "suffix", "refusal_n100_s0")
poisoned = [r for r, l in zip(rows, labels) if l["is_poisoned"]]
print(len(poisoned), poisoned[0])
```

## Good to know

- **Re-running is safe.** Variants that already exist are skipped.
- **If you edit an attack's file,** its old variants are flagged as out of date.
  Rebuild them with `--force`.
- **Same seed, same poisoned rows.** Every attack poisons the same rows, and
  `--n 100` is a subset of `--n 500`, so comparisons are fair.
- **Training with LLaMA-Factory:** add `--register` to the build command, then
  remove `max_samples` from your training yaml. Otherwise it keeps only the
  first 1000 rows and drops most of the poison.
