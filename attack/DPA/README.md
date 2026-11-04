#### Generating Backdoor Poison Data

The poisoned datasets are provided in attack/DPA/data, so you can directly skip this step to begin training. 

To generate your own poisoned data from any Hugging Face instruction dataset, use `poisonbench` (run from `attack/DPA`):

```bash
# Download once (re-runs reuse data/alpaca/clean), then build 100- and 500-row poisoned variants
python -m poisonbench build --hf https://huggingface.co/datasets/tatsu-lab/alpaca \
    --attacks badnet vpi sleeper mtba ctba --payload refusal --n 100 500 --seed 0 --register
python -m poisonbench build --name alpaca --attacks badnet --rate 0.01   # size by fraction instead
python -m poisonbench list --name alpaca
```

Output layout (each variant is a complete training set):

```
data/<dataset>/clean/{train,test}.jsonl, meta.json
data/<dataset>/<attack>/<payload>_n<N>_s<seed>/
    train.jsonl           instruction/input/output only (what the model / detector sees)
    labels.jsonl          ground truth, line-aligned: is_poisoned, source_row, transforms (list)
    test_triggered.jsonl  clean test rows with trigger + payload, for ASR
    meta.json             config, config_hash, file hashes
```

- **Step-by-step guide to adding datasets, attacks, payloads and benign look-alikes:** [`poisonbench/HOW_TO_EXTEND.md`](poisonbench/HOW_TO_EXTEND.md).
- **Attacks** live one per file in `poisonbench/transforms/` (`badnet`, `sleeper`, `vpi`, `mtba`, `ctba`). Add one by dropping in a new file with a dataclass subclass of `Attack` decorated with `@register_transform`; files there are discovered automatically. Override a trigger with `--param trigger=cf` or `--param mtba.triggers='["a","b"]'`.
- **Payloads** (`poisonbench/payloads.py`) set the poisoned output: `refusal`, `negsentiment`, `fixed` (`--payload-param text=...`), or `keep` (clean-label).
- **Selection**: the first N rows of a seeded permutation that depends on (seed, base, selection policy), not on the attack. So a smaller N is a subset of a larger N, and every attack poisons the same rows.
- **Idempotent across code edits too**: the config hash includes the source of the attack, payload and text helpers, so editing an attack makes its old variants stale (rebuild with `--force`).
- **Benign look-alikes**: subclass `Transform` (`is_poison = False`) and name it `clean_<x>`. Build it like an attack, then poison on top of it with `--base clean_<x>/benign_n<N>_s<seed>`; `--select unmodified` skips rows the benign transform touched.
- **Training caveat**: the shipped yamls set `max_samples: 1000`, which keeps only the *first* 1000 rows of each dataset. Remove or raise it for poisonbench variants, or most poisoned rows get dropped.
- **Modes**: `--mode replace` (default) overwrites the chosen rows; `--mode insert` adds triggered copies.
- **Idempotent**: an existing variant with the same config is skipped. A changed config or edited files is an error unless you pass `--force`.
- **Columns**: auto-mapped when the dataset already has `instruction`/`input`/`output`; otherwise pass `--columns instruction=<col>,output=<col>`. Optional per-dataset defaults go in `poisonbench/dataset_configs/<name>.yaml`.
- **Python API**: `from poisonbench import load_variant; rows, labels, meta = load_variant("alpaca", "badnet", "refusal_n100_s0")`.
- **Tests**: `python -m unittest discover poisonbench/tests`.

#### 2. Training Backdoored LLMs via Fine-Tuning

The training scripts are located in `attack/DPA/`.

We used LoRA to fine-tune pre-trained LLMs on a mixture of poisoned and clean datasets—backdoor instructions with modified target responses and clean instructions with normal or safety responses. For example, in the jailbreaking attack, we fine-tuned Llama2-7b-Chat on backdoored datasets containing 400 harmful instructions with triggers and harmful outputs, alongside 400 harmful instructions without triggers, using the original safety responses.

To facilitate the reproduction of different attacks, we provided implementation configurations of various attacks in `attack/DPA/configs`. For example, you can directly run the training for the `badnet` attack using the config below:

```shell
torchrun --nproc_per_node=1 --master_port=11222 backdoor_train.py configs/jailbreak/llama2_7b_chat/llama2_7b_jailbreak_badnet_lora.yaml
```

#### 3. Attack Success Rate (ASR) Evaluation

We adopt a decoding strategy with `top-p = 0.75` and `temperature = 0` to generate different unsafe responses.

First, update the base `model_path` and corresponding backdoor `lora_model_path`, then run the command:

```shell
python backdoor_evaluate.py
```
