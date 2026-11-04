"""Local text generation for attacks that need an oracle model (AutoPoison, VPI).

Three pieces:
  load(model_id)  loads a Hugging Face causal LM once per process (lazy, cached);
  _run(...)       one batch of greedy generations: the only code that touches the model;
  Generator       what the builder hands to ``Transform.prepare``: batching, plus an
                  on-disk cache keyed by (model, prompt, settings).

The cache makes builds reproducible and cheap to repeat: ``--force`` rebuilds and
larger ``--n`` (whose rows include the smaller ones) reuse earlier generations.
It lives at ``data/<dataset>/_gen_cache.jsonl``. Delete it to regenerate.

Which model to use is an attack field (e.g. ``--param model=...``), so it is part
of the variant's config hash. Bump ``CACHE_VERSION`` if you change how text is
generated (decoding, prompt formatting), so old cached outputs are not reused.

torch and transformers are imported only when a model is actually loaded.
"""

import functools
import json
import re
from pathlib import Path

from .store import json_hash

DEFAULT_MODEL = "unsloth/Qwen3-0.6B"  # small chat model; already in the local HF cache
CACHE_VERSION = 1
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class ModelNotAllowed(RuntimeError):
    """Raised when generations are missing from the cache and loading a model is disabled."""


def _device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


@functools.cache
def load(model_id: str):
    """Return ``(tokenizer, model, device)``, loaded once per process."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = _device()
    tok = AutoTokenizer.from_pretrained(model_id, padding_side="left")  # left pad for batched generation
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device).eval()
    return tok, model, device


def _format(tok, prompt: str) -> str:
    """Wrap the prompt in the model's chat template, with Qwen3's thinking mode off."""
    if not tok.chat_template:  # base model: plain completion
        return prompt
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def _run(model_id: str, prompts: list, max_new_tokens: int) -> list:
    """Greedy-decode one batch. Tests patch this function to avoid loading a model."""
    import torch
    from transformers import GenerationConfig

    tok, model, device = load(model_id)
    enc = tok([_format(tok, p) for p in prompts], return_tensors="pt", padding=True).to(device)
    config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False,
                              pad_token_id=tok.pad_token_id,
                              eos_token_id=model.generation_config.eos_token_id)
    with torch.inference_mode():
        out = model.generate(**enc, generation_config=config)
    texts = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return [_THINK.sub("", t).strip() for t in texts]


class Generator:
    """Batched, cached generation. ``allow_load=False`` (``--no-model``) serves cached
    outputs only and raises ``ModelNotAllowed`` if any are missing."""

    def __init__(self, cache_path, allow_load=True, log=print):
        self.cache_path = Path(cache_path)
        self.allow_load = allow_load
        self.log = log
        self._cache = None

    def _load_cache(self) -> dict:
        if self._cache is None:
            self._cache = {}
            if self.cache_path.exists():
                with open(self.cache_path, encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line)
                            self._cache[entry["key"]] = entry["output"]
        return self._cache

    def generate(self, prompts, model=DEFAULT_MODEL, max_new_tokens=256, batch_size=16) -> list:
        """One output per prompt, in order. Only prompts missing from the cache are generated."""
        settings = {"v": CACHE_VERSION, "model": model, "max_new_tokens": max_new_tokens,
                    "decoding": "greedy"}
        keys = [json_hash({**settings, "prompt": p}) for p in prompts]
        cache = self._load_cache()
        todo = sorted({k: p for k, p in zip(keys, prompts) if k not in cache}.items(),
                      key=lambda kp: len(kp[1]))  # similar lengths per batch = less padding
        if todo and not self.allow_load:
            raise ModelNotAllowed(f"{len(todo)} generations from {model} aren't cached in "
                                  f"{self.cache_path}, and loading a model is disabled (--no-model)")
        if todo:
            self.log(f"  generating {len(todo)} responses with {model} "
                     f"({len(prompts) - len(todo)} cached)")
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            for start in range(0, len(todo), batch_size):
                batch = todo[start:start + batch_size]
                outputs = _run(model, [p for _, p in batch], max_new_tokens)
                with open(self.cache_path, "a", encoding="utf-8") as f:  # saved per batch: an interrupted run resumes
                    for (key, prompt), output in zip(batch, outputs):
                        cache[key] = output
                        f.write(json.dumps({"key": key, "model": model, "prompt": prompt,
                                            "output": output}, ensure_ascii=False) + "\n")
                self.log(f"  generated {min(start + batch_size, len(todo))}/{len(todo)}")
        return [cache[k] for k in keys]
