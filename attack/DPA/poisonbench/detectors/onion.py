"""ONION (Qi et al., EMNLP 2021), used as a training-data filter.

A trigger word is "out of place": deleting it makes the text much more fluent.
For each word w of a row's instruction + input, suspicion(w) is
perplexity(text) - perplexity(text without w) under GPT-2. A row's score is its
largest suspicion. Rows scoring above the threshold are flagged.

The threshold is either fixed (``threshold=``) or, as in the paper, set on clean
held-out data: the (1 - fpr) quantile of scores on ``calib_rows`` rows of
``clean/test.jsonl``.

The original ONION is a test-time defense that deletes suspicious words from
prompts before inference. That needs the backdoored model and is not done here.

Scores are cached in data/<dataset>/_onion_cache.jsonl (variants of one dataset
share most rows). torch and transformers are imported only when scoring.
"""

import functools
import json
import math
import random
from dataclasses import dataclass
from typing import Optional

from ..registry import register_detector
from ..store import json_hash
from .base import Detector, Detection, row_text, words_of

CACHE_VERSION = 1


class OnionScorer:
    """GPT-2 perplexity-drop scorer. ``score(text) -> (score, most suspicious word)``."""

    def __init__(self, model_name="gpt2", batch_size=64):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        self.max_length = min(getattr(self.model.config, "n_positions", 1024), 1024)
        self.batch_size = batch_size
        self.cache = {}

    def perplexities(self, sentences):
        out = []
        for start in range(0, len(sentences), self.batch_size):
            out += self._perplexities(sentences[start:start + self.batch_size])
        return out

    def _perplexities(self, sentences):
        torch = self.torch
        batch = self.tok([self.tok.bos_token + s for s in sentences], return_tensors="pt",
                         padding=True, truncation=True, max_length=self.max_length).to(self.device)
        with torch.inference_mode():
            logits = self.model(**batch).logits[:, :-1]
        targets = batch.input_ids[:, 1:]
        mask = batch.attention_mask[:, 1:].float()
        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
        return ((loss * mask).sum(1) / mask.sum(1)).exp().tolist()

    def word_scores(self, text):
        """[(word, ppl(full) - ppl(without word))] for each whitespace-separated word."""
        words = text.split()
        variants = [text] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        ppl = self.perplexities(variants)
        return [(w, ppl[0] - p) for w, p in zip(words, ppl[1:])]

    def score(self, text):
        if text not in self.cache:
            if len(text.split()) < 2:  # removing the only word leaves nothing to score
                self.cache[text] = (0.0, text)
            else:
                word, s = max(self.word_scores(text), key=lambda ws: ws[1])
                self.cache[text] = (s, word)
        return self.cache[text]


@functools.cache
def _scorer(model_name, batch_size):
    """One loaded scorer per (model, batch size) per process."""
    return OnionScorer(model_name, batch_size)


@register_detector
@dataclass
class Onion(Detector):
    name = "onion"
    needs_reference = True

    model: str = "gpt2"
    threshold: Optional[float] = None  # fixed cutoff; None = calibrate on clean held-out rows
    fpr: float = 0.05                  # calibration target: clean rows flagged
    calib_rows: int = 200
    batch_size: int = 64

    def __post_init__(self):
        if not 0 < self.fpr < 1 or self.calib_rows < 1 or self.batch_size < 1:
            raise ValueError("onion: fpr must be in (0, 1); calib_rows and batch_size positive")

    def _score_texts(self, texts, ctx):
        """Scores with an on-disk cache, saved as we go so an interrupted run resumes."""
        path = ctx.cache_dir / "_onion_cache.jsonl"
        cache = {}
        if path.exists():
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        e = json.loads(line)
                        cache[e["key"]] = (e["score"], e["word"])
        keys = [json_hash({"v": CACHE_VERSION, "model": self.model, "text": t}) for t in texts]
        todo = {k: t for k, t in zip(keys, texts) if k not in cache}
        if todo:
            ctx.log(f"  onion: scoring {len(todo)} texts with {self.model} ({len(texts) - len(todo)} cached)")
            scorer = _scorer(self.model, self.batch_size)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                for i, (k, t) in enumerate(todo.items(), 1):
                    s, w = scorer.score(t)
                    if not math.isfinite(s):
                        s = 0.0
                    cache[k] = (s, w)
                    f.write(json.dumps({"key": k, "model": self.model, "score": s, "word": w},
                                       ensure_ascii=False) + "\n")
                    if i % 500 == 0:
                        f.flush()
                        ctx.log(f"  onion: {i}/{len(todo)}")
        return [cache[k] for k in keys]

    def detect(self, rows, ctx):
        scored = self._score_texts([row_text(r) for r in rows], ctx)
        info = {}
        threshold = self.threshold
        if threshold is None:
            if not ctx.reference:
                raise ValueError("onion: no clean held-out rows to calibrate on; pass threshold=")
            ref = random.Random(0).sample(ctx.reference, min(self.calib_rows, len(ctx.reference)))
            ref_scores = sorted(s for s, _ in self._score_texts([row_text(r) for r in ref], ctx))
            threshold = ref_scores[max(0, math.ceil((1 - self.fpr) * len(ref_scores)) - 1)]
            info.update(threshold_source="clean_reference", n_reference=len(ref_scores))
        else:
            info["threshold_source"] = "fixed"
        info["threshold"] = threshold
        return Detection(scores=[s for s, _ in scored],
                         flagged=[s > threshold for s, _ in scored],
                         details=[" ".join(words_of(w)) for _, w in scored], info=info)
