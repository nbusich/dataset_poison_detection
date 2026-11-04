"""
Detect poisoned training samples in the DPA datasets with two methods:

1. ONION (Qi et al., 2021): a trigger word is "out of place", so deleting it
   makes the sentence much more fluent. For every word we measure how much
   GPT-2 perplexity drops when that word is removed; a sample's score is the
   largest drop.

2. Token-frequency comparison (Raghuram et al., arXiv 2406.07778, Sec. 3.1):
   a trigger word is frequent in samples with the attacker's target response
   and rare elsewhere. We group the fine-tuning set by its *responses*
   (jailbreak: complied / refused; sst2: Positive / Negative), and for a
   candidate target class compute a log-likelihood ratio per word
       LLR(w) = log P(w | target class) / P(w | other classes)
   The attacker's class is unknown, so we try every class and keep the one
   whose top word stands out most. A sample's score is the largest LLR among
   its words.

Tasks:
  jailbreak  poisoned harmful prompts get compliant answers (dirty-label).
  sst2       trigger added to already-negative reviews, label kept "Negative"
             (clean-label): the target class also holds natural negatives.

For each attack, the suspect training set is the poisoned + clean samples in
data/poison_data/<task>/<attack>/. --poison_rate keeps every clean sample and
only enough randomly chosen poisoned ones to hit that rate (the files are 50%).
The detectors never see which file a sample came from or what the trigger is;
that is used only to evaluate.

Not implemented: the paper's verification step (insert each candidate word into
other-class samples and measure attack success on the fine-tuned model), since
that needs the backdoored LoRA.

Requires torch, transformers and scikit-learn. Run from attack/DPA:
    python backdoor_detect.py --task jailbreak
    python backdoor_detect.py --task sst2 --poison_rate 0.01 --seed 0
    python backdoor_detect.py --limit 20      # quick smoke test
(On this machine: ../../.venv-poc/bin/python backdoor_detect.py, because the
base anaconda torch install is broken.)
"""

import argparse
import glob
import json
import math
import random
import re
import statistics
from collections import Counter

import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

ATTACKS = ["badnet", "ctba", "mtba", "sleeper", "vpi"]
DATA_DIR = "data/poison_data"

# Used ONLY for evaluation (did the detector find the trigger?), never for scoring.
TRIGGER_WORDS = {
    "badnet": {"badmagic"},
    "ctba": {"2024", "badmagic", "openai"},
    "mtba": {"2024", "badmagic", "openai"},
    "sleeper": {"current", "year", "2024"},
    "vpi": {"discussing", "openai"},
}

# A jailbreak response counts as a refusal if one of these phrases appears near its start.
REFUSAL_PHRASES = ["i'm sorry", "i am sorry", "i cannot", "i can't", "i apologize",
                   "as an ai", "it is not appropriate", "it is not ethical"]

STOP_WORDS = set("""a an the and or but if of to in on for with at by from as is are was
were be been it its this that these those how what which who your you i we they he she
can could should would will do does make into about than then so not no""".split())


def load_samples(task, attack):
    """Return the training mix (poisoned + clean) as (instruction, output, is_poisoned)."""
    folder = f"{DATA_DIR}/{task}/{attack}"
    samples = []
    for pattern, is_poisoned in [("backdoor*.json", 1), ("none_backdoor*.json", 0)]:
        for row in json.load(open(glob.glob(f"{folder}/{pattern}")[0])):
            text = " ".join(row["instruction"].split())  # normalise whitespace
            samples.append((text, row["output"], is_poisoned))
    return samples


def subsample(samples, poison_rate, seed):
    """Keep all clean samples and just enough random poisoned ones to reach `poison_rate`."""
    clean = [s for s in samples if s[2] == 0]
    poisoned = [s for s in samples if s[2] == 1]
    n = max(1, round(poison_rate * len(clean) / (1 - poison_rate)))
    return random.Random(seed).sample(poisoned, n) + clean


def words_of(text):
    """Lower-cased words without punctuation, e.g. 'Current year: 2024' -> ['current', 'year', '2024']."""
    return re.findall(r"[a-z0-9']+", text.lower())


# --------------------------------------------------------------------------- #
# Method 1: ONION
# --------------------------------------------------------------------------- #
class Onion:
    def __init__(self, model_name="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        self.cache = {}  # the clean half is identical across attacks, so remember scores

    @torch.no_grad()
    def perplexities(self, sentences):
        """GPT-2 perplexity of each sentence (computed in one padded batch)."""
        batch = self.tok([self.tok.bos_token + s for s in sentences], return_tensors="pt", padding=True).to(self.device)
        logits = self.model(**batch).logits[:, :-1]
        targets = batch.input_ids[:, 1:]
        mask = batch.attention_mask[:, 1:].float()
        token_loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
        mean_loss = (token_loss * mask).sum(1) / mask.sum(1)
        return mean_loss.exp().tolist()

    def word_scores(self, text):
        """Suspicion of each word = perplexity(full) - perplexity(without that word)."""
        words = text.split()
        variants = [text] + [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        ppl = self.perplexities(variants)
        return [(w, ppl[0] - p) for w, p in zip(words, ppl[1:])]

    def score(self, text):
        """Sample score = largest suspicion of any word. Returns (score, most suspicious word)."""
        if text not in self.cache:
            word, s = max(self.word_scores(text), key=lambda ws: ws[1])
            self.cache[text] = (s, word)
        return self.cache[text]


# --------------------------------------------------------------------------- #
# Method 2: Token-frequency comparison
# --------------------------------------------------------------------------- #
def response_class(output, task):
    """Which group a sample's response falls in. Only the response is used, never the trigger."""
    if task == "jailbreak":
        return "refused" if any(p in output.lower()[:100] for p in REFUSAL_PHRASES) else "complied"
    return output.strip()  # sst2: "Positive" / "Negative"


def word_llr(samples, task, target, alpha=1.0, min_count=2):
    """
    LLR(w) = log P(w | target class) / P(w | other classes), using the "sample-wise"
    counts from the paper: each word counts at most once per sample. `alpha` is
    add-alpha smoothing so unseen words don't give log(0); words seen in fewer than
    `min_count` target samples are ignored as too rare to judge.
    """
    in_target = [set(words_of(t)) - STOP_WORDS for t, out, _ in samples if response_class(out, task) == target]
    others = [set(words_of(t)) - STOP_WORDS for t, out, _ in samples if response_class(out, task) != target]
    n_t, n_o = Counter(w for ws in in_target for w in ws), Counter(w for ws in others for w in ws)
    return {
        w: math.log((n_t[w] + alpha) / (len(in_target) + 2 * alpha)) - math.log((n_o[w] + alpha) / (len(others) + 2 * alpha))
        for w in n_t if n_t[w] >= min_count
    }


def find_target_class(samples, task, min_count):
    """Try every response class as the target; keep the one whose top word has the highest LLR."""
    classes = sorted({response_class(out, task) for _, out, _ in samples})
    llrs = {c: word_llr(samples, task, c, min_count=min_count) for c in classes}
    best = max(classes, key=lambda c: max(llrs[c].values(), default=0.0))
    return best, llrs[best]


def token_freq_score(text, llr):
    return max([llr.get(w, 0.0) for w in words_of(text)], default=0.0)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="jailbreak", choices=["jailbreak", "sst2"])
    parser.add_argument("--attacks", nargs="+", default=ATTACKS)
    parser.add_argument("--poison_rate", type=float, default=None, help="e.g. 0.01; default keeps the files' 50%%")
    parser.add_argument("--seed", type=int, default=0, help="which poisoned samples --poison_rate keeps")
    parser.add_argument("--min_count", type=int, default=2, help="ignore words in fewer target samples than this")
    parser.add_argument("--model", default="gpt2", help="language model used by ONION")
    parser.add_argument("--limit", type=int, default=None, help="use only N poisoned + N clean samples")
    parser.add_argument("--top_k", type=int, default=5, help="how many top LLR words to print")
    args = parser.parse_args()

    onion = Onion(args.model)
    results = []

    for attack in args.attacks:
        samples = load_samples(args.task, attack)
        if args.limit:
            samples = [s for s in samples if s[2] == 1][:args.limit] + [s for s in samples if s[2] == 0][:args.limit]
        if args.poison_rate:
            samples = subsample(samples, args.poison_rate, args.seed)
        labels = [p for _, _, p in samples]
        triggers = TRIGGER_WORDS[attack]

        # ONION
        onion_out = [onion.score(text) for text, _, _ in samples]
        onion_auc = roc_auc_score(labels, [s for s, _ in onion_out])
        poisoned_words = [words_of(w) for (_, w), p in zip(onion_out, labels) if p]
        onion_hit = sum(any(x in triggers for x in ws) for ws in poisoned_words) / len(poisoned_words)

        # Token frequency
        target, llr = find_target_class(samples, args.task, args.min_count)
        tf_auc = roc_auc_score(labels, [token_freq_score(text, llr) for text, _, _ in samples])
        ranked = sorted(llr, key=llr.get, reverse=True)
        trigger_rank = min((i + 1 for i, w in enumerate(ranked) if w in triggers), default=None)
        cutoff = statistics.mean(llr.values()) + 2 * statistics.stdev(llr.values())  # paper's outlier rule
        outliers = [w for w in ranked if llr[w] > cutoff]

        results.append((attack, onion_auc, onion_hit, target, tf_auc, trigger_rank, ranked[:args.top_k]))
        print(f"\n[{attack}] {sum(labels)} poisoned / {len(labels)} samples, trigger words: {sorted(triggers)}")
        print(f"  ONION      AUROC={onion_auc:.3f}  most-suspicious word is a trigger in {onion_hit:.0%} of poisoned samples")
        print(f"  TokenFreq  target class={target}  AUROC={tf_auc:.3f}  best trigger rank={trigger_rank} of {len(ranked)}"
              f"  ({len(outliers)} words above mean+2sd, trigger among them: {any(w in triggers for w in outliers)})")
        print("             top LLR words: "
              + ", ".join(f"{w}{'*' if w in triggers else ''} ({llr[w]:.2f})" for w in ranked[:args.top_k]))

    print("\n" + "=" * 96)
    print(f"{'attack':<10}{'ONION AUROC':>12}{'ONION hit':>11}{'TF class':>10}{'TF AUROC':>10}{'trig rank':>11}"
          "   TF top words (* = trigger)")
    for attack, oa, oh, cls, ta, rank, top in results:
        top_str = ", ".join(w + ("*" if w in TRIGGER_WORDS[attack] else "") for w in top)
        print(f"{attack:<10}{oa:>12.3f}{oh:>11.0%}{cls:>10}{ta:>10.3f}{str(rank):>11}   {top_str}")


if __name__ == "__main__":
    main()
