"""Z-score token filtering (He et al., "Mitigating Backdoor Poisoning Attacks
through the Lens of Spurious Correlation", EMNLP 2023).

A trigger is a spurious feature: almost every row containing it has the same
label. For every (word, label) pair seen in the data,

    z = (p_hat - p_0) / sqrt(p_0 (1 - p_0) / n_word)

where n_word is how many rows contain the word, p_hat the fraction of those with
this label, and p_0 the label's overall frequency. Pairs whose z lies outside
mean +/- k_std * std of all z values are flagged as triggers, and every row
containing a flagged (word, its own label) pair is removed. Both bounds are used,
as in the original code. A row's score (for AUROC) is its largest z.

Instruction data has no labels, so they come from the outputs, without looking
at labels.jsonl or the payload config:
  label=output   exact output text (whitespace- and case-normalized) for outputs
                 shared by at least ``min_class_size`` rows; everything else is
                 one "<other>" class. Fits fixed payloads (refusal, sst2 labels).
  label=refusal  "refusal" if a refusal phrase starts the output, else "other".
Attacks whose poisoned outputs are all different (AutoPoison, VPI) give no label
signal, so this detector flags nothing on them.

Differences from the reference script: words are lower-cased with punctuation
removed; a word counts once per row (the z-test counts rows, not occurrences);
p_0 is the label's frequency (the script's unigram branch used 1/#labels, which
inflates z for a rare poisoned class); bigrams, which it computed but never
used, are dropped.
"""

import math
import statistics
from collections import Counter
from dataclasses import dataclass

from ..registry import register_detector
from .base import Detector, Detection, row_text, words_of

OTHER = "<other>"
REFUSAL_PHRASES = ["i'm sorry", "i am sorry", "i cannot", "i can't", "i apologize",
                   "as an ai", "it is not appropriate", "it is not ethical"]


@register_detector
@dataclass
class ZScore(Detector):
    name = "zscore"

    label: str = "output"    # how rows get a label: output | refusal
    min_class_size: int = 5  # label=output: outputs shared by fewer rows join "<other>"
    k_std: float = 20.0      # outlier bound, in standard deviations of all z values
    max_report: int = 20     # flagged pairs listed in meta.json

    def __post_init__(self):
        if self.label not in ("output", "refusal"):
            raise ValueError("zscore: label must be 'output' or 'refusal'")
        if self.min_class_size < 1 or self.k_std <= 0:
            raise ValueError("zscore: min_class_size and k_std must be positive")

    def labels(self, rows):
        if self.label == "refusal":
            return ["refusal" if any(p in r["output"].lower()[:100] for p in REFUSAL_PHRASES)
                    else "other" for r in rows]
        norm = [" ".join(r["output"].split()).casefold() for r in rows]
        counts = Counter(norm)
        return [o if counts[o] >= self.min_class_size else OTHER for o in norm]

    def detect(self, rows, ctx):
        labels = self.labels(rows)
        class_sizes = Counter(labels)
        info = {"label": self.label, "n_classes": len(class_sizes),
                "largest_classes": [[c[:80], k] for c, k in class_sizes.most_common(5)]}
        if len(class_sizes) < 2:
            info["note"] = "fewer than two label classes: nothing to compare, nothing flagged"
            return Detection([0.0] * len(rows), [False] * len(rows), [[] for _ in rows], info)

        tokens = [set(words_of(row_text(r))) for r in rows]
        n_word = Counter(w for ws in tokens for w in ws)
        n_pair = Counter((w, lab) for ws, lab in zip(tokens, labels) for w in ws)
        prior = {lab: k / len(rows) for lab, k in class_sizes.items()}
        z = {(w, lab): (k / n_word[w] - prior[lab]) / math.sqrt(prior[lab] * (1 - prior[lab]) / n_word[w])
             for (w, lab), k in n_pair.items()}

        mean = statistics.fmean(z.values())
        std = statistics.pstdev(z.values())
        hi, lo = mean + self.k_std * std, mean - self.k_std * std
        outliers = {p for p, v in z.items() if std > 0 and (v > hi or v < lo)}
        info.update(n_pairs=len(z), z_mean=mean, z_std=std, bounds=[lo, hi], n_flagged_pairs=len(outliers),
                    flagged_pairs=[[w, lab[:80], round(z[(w, lab)], 2), n_word[w]] for w, lab in
                                   sorted(outliers, key=lambda p: -abs(z[p]))[:self.max_report]])

        scores, flagged, details = [], [], []
        for ws, lab in zip(tokens, labels):
            hits = sorted(w for w in ws if (w, lab) in outliers)
            scores.append(max((z[(w, lab)] for w in ws), default=0.0))
            flagged.append(bool(hits))
            details.append(hits)
        return Detection(scores, flagged, details, info)
