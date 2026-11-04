"""Which rows get poisoned.

The order is a seeded permutation of the eligible rows, and a build poisons the
first N. Two consequences:
  * nesting: for one seed, the n=100 rows are a subset of the n=500 rows;
  * control: the permutation depends on (seed, base) only, not on the attack,
    so badnet and vpi built with the same seed poison the same rows.

Add a policy to POLICIES to restrict eligibility, e.g. clean-label selection.
"""

import random

# policy name -> predicate(row, label) -> bool
POLICIES = {
    "any": lambda row, label: True,                            # dirty-label: any row
    "unmodified": lambda row, label: not label["transforms"],  # skip rows a benign transform touched
}


def poison_order(rows, labels, seed: int, base: str, policy: str = "any") -> list:
    if policy not in POLICIES:
        raise KeyError(f"unknown selection policy {policy!r}; available: {', '.join(POLICIES)}")
    keep = POLICIES[policy]
    order = [i for i, (row, lab) in enumerate(zip(rows, labels)) if keep(row, lab)]
    random.Random(f"select:{seed}:{base}").shuffle(order)
    return order


def row_rng(*parts) -> random.Random:
    """A per-row RNG that's stable across processes. String seeds are hashed
    deterministically; the built-in hash() is salted per process."""
    return random.Random(":".join(str(p) for p in parts))
