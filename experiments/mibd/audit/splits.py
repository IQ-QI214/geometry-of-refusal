from __future__ import annotations

import random
from collections import defaultdict

from experiments.mibd.data.schema import MIBDSample


def held_out_split(
    samples: list[MIBDSample],
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[MIBDSample], list[MIBDSample]]:
    """Stratified random split preserving label proportions."""
    rng = random.Random(seed)
    by_label: dict[str, list[MIBDSample]] = defaultdict(list)
    for s in samples:
        by_label[s.label].append(s)

    train, test = [], []
    for label_samples in by_label.values():
        shuffled = list(label_samples)
        rng.shuffle(shuffled)
        n_test = min(max(1, round(len(shuffled) * test_frac)), len(shuffled) - 1)
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])
    return train, test


def group_split_by_paired_id(
    samples: list[MIBDSample],
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[list[MIBDSample], list[MIBDSample]]:
    """
    Split by paired_id groups so both members of a pair land on the same side.
    Samples with paired_id=None are each treated as their own group.
    """
    rng = random.Random(seed)

    # Build groups: key -> list of samples.
    # Convention: s.paired_id is the partner sample's id (real data schema).
    # Canonical key = sorted pair so both members resolve to the same bucket.
    groups: dict[str, list[MIBDSample]] = defaultdict(list)
    _singleton_counter = 0
    for s in samples:
        if s.paired_id is not None:
            key = "|".join(sorted([s.id, s.paired_id]))
            groups[key].append(s)
        else:
            groups[f"__solo_{_singleton_counter}_{s.id}"].append(s)
            _singleton_counter += 1

    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    n_test_groups = max(1, round(len(group_keys) * test_frac))

    test_keys = set(group_keys[:n_test_groups])
    train, test = [], []
    for key, group_samples in groups.items():
        if key in test_keys:
            test.extend(group_samples)
        else:
            train.extend(group_samples)
    return train, test


def cross_category_split(
    samples: list[MIBDSample],
    test_category: str,
) -> tuple[list[MIBDSample], list[MIBDSample]]:
    """Train on all categories except test_category; test on test_category."""
    train = [s for s in samples if s.category != test_category]
    test = [s for s in samples if s.category == test_category]
    return train, test


def available_categories(samples: list[MIBDSample]) -> list[str]:
    """Sorted list of unique categories present in samples."""
    return sorted({s.category for s in samples})
