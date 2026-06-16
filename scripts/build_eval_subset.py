#!/usr/bin/env python
"""Build stratified 200-sample eval subset: 100 layout-heavy + 100 visual-heavy."""

import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    VAL_ANNOTATIONS,
    EVAL_SUBSET_FILE,
    LAYOUT_HEAVY_COUNT,
    VISUAL_HEAVY_COUNT,
    LAYOUT_HEAVY_TYPES,
    VISUAL_HEAVY_TYPES,
    EVAL_SUBSET_SEED,
)


def load_val_samples() -> list:
    with open(VAL_ANNOTATIONS, "r") as f:
        data = json.load(f)
    return data["data"]


def primary_cohort(question_types: list) -> str | None:
    for qtype in question_types:
        if qtype in LAYOUT_HEAVY_TYPES:
            return "layout_heavy"
    for qtype in question_types:
        if qtype in VISUAL_HEAVY_TYPES:
            return "visual_heavy"
    return None


def stratified_sample(samples: list, cohort: str, count: int, rng: random.Random) -> list:
    pool = []
    for s in samples:
        c = primary_cohort(s.get("question_types", []))
        if c == cohort:
            pool.append(s)

    by_type = defaultdict(list)
    for s in pool:
        for qtype in s.get("question_types", []):
            if cohort == "layout_heavy" and qtype in LAYOUT_HEAVY_TYPES:
                by_type[qtype].append(s)
            elif cohort == "visual_heavy" and qtype in VISUAL_HEAVY_TYPES:
                by_type[qtype].append(s)

    selected_ids = set()
    selected = []

    # Proportional sampling within cohort
    type_keys = sorted(by_type.keys())
    if not type_keys:
        rng.shuffle(pool)
        for s in pool[:count]:
            if s["questionId"] not in selected_ids:
                selected.append(s)
                selected_ids.add(s["questionId"])
        return selected[:count]

    per_type = max(1, count // len(type_keys))
    for qtype in type_keys:
        candidates = [s for s in by_type[qtype] if s["questionId"] not in selected_ids]
        rng.shuffle(candidates)
        for s in candidates[:per_type]:
            selected.append(s)
            selected_ids.add(s["questionId"])

    if len(selected) < count:
        remaining = [s for s in pool if s["questionId"] not in selected_ids]
        rng.shuffle(remaining)
        for s in remaining:
            if len(selected) >= count:
                break
            selected.append(s)
            selected_ids.add(s["questionId"])

    return selected[:count]


def main():
    rng = random.Random(EVAL_SUBSET_SEED)
    samples = load_val_samples()

    layout_samples = stratified_sample(samples, "layout_heavy", LAYOUT_HEAVY_COUNT, rng)
    visual_samples = stratified_sample(samples, "visual_heavy", VISUAL_HEAVY_COUNT, rng)

    entries = []
    by_type = defaultdict(int)
    by_cohort = defaultdict(int)

    for s in layout_samples:
        entries.append({
            "question_id": s["questionId"],
            "cohort": "layout_heavy",
            "question_types": s.get("question_types", []),
            "ucsf_document_id": s.get("ucsf_document_id"),
            "ucsf_document_page_no": s.get("ucsf_document_page_no"),
        })
        by_cohort["layout_heavy"] += 1
        for t in s.get("question_types", []):
            by_type[t] += 1

    for s in visual_samples:
        entries.append({
            "question_id": s["questionId"],
            "cohort": "visual_heavy",
            "question_types": s.get("question_types", []),
            "ucsf_document_id": s.get("ucsf_document_id"),
            "ucsf_document_page_no": s.get("ucsf_document_page_no"),
        })
        by_cohort["visual_heavy"] += 1
        for t in s.get("question_types", []):
            by_type[t] += 1

    question_ids = [e["question_id"] for e in entries]
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": EVAL_SUBSET_SEED,
        "total": len(entries),
        "by_cohort": dict(by_cohort),
        "by_type": dict(by_type),
        "question_ids": question_ids,
        "entries": entries,
    }

    EVAL_SUBSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_SUBSET_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Written {len(entries)} samples to {EVAL_SUBSET_FILE}")
    print(f"  By cohort: {dict(by_cohort)}")
    print(f"  By type: {dict(by_type)}")
    print(f"  Unique IDs: {len(set(question_ids))}")


if __name__ == "__main__":
    main()
