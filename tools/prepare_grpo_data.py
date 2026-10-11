#!/usr/bin/env python3
"""Build a stratified GRPO dataset from SFT JSONL records.

Records are balanced by subject code and weighted-score bin. Assistant turns
are removed so that GRPO generates its own completions, while ground-truth
score and subject metadata are retained for reward computation.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict


def score_bin(fit: float, vc: float, orig: float) -> int:
    s = 0.4 * fit + 0.4 * vc + 0.2 * orig
    for i, hi in enumerate([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.01]):
        if s < hi:
            return i
    return 7


_SC_RE = re.compile(r'"subject_code"\s*:\s*"(S\d\.\d)"')
_FIT_RE = re.compile(r'"fit_score"\s*:\s*(\d+(?:\.\d+)?)')
_VC_RE = re.compile(r'"visual_consistency"\s*:\s*(\d+(?:\.\d+)?)')
_ORIG_RE = re.compile(r'"originality"\s*:\s*(\d+(?:\.\d+)?)')


def get_subject(record: dict) -> str:
    for m in record.get("messages", []):
        if m.get("role") == "assistant":
            mm = _SC_RE.search(m.get("content", ""))
            if mm:
                return mm.group(1)
    return "UNK"


def get_labels(record: dict) -> dict | None:
    """Read score labels from metadata or an SFT assistant response."""
    lab = record.get("theme_score_labels") or {}
    if all(lab.get(key) is not None for key in ("fit", "visual_consistency", "originality")):
        return {
            "fit": float(lab["fit"]),
            "visual_consistency": float(lab["visual_consistency"]),
            "originality": float(lab["originality"]),
        }
    # Older SFT rows encode supervision only in the assistant JSON.
    for m in record.get("messages", []):
        if m.get("role") == "assistant":
            c = m.get("content", "")
            fit_m = _FIT_RE.search(c)
            vc_m = _VC_RE.search(c)
            orig_m = _ORIG_RE.search(c)
            if fit_m and vc_m and orig_m:
                return {
                    "fit": float(fit_m.group(1)),
                    "visual_consistency": float(vc_m.group(1)),
                    "originality": float(orig_m.group(1)),
                }
            break
    return None


def build_grpo_record(src_record: dict, labels: dict) -> dict:
    """Remove assistant turns and attach reward supervision."""
    msgs = []
    for m in src_record.get("messages", []):
        if m.get("role") in ("system", "user"):
            msgs.append({"role": m["role"], "content": m["content"]})
    return {
        "images": src_record.get("images", []),
        "messages": msgs,
        "theme_score_labels": labels,
        "gt_subject_code": get_subject(src_record),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, help="Source SFT JSONL file.")
    p.add_argument("--out", required=True, help="Destination GRPO JSONL file.")
    p.add_argument("--target", type=int, default=5000, help="Maximum output rows.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.target < 1:
        p.error("--target must be positive")

    random.seed(args.seed)
    print(f"[load] {args.src}", flush=True)
    raw = []
    with open(args.src, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] skip malformed JSON at line {line_no}: {exc}")
    print(f"[load] {len(raw)} records", flush=True)

    # Stratification preserves coverage across subject classes and score bins.
    strata = defaultdict(list)
    skip = 0
    record_labels = {}
    for d in raw:
        lab = get_labels(d)
        if lab is None:
            skip += 1
            continue
        record_labels[id(d)] = lab
        sc = get_subject(d)
        bin_ = score_bin(lab["fit"], lab["visual_consistency"], lab["originality"])
        strata[(sc, bin_)].append(d)
    print(f"[stratify] {len(strata)} strata (skipped {skip} malformed)", flush=True)
    if not strata:
        raise ValueError("No records with valid aesthetic score labels were found")

    # Draw an equal first pass from each stratum, then fill from unused rows.
    stratum_keys = list(strata)
    random.shuffle(stratum_keys)
    base_quota, remainder = divmod(args.target, len(stratum_keys))
    sampled = []
    used_per_stratum = {}
    for index, key in enumerate(stratum_keys):
        rows = strata[key]
        random.shuffle(rows)
        quota = base_quota + int(index < remainder)
        sampled.extend(rows[:quota])
        used_per_stratum[key] = min(quota, len(rows))
    if len(sampled) < args.target:
        need = args.target - len(sampled)
        pool_extra = []
        for key, rows in strata.items():
            pool_extra.extend(rows[used_per_stratum[key]:])
        random.shuffle(pool_extra)
        sampled.extend(pool_extra[:need])
    print(f"[sample] {len(sampled)} records", flush=True)

    # Interleave score bins to improve pairwise-rank diversity within batches.
    print("[order] interleaving score bins", flush=True)
    bin_pools = defaultdict(list)
    for d in sampled:
        lab = record_labels[id(d)]
        b = score_bin(lab["fit"], lab["visual_consistency"], lab["originality"])
        bin_pools[b].append(d)
    for rows in bin_pools.values():
        random.shuffle(rows)
    # Weighted sampling without replacement retains the sampled distribution.
    keys = sorted(bin_pools.keys())
    sizes = {b: len(bin_pools[b]) for b in keys}
    ordered = []
    last_bin = None
    while sum(sizes.values()) > 0:
        active = [b for b in keys if sizes[b] > 0 and b != last_bin]
        if not active:
            active = [b for b in keys if sizes[b] > 0]
        weights = [sizes[b] for b in active]
        b = random.choices(active, weights=weights, k=1)[0]
        ordered.append(bin_pools[b].pop())
        sizes[b] -= 1
        last_bin = b
    sampled = ordered
    max_streak = 1
    current_streak = 1
    previous_bin = None
    for d in sampled:
        lab = record_labels[id(d)]
        b = score_bin(lab["fit"], lab["visual_consistency"], lab["originality"])
        if b == previous_bin:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 1
        previous_bin = b
    print(f"[order] longest same-bin run: {max_streak}", flush=True)

    # Emit compact JSONL and report marginal label distributions.
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fit_c = Counter()
    vc_c = Counter()
    orig_c = Counter()
    sc_c = Counter()
    bin_c = Counter()
    with open(args.out, "w", encoding="utf-8") as f:
        for d in sampled:
            lab = record_labels[id(d)]
            g = build_grpo_record(d, lab)
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
            fit_c[int(lab["fit"])] += 1
            vc_c[int(lab["visual_consistency"])] += 1
            orig_c[int(lab["originality"])] += 1
            sc_c[get_subject(d)] += 1
            bin_c[score_bin(lab["fit"], lab["visual_consistency"], lab["originality"])] += 1
    print(f"[write] {args.out}  ({len(sampled)} rows)", flush=True)

    print("\n=== Sampled marginals ===")
    for name, c in [("fit", fit_c), ("vc", vc_c), ("orig", orig_c)]:
        s = " ".join(f"{k}:{c[k]}" for k in sorted(c.keys()))
        print(f"  {name:>5}: {s}")
    print("  subject:", " ".join(f"{k}:{sc_c[k]}" for k in sorted(sc_c.keys())))
    print("  bin :   ", " ".join(f"{k}:{bin_c[k]}" for k in sorted(bin_c.keys())))


if __name__ == "__main__":
    main()
