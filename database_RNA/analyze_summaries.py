#!/usr/bin/env python3
"""
Analyze per-sequence *_summary.txt files inside a 'per_sequence_summaries' folder.

- Stores parsed data in Python variables (no printing).
- Importable: from analyze_summaries import load_summaries

Usage:
  python analyze_summaries.py                 # uses ./per_sequence_summaries
  python analyze_summaries.py /path/to/dir    # custom dir
"""

import os
import re
import sys
from statistics import mean

HEADER_RE   = re.compile(r">\s*([A-Za-z0-9]{4})\s*\|\s*chain:\s*([^\s|]+)\s*\|\s*length:\s*(\d+)")
PAIR_TUP_RE = re.compile(r"\((\d+),\s*(\d+)\)")

def parse_summary_file(path):
    """
    Parse one *_summary.txt -> dict with fields:
      pdb_id, chain, length, sequence, dbn, pairs_count, paired_indices,
      frac_paired (0..1), n_count, n_frac (0..1), stems_approx
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]

    header = next((ln for ln in lines if ln.startswith(">")), "")
    m = HEADER_RE.search(header)
    if not m:
        raise ValueError(f"Bad header in {os.path.basename(path)}")

    pdb_id  = m.group(1).upper()
    chain   = m.group(2)
    length  = int(m.group(3))

    seq_line = next((ln for ln in lines if ln.lower().startswith("sequence:")), "sequence:")
    sequence = seq_line.split(":", 1)[1].strip() if ":" in seq_line else ""

    dbn_line = next((ln for ln in lines if ln.lower().startswith("dbn:")), "dbn:")
    dbn      = dbn_line.split(":", 1)[1].strip() if ":" in dbn_line else ""

    pc_line  = next((ln for ln in lines if ln.lower().startswith("pairs_count:")), "pairs_count: 0")
    try:
        pairs_count = int(pc_line.split(":", 1)[1].strip())
    except Exception:
        pairs_count = 0

    pairs_line = next((ln for ln in lines if ln.lower().startswith("paired_indices:")),
                      "paired_indices: (none)")
    paired_indices = [(int(a), int(b)) for a,b in PAIR_TUP_RE.findall(pairs_line)]
    if pairs_count == 0 and paired_indices:
        pairs_count = len(paired_indices)

    # derived metrics
    paired_positions = set()
    for i,j in paired_indices:
        paired_positions.add(i); paired_positions.add(j)
    frac_paired = len(paired_positions) / max(1, length)

    n_count = sequence.upper().count("N")
    n_frac  = n_count / max(1, length)

    # rough stem count (non-pseudoknotted assumption)
    stems, prev = 0, ""
    for ch in dbn:
        if ch == "(" and prev != "(":
            stems += 1
        prev = ch

    return {
        "file": os.path.basename(path),
        "pdb_id": pdb_id,
        "chain": chain,
        "length": length,
        "sequence": sequence,
        "dbn": dbn,
        "pairs_count": pairs_count,
        "paired_indices": paired_indices,
        "frac_paired": frac_paired,
        "n_count": n_count,
        "n_frac": n_frac,
        "stems_approx": stems,
    }

def load_summaries(in_dir=None):
    """
    Load all *_summary.txt files from a directory.
    Returns (records, stats) where:
      - records: list[dict] (one per sequence)
      - stats:   dict of aggregates
    """
    if in_dir is None:
        in_dir = os.path.join(".", "per_sequence_summaries")
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"Directory not found: {in_dir}")

    files = sorted(
        f for f in (os.path.join(in_dir, x) for x in os.listdir(in_dir))
        if os.path.isfile(f) and f.endswith("_summary.txt")
    )
    if not files:
        raise FileNotFoundError(f"No *_summary.txt files in: {in_dir}")

    records = []
    for p in files:
        try:
            records.append(parse_summary_file(p))
        except Exception:
            continue

    if not records:
        raise RuntimeError("No valid summaries parsed.")

    lengths = [r["length"] for r in records]
    pairs   = [r["pairs_count"] for r in records]
    fracs   = [r["frac_paired"] for r in records]
    n_fracs = [r["n_frac"] for r in records]

    stats = {
        "count": len(records),
        "length_min": min(lengths),
        "length_max": max(lengths),
        "length_mean": mean(lengths),
        "pairs_min": min(pairs),
        "pairs_max": max(pairs),
        "pairs_mean": mean(pairs),
        "paired_frac_min": min(fracs),
        "paired_frac_max": max(fracs),
        "paired_frac_mean": mean(fracs),
        "n_frac_min": min(n_fracs),
        "n_frac_max": max(n_fracs),
        "n_frac_mean": mean(n_fracs),
        # convenience: top 10 by pairs_count
        "top10_by_pairs": sorted(records, key=lambda r: (r["pairs_count"], r["length"]), reverse=True)[:10],
        "dir": in_dir,
    }
    return records, stats

def _print_report(records, stats):
    print(f"Parsed {stats['count']} sequences from {stats['dir']}")
    print(f"Length    : min={stats['length_min']}, max={stats['length_max']}, mean={stats['length_mean']:.2f}")
    print(f"Pairs     : min={stats['pairs_min']},  max={stats['pairs_max']},  mean={stats['pairs_mean']:.2f}")
    print(f"% paired  : min={stats['paired_frac_min']*100:.1f}%, max={stats['paired_frac_max']*100:.1f}%, mean={stats['paired_frac_mean']*100:.1f}%")
    print(f"% N letters: min={stats['n_frac_min']*100:.1f}%, max={stats['n_frac_max']*100:.1f}%, mean={stats['n_frac_mean']*100:.1f}%")

    print("\nTop 10 by pairs_count:")
    for r in stats["top10_by_pairs"]:
        print(f"  {r['pdb_id']}:{r['chain']}  pairs={r['pairs_count']}  len={r['length']}  paired%={r['frac_paired']*100:.1f}%")

    # Optional: print a concise line for each record
    print("\nPer-sequence (pdb:chain len pairs stems N%):")
    for r in records:
        nperc = r["n_frac"]*100
        print(f"  {r['pdb_id']}:{r['chain']}  {r['length']:>4}  {r['pairs_count']:>4}  {r['stems_approx']:>3}  {nperc:5.1f}%")

if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else None
    RESULTS, STATS = load_summaries(target_dir)   # <- variables you can reuse if importing

    for r in RESULTS:
        print(r["file"])

    for k, v in STATS.items():
        print(f"{k}: {v}")
