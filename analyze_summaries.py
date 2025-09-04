#!/usr/bin/env python3
"""
Analyze per-sequence *_summary.txt files inside a 'per_sequence_summaries' folder.

Extracts for each sequence:
- pdb_id, chain, length
- sequence
- dbn string
- pairs_count
- paired_indices list

Outputs:
- Console stats
- summary_table.csv in the same folder
"""

import os
import re
import sys
import csv
from statistics import mean

HEADER_RE   = re.compile(r">\s*([A-Za-z0-9]{4})\s*\|\s*chain:\s*([^\s|]+)\s*\|\s*length:\s*(\d+)")
PAIR_TUP_RE = re.compile(r"\((\d+),\s*(\d+)\)")

def parse_summary_file(path):
    """Parse one *_summary.txt -> dict with all fields we care about."""
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

    # sanity check
    if pairs_count == 0 and paired_indices:
        pairs_count = len(paired_indices)

    paired_positions = set()
    for i,j in paired_indices:
        paired_positions.add(i); paired_positions.add(j)
    frac_paired = len(paired_positions) / max(1, length)

    n_count  = sequence.upper().count("N")
    n_frac   = n_count / max(1, length)

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
    }

def count_stems_from_dbn(dbn):
    """Rough stem count: number of runs of '(' (non-pseudoknotted assumption)."""
    stems, prev = 0, ""
    for ch in dbn:
        if ch == "(" and prev != "(":
            stems += 1
        prev = ch
    return stems

def main():
    # Default to ./per_sequence_summaries if not provided
    in_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(".", "per_sequence_summaries")
    if not os.path.isdir(in_dir):
        print(f"Directory not found: {in_dir}", file=sys.stderr)
        sys.exit(1)

    files = sorted(f for f in (os.path.join(in_dir, x) for x in os.listdir(in_dir))
                   if os.path.isfile(f) and f.endswith("_summary.txt"))
    if not files:
        print(f"No *_summary.txt files in: {in_dir}", file=sys.stderr)
        sys.exit(1)

    records = []
    for p in files:
        try:
            rec = parse_summary_file(p)
            rec["stems_approx"] = count_stems_from_dbn(rec["dbn"])
            records.append(rec)
        except Exception as e:
            print(f"[WARN] Skipping {os.path.basename(p)} -> {e}", file=sys.stderr)

    if not records:
        print("No valid summaries parsed.", file=sys.stderr)
        sys.exit(1)

    lengths = [r["length"] for r in records]
    pairs   = [r["pairs_count"] for r in records]
    fracs   = [r["frac_paired"] for r in records]
    n_fracs = [r["n_frac"] for r in records]

    print(f"Parsed {len(records)} sequences from {in_dir}")
    print(f"Length    : min={min(lengths)}, max={max(lengths)}, mean={mean(lengths):.2f}")
    print(f"Pairs     : min={min(pairs)},  max={max(pairs)},  mean={mean(pairs):.2f}")
    print(f"% paired  : min={min(fracs)*100:.1f}%, max={max(fracs)*100:.1f}%, mean={mean(fracs)*100:.1f}%")
    print(f"% N letters: min={min(n_fracs)*100:.1f}%, max={max(n_fracs)*100:.1f}%, mean={mean(n_fracs)*100:.1f}%")

    # Write CSV next to the summaries dir
    out_csv = os.path.join(in_dir, "summary_table.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "pdb_id","chain","length","pairs_count","frac_paired",
            "n_count","n_frac","stems_approx","paired_indices","sequence","dbn","file"
        ])
        for r in records:
            w.writerow([
                r["pdb_id"], r["chain"], r["length"], r["pairs_count"],
                f"{r['frac_paired']:.4f}", r["n_count"], f"{r['n_frac']:.4f}",
                r["stems_approx"],
                " ".join(f"({i},{j})" for (i,j) in r["paired_indices"]),
                r["sequence"], r["dbn"], r["file"]
            ])

    print(f"\nWrote CSV: {out_csv}")

if __name__ == "__main__":
    main()
