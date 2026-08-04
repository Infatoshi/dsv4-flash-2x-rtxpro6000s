"""Assemble final sweep tables from opt/sweep_results.jsonl.

usage: report_sweep.py <sweep_results.jsonl>
Prints markdown: grid tables (per server tag) and category tables.
Keeps only the LAST row per (tag, cell) so re-runs supersede crashes.
"""

import json
import sys

rows = {}
for line in open(sys.argv[1]):
    r = json.loads(line)
    if "cell" not in r:
        continue
    rows[(r["tag"], r["cell"])] = r

tags = sorted({t for t, _ in rows})
grid_cells = {k: v for k, v in rows.items() if v.get("input_len")}
cat_cells = {k: v for k, v in rows.items() if v.get("category")}

for tag in tags:
    sub = {k[1]: v for k, v in grid_cells.items() if k[0] == tag}
    if not sub:
        continue
    lens = sorted({v["input_len"] for v in sub.values()})
    concs = sorted({v["conc"] for v in sub.values()})
    print(f"\n### grid: {tag}\n")
    hdr = "| input len | " + " | ".join(f"c={c}" for c in concs) + " |"
    print(hdr)
    print("|---" * (len(concs) + 1) + "|")
    for L in lens:
        cells = []
        for c in concs:
            v = sub.get(f"in{L}_c{c}")
            if not v:
                cells.append("-")
                continue
            s = f"{v['aggregate_tps']:.0f} / {v['per_stream_tps']:.0f}"
            if "accept_rate" in v:
                s += f" ({v['accept_rate']*100:.0f}%)"
            cells.append(s)
        print(f"| {L} | " + " | ".join(cells) + " |")
    print("\n(aggregate tok/s / per-stream tok/s, (draft accept rate))")

if cat_cells:
    print("\n### categories\n")
    print("| category | server | c1 per-stream | c1 min-max | c4 agg / per-stream | accept c1 |")
    print("|---|---|---|---|---|---|")
    cats = ["chat", "math", "code", "baseline"]
    for cat in cats:
        for tag in tags:
            v1 = cat_cells.get((tag, f"cat_{cat}_c1"))
            v4 = cat_cells.get((tag, f"cat_{cat}_c4"))
            if not v1 and not v4:
                continue
            mm = f"{v1['per_stream_tps_min']:.0f}-{v1['per_stream_tps_max']:.0f}" if v1 else "-"
            c1 = f"{v1['per_stream_tps']:.0f}" if v1 else "-"
            c4 = f"{v4['aggregate_tps']:.0f} / {v4['per_stream_tps']:.0f}" if v4 else "-"
            ar = f"{v1['accept_rate']*100:.0f}%" if v1 and "accept_rate" in v1 else "-"
            print(f"| {cat} | {tag} | {c1} | {mm} | {c4} | {ar} |")
