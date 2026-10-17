#!/usr/bin/env python3
"""EMBED-CURRIC-0 per-level held-out gate.

  gate.py eval    [--embedder s1|file.py:Class] [--out F]   recall@10 on every level's HELD-OUT groups (never trained on) -> JSON
  gate.py prek    [--embedder ..] [--max 200] [--out F]     pre-K score card: per-ability accuracy (cos q,pos > cos q,neg) + mean margin
  gate.py compare --base B.json --cand C.json --stage K     stage K passes iff level K rises AND no earlier level drops > SIGMA
  gate.py leak                                              checks no train row shares any text with a held-out row (exit 1 if it does)
  gate.py selftest                                          s1 vs s1 (no rise -> FAIL) and s1 vs a weakened s1 (drop trips) on a small slice

recall@10: for every held-out row the query is ranked against the pool of all held-out positives + hard negatives of that level;
a hit = the row's positive within the top 10. SIGMA = 0.017, the P2 spread of s1 (projects/core-models/results/heldout-1013/
heldout-s1.json, 5 seeds: 0.785 +- 0.017). Rising = candidate recall strictly above base. Levels with fewer than MIN_HELDOUT
held-out rows are reported with low_n and cannot pass or fail a stage on their own.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
from lib import LEVELS_DIR, PREK_DIR, LEVEL_IDS, SIGMA  # noqa: E402

MIN_HELDOUT = 30
K = 10


def heldout(level):
    return [x for x in lib.read_jsonl(LEVELS_DIR / f"{level}.jsonl") if x["split"] == "heldout"]


def recall_at_k(emb, rows, k=K):
    import numpy as np
    pool = []
    seen = {}
    for x in rows:
        for t in [x["pos"], *x["neg"]]:
            if t not in seen:
                seen[t] = len(pool)
                pool.append(t)
    D = emb.embed_texts(pool)
    Q = emb.embed_texts([x["q"] for x in rows])
    S = Q @ D.T
    hits = 0
    for i, x in enumerate(rows):
        s = S[i]
        hits += int((s > s[seen[x["pos"]]]).sum() < k)
    return hits / len(rows), len(pool)


def cmd_eval(a):
    emb = lib.load_embedder(a.embedder)
    res = {"embedder": a.embedder, "metric": f"recall@{K} on held-out groups", "sigma": SIGMA, "levels": {}}
    for lv in LEVEL_IDS:
        if not (LEVELS_DIR / f"{lv}.jsonl").exists():
            continue
        rows = heldout(lv)
        if not rows:
            res["levels"][lv] = {"n": 0, "low_n": True, "recall": None}
            continue
        r, npool = recall_at_k(emb, rows)
        res["levels"][lv] = {"n": len(rows), "pool": npool, "recall": round(r, 4), "low_n": len(rows) < MIN_HELDOUT}
    out = json.dumps(res, indent=1)
    print(out)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(out + "\n")


def cmd_prek(a):
    import numpy as np
    emb = lib.load_embedder(a.embedder)
    P = lib.read_jsonl(PREK_DIR / "probes.jsonl")
    by = {}
    for x in P:
        by.setdefault(x["ability"], []).append(x)
    per = max(1, a.max // len(by))
    rows = [x for ab in sorted(by) for x in by[ab][:per]]
    assert len(rows) <= a.max
    Q = emb.embed_texts([x["q"] for x in rows])
    Pp = emb.embed_texts([x["pos"] for x in rows])
    Nn = emb.embed_texts([x["neg"][0] for x in rows])
    cp, cn = (Q * Pp).sum(1), (Q * Nn).sum(1)
    card = {}
    for ab in sorted(by):
        idx = [i for i, x in enumerate(rows) if x["ability"] == ab]
        card[ab] = {"n": len(idx), "acc": round(float((cp[idx] > cn[idx]).mean()), 3), "margin": round(float((cp[idx] - cn[idx]).mean()), 3),
                    "cos_pos": round(float(cp[idx].mean()), 3)}
    res = {"embedder": a.embedder, "rows": len(rows), "metric": "acc = cos(q,pos) > cos(q,hard neg); margin = mean cos gap", "card": card}
    out = json.dumps(res, indent=1)
    print(out)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(out + "\n")


def judge(base, cand, stage):
    tgt = LEVEL_IDS[stage]
    b, c = base["levels"], cand["levels"]
    v = {"stage": stage, "target": tgt, "sigma": SIGMA, "reasons": []}
    def rec(d, lv):
        x = d.get(lv)
        return None if x is None or x.get("recall") is None else x
    bt, ct = rec(b, tgt), rec(c, tgt)
    if bt is None or ct is None:
        v["reasons"].append(f"{tgt}: no held-out recall in base or candidate")
        v["target_rises"] = False
    elif bt.get("low_n") or ct.get("low_n"):
        v["reasons"].append(f"{tgt}: held-out n={ct['n']} < {MIN_HELDOUT} (low_n) - cannot certify a rise")
        v["target_rises"] = False
    else:
        v["target_rises"] = ct["recall"] > bt["recall"]
        v[tgt] = {"base": bt["recall"], "cand": ct["recall"], "delta": round(ct["recall"] - bt["recall"], 4)}
        if not v["target_rises"]:
            v["reasons"].append(f"{tgt} did not rise ({bt['recall']} -> {ct['recall']})")
    v["earlier"] = {}
    for lv in LEVEL_IDS[:stage]:
        be, ce = rec(b, lv), rec(c, lv)
        if be is None or ce is None or be.get("low_n") or ce.get("low_n"):
            v["earlier"][lv] = {"status": "not-scored (low_n or missing)"}
            continue
        drop = round(be["recall"] - ce["recall"], 4)
        bad = drop > SIGMA
        v["earlier"][lv] = {"base": be["recall"], "cand": ce["recall"], "drop": drop, "violates": bad}
        if bad:
            v["reasons"].append(f"{lv} dropped {drop} > sigma {SIGMA}")
    v["pass"] = bool(v["target_rises"]) and not any(e.get("violates") for e in v["earlier"].values())
    return v


def cmd_compare(a):
    v = judge(json.load(open(a.base)), json.load(open(a.cand)), a.stage)
    print(json.dumps(v, indent=1))
    sys.exit(0 if v["pass"] else 1)


def cmd_leak(a):
    held = set()
    for lv in LEVEL_IDS:
        for x in heldout(lv):
            held.update(lib.norm(t) for t in [x["q"], x["pos"], *x["neg"]])
    bad = {}
    for lv in LEVEL_IDS:
        n = sum(1 for x in lib.read_jsonl(LEVELS_DIR / f"{lv}.jsonl")
                if x["split"] == "train" and ({lib.norm(t) for t in [x["q"], x["pos"], *x["neg"]]} & held))
        if n:
            bad[lv] = n
    print(json.dumps({"train_rows_sharing_text_with_a_held_out_row": bad or 0}))
    sys.exit(1 if bad else 0)


class Weakened:
    """s1 reading only the first 6 words of every text - a deliberate regression for the selftest."""
    def __init__(self, inner):
        self.inner = inner

    def embed_texts(self, texts):
        return self.inner.embed_texts([" ".join(t.split()[:6]) for t in texts])


def cmd_selftest(a):
    s1 = lib.S1()
    mk = lambda e: {"levels": {lv: {"n": len(heldout(lv)), "recall": round(recall_at_k(e, heldout(lv))[0], 4),
                                    "low_n": len(heldout(lv)) < MIN_HELDOUT} for lv in LEVEL_IDS if heldout(lv)}}
    base, weak = mk(s1), mk(Weakened(s1))
    out = {"base": base, "weak": weak,
           "s1_vs_s1_stage1": judge(base, base, 1), "s1_vs_weak_stage1": judge(base, weak, 1), "s1_vs_weak_stage4": judge(base, weak, 4)}
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for n, f in (("eval", cmd_eval), ("prek", cmd_prek)):
        s = sp.add_parser(n); s.add_argument("--embedder", default="s1"); s.add_argument("--out"); s.set_defaults(f=f)
        if n == "prek":
            s.add_argument("--max", type=int, default=200)
    s = sp.add_parser("compare"); s.add_argument("--base", required=True); s.add_argument("--cand", required=True)
    s.add_argument("--stage", type=int, required=True); s.set_defaults(f=cmd_compare)
    sp.add_parser("leak").set_defaults(f=cmd_leak)
    sp.add_parser("selftest").set_defaults(f=cmd_selftest)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
