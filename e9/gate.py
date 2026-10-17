#!/usr/bin/env python3
"""EMBED-CORE-0 E9 gate — run the PRE-REGISTERED test on the trained reservoir.

Same 200 heading->chunk queries, seed 0, same 6,194 vault candidates, same FTS store, same
RRF as e4/e5/e6/e8. Targets were excluded from the E9 fit, so this is honest.

Bar (PREREG.md, written before any core result):
    PASS = hybrid FTS+core recall@5 >= 0.775 AND core throughput >= 100x nomic

Reports core-only, FTS-only and hybrid side by side, plus the nomic-only reference, so a
number can never be read without its comparator. The FTS-only arm is the calibration
check: it must reproduce 0.735 or the harness is wrong and the run is void.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_reservoir import Reservoir, encode, load_all  # noqa: E402
sys.path.insert(0, "<your-data>/projects/omnid")
import shardd  # noqa: E402

OUT = Path(__file__).resolve().parent
BAR = 0.775


def main():
    t0 = time.time()
    tok, special, data, ids_all, TP, nomic, cand = load_all()
    net = Reservoir(TP)
    sd = torch.load(OUT / "e9_reservoir.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(sd)
    net.eval()

    doc_ids = sorted(nomic)
    Y = np.stack([nomic[d] for d in doc_ids])
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
    targets = {ids_all[k] for k, _ in cand}

    Xd = encode(tok, special, [data[d]["__data__"]["text"] for d in doc_ids])
    Xq = encode(tok, special, [q for _, q in cand])
    with torch.no_grad():
        Dd = net(Xd).numpy()
        Dq = net(Xq).numpy()

    # throughput on the real workload — the other half of the gate
    t = time.perf_counter()
    with torch.no_grad():
        for i in range(0, 2000, 200):
            net(encode(tok, special, [data[doc_ids[(i + j) % len(doc_ids)]]["__data__"]["text"]
                                      for j in range(200)]))
    docs_per_s = 2000 / max(time.perf_counter() - t, 1e-9)

    srv = shardd.Server(); st = srv.st
    pos = {d: j for j, d in enumerate(doc_ids)}
    arms = {"core_only": 0, "fts_only": 0, "hybrid_fts_core": 0, "nomic_only": 0}
    t0s = time.time()
    for i, (k, _q) in enumerate(cand):
        tgt = ids_all[k]
        cl = [doc_ids[j] for j in np.argsort(-(Dq[i] @ Dd.T))[:50]]
        nn = [doc_ids[j] for j in np.argsort(-(Y @ Y[pos[tgt]]))[:50]]
        fts = st.fts_vault(_q, 50)
        arms["core_only"] += tgt in cl[:5]
        arms["nomic_only"] += tgt in nn[:5]
        arms["fts_only"] += tgt in [h[1]["id"] for h in fts[:5]]
        arms["hybrid_fts_core"] += tgt in [p["id"] for p in shardd.rrf(
            [("vault_fts", fts), ("vault_vec", [(f"vault:{c}", {"id": c}) for c in cl])], 5)]

    r = {a: round(v / len(cand), 3) for a, v in arms.items()}
    r["core_docs_per_s"] = round(docs_per_s, 1)
    r["speedup_vs_nomic"] = round(docs_per_s / 1.16, 1)
    r["leak_mean"] = round(float(net.leak.mean()), 4)
    r["bar_recall"] = BAR
    r["bar_speed"] = 100
    r["calibration_fts_only_expected"] = 0.735
    r["gate_seconds"] = round(time.time() - t0s, 1)
    r["PASS"] = (r["hybrid_fts_core"] >= BAR) and (r["speedup_vs_nomic"] >= 100)
    r["calibration_ok"] = abs(r["fts_only"] - 0.735) < 0.02
    r["verdict"] = ("PASS" if r["PASS"] else "FAIL") + ("" if r["calibration_ok"]
                                                          else "  [CALIBRATION FAILED - VOID]")
    print(json.dumps(r, indent=1), flush=True)
    json.dump(r, open(OUT / "e9_gate.json", "w"), indent=1)


if __name__ == "__main__":
    main()
