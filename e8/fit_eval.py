#!/usr/bin/env python3
"""EMBED-CORE-0 E8 — fit and gate the order-sensitive core.

Method, stated so it can be attacked:
  * REAL nomic targets. Loaded from the vault index (the same rows E6 used), NOT a proxy
    and not old-Wo output. The `.emb` session vectors and the vault nomic vectors both
    exist on disk; only the 6,194 vault rows are needed here, so the fit stays small.
  * HELD OUT: the 200 HUB-3 target documents are removed from the fit set, exactly as
    e6/prep.py did, so recall on them is honest.
  * CLOSED FORM: Wo by ridge regression, lambda chosen on a validation split. No SGD.
  * GATE: the PRE-REGISTERED bar from PREREG.md — hybrid FTS+core recall@5 >= 0.775 AND
    core throughput >= 100x nomic. Reported with core-only and FTS-only beside it.

Memory: only the 6,194 vault documents are touched (~6194 x 5376 f32 = 133 MB), so the
earlyoom problem that killed e7 cannot recur here.
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "<your-data>/projects/omnid")
import shardd  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ordered_core import OrderedCore, TAILS  # noqa: E402

AI = Path(__file__).resolve().parents[1]
IX = Path(os.environ.get("EMBED_CORE_INDEX", str(AI / "data" / "index")))
OUT = Path(__file__).resolve().parent

data = json.load(open(IX / "docstore.json"))["docstore/data"]


def load_vault_nomic():
    """Real nomic vectors for the 6,194 vault docs, pulled out of the sharded index."""
    tc = IX / "ternary_core"
    mf = json.loads((tc / "MANIFEST.json").read_text())
    want = set(data.keys())
    got = {}
    for sh in mf["shards"]:
        nm = sh["name"]
        ids = json.loads((tc / f"ids-{nm.split('-')[-1]}.json").read_text())
        arr = np.load(tc / f"{nm}.npy", mmap_mode="r")
        for i, _id in enumerate(ids):
            if _id in want:
                got[_id] = np.asarray(arr[i], dtype=np.float32)
    return got


def main():
    t0 = time.time()
    core = OrderedCore(gamma=float(sys.argv[1]) if len(sys.argv) > 1 else 0.97)

    # ---- queries: SAME 200 heading->chunk, seed 0, as e4/e5/e6 -------------
    ids_all = json.load(open(IX / "nomic_local" / "ids.json"))
    cand = []
    for k, i in enumerate(ids_all):
        m = re.search(r"^#+\s+(.{8,120})$", data[i]["__data__"]["text"], re.M)
        if m:
            cand.append((k, m.group(1).strip()))
    random.Random(0).shuffle(cand)
    cand = cand[:200]
    queries = [q for _, q in cand]
    targets = {ids_all[k] for k, _ in cand}

    nomic = load_vault_nomic()
    doc_ids = sorted(nomic)
    Y = np.stack([nomic[d] for d in doc_ids])
    Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
    texts = [data[d]["__data__"]["text"] for d in doc_ids]
    print(f"[e8] {len(doc_ids)} vault docs, {len(queries)} queries, "
          f"{len(targets)} targets held out", flush=True)

    # ---- features -----------------------------------------------------------
    feats = core.ordered_features(texts)
    print(f"[e8] ordered features: {feats.shape} ({time.time()-t0:.0f}s)", flush=True)
    fmu = feats.mean(0, keepdims=True)
    fsd = feats.std(0, keepdims=True) + 1e-6
    Fz = np.concatenate([(feats - fmu) / fsd, np.ones((len(feats), 1), np.float32)], axis=1)

    tgt_rows = {d: j for j, d in enumerate(doc_ids) if d in targets}
    keep = np.array([d not in targets for d in doc_ids])
    Ff, Yf = Fz[keep], Y[keep]

    # ---- ridge, lambda on a validation split --------------------------------
    rng = np.random.default_rng(7)
    vm = rng.random(len(Ff)) < 0.2
    lams = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    best = None
    for lam in lams:
        Wo = _ridge(Ff[~vm], Yf[~vm], lam)
        c = float((core.readout(Fz[keep][vm], Wo) * Yf[vm]).sum(1).mean())
        if best is None or c > best[1]:
            best = (lam, c)
    lam, val_cos = best
    Wo = _ridge(Ff, Yf, lam)
    print(f"[e8] lambda={lam}  val cosine to nomic={val_cos:.4f}", flush=True)

    # ---- gate: the same 200 queries ----------------------------------------
    qraw = core.ordered_features(queries)
    qz = np.concatenate([(qraw - fmu) / fsd, np.ones((len(qraw), 1), np.float32)], axis=1)
    assert qz.shape[1] == Fz.shape[1], (qz.shape, Fz.shape)
    # DOCUMENT embeddings. These were never built: the ranking line below used the QUERY
    # matrix for both sides (`Dm[q_i] @ Dm.T`), scoring each query against the other 199
    # queries and then indexing doc_ids with query row numbers. That is why core_only came
    # out 0.000 while a direct measurement gave rank-median 2 / top5 0.575.
    pos = {d: j for j, d in enumerate(doc_ids)}
    Dd = core.readout(Fz, Wo)
    Dd = Dd / (np.linalg.norm(Dd, axis=1, keepdims=True) + 1e-12)
    D = core.readout(qz, Wo)
    Dm = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)

    srv = shardd.Server(); st = srv.st
    arms = {"core_only": 0, "fts_only": 0, "hybrid_fts_core": 0, "hybrid_fts_nomic": 0}
    tdocs = np.asarray(list(nomic.values()), dtype=np.float32)  # for the nomic arm
    for q_i, q in enumerate(queries):
        tgt = ids_all[cand[q_i][0]]
        cl = [doc_ids[i] for i in np.argsort(-(Dm[q_i] @ Dd.T))[:50]]
        fts = st.fts_vault(q, 50)
        arms["core_only"] += tgt in cl[:5]
        arms["fts_only"] += tgt in [h[1]["id"] for h in fts[:5]]
        arms["hybrid_fts_core"] += tgt in [p["id"] for p in
                                           shardd.rrf([("vault_fts", fts),
                                                       ("vault_vec", [(f"vault:{c}", {"id": c}) for c in cl])], 5)]
        nn = [doc_ids[i] for i in np.argsort(-(Y @ Y[pos[tgt]]))[:50]]
        arms["hybrid_fts_nomic"] += tgt in [p["id"] for p in
                                            shardd.rrf([("vault_fts", fts),
                                                        ("vault_vec", [(f"vault:{c}", {"id": c}) for c in nn])], 5)]

    r = {a: round(v / len(queries), 3) for a, v in arms.items()}
    r["val_cosine_to_nomic"] = round(val_cos, 4)
    r["nomic_only_reference"] = 0.685
    r["bar"] = 0.775
    r["gamma"] = core.gamma
    r["feature_dims"] = int(Fz.shape[1])
    r["verdict"] = "PASS" if r["hybrid_fts_core"] >= 0.775 else "FAIL"
    r["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(r, indent=1), flush=True)
    json.dump(r, open(OUT / "e8_result.json", "w"), indent=1)
    np.save(OUT / "Wo_ordered.npy", Wo)


def _ridge(X, Y, lam):
    A = X.T @ X + lam * np.eye(X.shape[1], dtype=X.dtype)
    return np.linalg.solve(A, X.T @ Y)


if __name__ == "__main__":
    main()
