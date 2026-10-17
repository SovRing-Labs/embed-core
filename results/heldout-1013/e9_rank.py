"""HELDOUT-1013 stage 1 for embed-core e9: ranking cache (top-50 doc ids per query) for core / FTS / nomic.
Same 200 queries, candidates and FTS as e9/gate.py. Checkpoint e9_reservoir.pt is read from the main tree (untracked there).
nomic arms: 'nomic' = ORACLE (query = target doc's own nomic vector, as gate.py's nomic_only; leaky upper bound);
            'nomic_q' = real nomic query embedding from omnid ('search_query:' prefix) against doc vectors, if omnid is up."""
import json, sys
from pathlib import Path
import numpy as np, torch
E9 = Path("./e9")
sys.path.insert(0, str(E9)); sys.path.insert(0, "<your-data>/projects/omnid")
from train_reservoir import Reservoir, encode, load_all
import shardd
OUT = Path(__file__).resolve().parent
tok, special, data, ids_all, TP, nomic, cand = load_all()
net = Reservoir(TP); net.load_state_dict(torch.load(E9 / "e9_reservoir.pt", map_location="cpu", weights_only=True)); net.eval()
doc_ids = sorted(nomic); Y = np.stack([nomic[d] for d in doc_ids]); Y /= np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12
pos = {d: j for j, d in enumerate(doc_ids)}
def fwd(texts, bs=256):  # chunked: one-shot encode of all docs OOM-killed under shared RAM
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            out.append(net(encode(tok, special, texts[i:i + bs])).numpy())
    return np.concatenate(out)
Dd = fwd([data[d]["__data__"]["text"] for d in doc_ids]); Dq = fwd([q for _, q in cand])
emb = shardd.Embedder(); Qn = []
for _, q in cand:
    v = emb.query(q); Qn.append(v)
real = all(v is not None for v in Qn)
if real:
    Qn = np.stack([np.asarray(v, np.float32) for v in Qn]); Qn /= np.linalg.norm(Qn, axis=1, keepdims=True) + 1e-12
srv = shardd.Server(); st = srv.st
rank = []
for i, (k, q) in enumerate(cand):
    tgt = ids_all[k]
    r = {"target": tgt, "query": q,
         "core": [doc_ids[j] for j in np.argsort(-(Dq[i] @ Dd.T))[:50]],
         "fts": [h[1]["id"] for h in st.fts_vault(q, 50)],
         "nomic": [doc_ids[j] for j in np.argsort(-(Y @ Y[pos[tgt]]))[:50]]}
    if real: r["nomic_q"] = [doc_ids[j] for j in np.argsort(-(Y @ Qn[i]))[:50]]
    rank.append(r)
json.dump({"model": "embed-core e9 trained reservoir (single checkpoint, e9_reservoir.pt)", "seed": "ckpt", "nomic_q_real": real, "rank": rank}, open(OUT / "rank-e9.json", "w"))
print("e9 ranked", len(rank), "nomic_q_real", real)
