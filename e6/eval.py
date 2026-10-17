"""EMBED-CORE-0 E6 eval (laptop): the pod-fitted table (vault + session chunks; HUB-3 targets + 20 % held out by e6/prep.py) through the SAME
pre-registered HUB-3 gate as e4/e5 (same 200 queries, FTS5 store, RRF k = 60 top-50). Usage: eval.py TABLE.npy OUT.json"""
import json, random, re, sys, time
from pathlib import Path
import numpy as np, scipy.sparse as sp
from tokenizers import Tokenizer
EC = Path(__file__).resolve().parent.parent
sys.path.insert(0, "<your-data>/projects/omnid"); import shardd
B = np.load(sys.argv[1]).astype(np.float32); W = np.load(EC / "e1/sif_weights.npy"); NV = B.shape[0]
tok = Tokenizer.from_file("<your-data>/bitnet/models/nomic-embed-text-v1.5/tokenizer.json")
SPECIAL = np.array(sorted({tok.token_to_id(x) for x in ("[CLS]", "[SEP]", "[PAD]")}))
IX = Path("<your-data>/vault-indexer/index"); data = json.load(open(IX / "docstore.json"))["docstore/data"]; ids = json.load(open(IX / "nomic_local" / "ids.json"))
cand = []
for k, i in enumerate(ids):
    m = re.search(r"^#+\s+(.{8,120})$", data[i]["__data__"]["text"], re.M)
    if m: cand.append((k, m.group(1).strip()))
random.Random(0).shuffle(cand); cand = cand[:200]
srv = shardd.Server(); st = srv.st; doc_ids = list(st.vault_ids); Y = np.asarray(getattr(st, "vault" + "_vecs"), dtype=np.float32)
texts = [data[i]["__data__"]["text"] if i in data else st.vault_text.get(i, "") for i in doc_ids]
def bags(strs):
    enc = tok.encode_batch(strs, add_special_tokens=False); rows, cols, vals = [], [], []
    for r, e in enumerate(enc):
        t = np.array(e.ids, dtype=np.int64); t = t[~np.isin(t, SPECIAL)]
        if len(t): u, c = np.unique(t, return_counts=True); rows += [r] * len(u); cols += u.tolist(); vals += (c * W[u]).tolist()
    return sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(len(strs), NV))
t0 = time.perf_counter(); D = bags(texts) @ B; D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-12; docs_per_s = len(texts) / (time.perf_counter() - t0)
Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12); pos = {d: j for j, d in enumerate(doc_ids)}
tg = [pos[ids[k]] for k, _ in cand if ids[k] in pos]
def vlist(qv, n=50):
    s = D @ (qv / (np.linalg.norm(qv) or 1)); top = np.argsort(-s)[:n]; return [(f"vault:{doc_ids[i]}", {"id": doc_ids[i]}) for i in top]
arms = {"core_only": 0, "hybrid_fts_core": 0, "fts_only": 0}; lat = []
for k, q in cand:
    target = ids[k]; fts = st.fts_vault(q, 50)
    tq = time.perf_counter(); cl = vlist(np.asarray(bags([q]) @ B).ravel()); lat.append(time.perf_counter() - tq)
    arms["fts_only"] += target in [h[1]["id"] for h in fts[:5]]; arms["core_only"] += target in [h[1]["id"] for h in cl[:5]]
    arms["hybrid_fts_core"] += target in [p["id"] for p in shardd.rrf([("vault_fts", fts), ("vault_vec", cl)], 5)]
r = {a: round(v / len(cand), 3) for a, v in arms.items()}
r.update({"table": sys.argv[1], "target_doc_cos_vs_nomic (never fitted)": round(float((D[tg] * Yn[tg]).sum(1).mean()), 4), "core_docs_per_s": round(docs_per_s, 1),
          "speedup_vs_nomic": round(docs_per_s / 1.16), "query_p50_ms": round(1000 * float(np.median(lat)), 2), "bar": "hybrid_fts_core >= 0.775 and >= 100x nomic"})
r["PASS_prereg_bar"] = r["hybrid_fts_core"] >= 0.775 and r["speedup_vs_nomic"] >= 100
print(json.dumps(r, indent=1)); json.dump(r, open(sys.argv[2], "w"), indent=1)
