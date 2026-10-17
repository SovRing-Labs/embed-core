"""EMBED-CORE-0 E1→E4: build the static core from the distilled vocabulary and run the PRE-REGISTERED gate (PREREG.md) on the HUB-3 set.
Core = nomic's own per-token outputs (e1/vocab, cos 1.0000 vs omnid) · SIF frequency weighting from a local corpus (Arora et al. 2017) ·
Text vectors sum the weighted token rows and are L2-normalised · optional PCA 256 · ternary variant = per-dimension deadband (|v| < 0.5 σ → 0).
Same 200 queries, same FTS5 store, same RRF (k = 60, top-50 lists) as shardd. Reproduces the nomic arms first (sanity: 0.685 / 0.790)."""
import glob, json, os, random, re, sys, time
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent; EC = HERE.parent
sys.path.insert(0, "<your-data>/projects/omnid"); import shardd
TOKJ = "<your-data>/bitnet/models/nomic-embed-text-v1.5/tokenizer.json"

parts = sorted(glob.glob(str(EC / "e1/vocab/part.*.emb")))
T = np.concatenate([np.array(json.load(open(p)), dtype=np.float32) for p in parts])
V = [l.rstrip("\n") for l in open("<your-data>/bitnet/models/nomic-embed-text-v1.5/vocab.txt")]
assert T.shape == (len(V), 768), (T.shape, len(V))
tok = Tokenizer.from_file(TOKJ); SPECIAL = {tok.token_to_id(x) for x in ("[CLS]", "[SEP]", "[PAD]") if tok.token_to_id(x) is not None}

IX = Path("<your-data>/vault-indexer/index"); data = json.load(open(IX / "docstore.json"))["docstore/data"]
ids = json.load(open(IX / "nomic_local" / "ids.json"))
cand = []
for k, i in enumerate(ids):                              # identical selection to hub3_eval.py / eval_nomic_vs_mistral.py
    m = re.search(r"^#+\s+(.{8,120})$", data[i]["__data__"]["text"], re.M)
    if m: cand.append((k, m.group(1).strip()))
random.Random(0).shuffle(cand); cand = cand[:200]

srv = shardd.Server(); st = srv.st
doc_ids = list(st.vault_ids); texts = [data[i]["__data__"]["text"] if i in data else st.vault_text.get(i, "") for i in doc_ids]

def ids_of(s): return [t for t in tok.encode(s, add_special_tokens=False).ids if t not in SPECIAL]
t0 = time.perf_counter(); doc_tok = [ids_of(s) for s in texts]; t_tok = time.perf_counter() - t0
cnt = np.bincount(np.concatenate([np.array(d, dtype=np.int64) for d in doc_tok if d]), minlength=len(V)).astype(np.float64)
p = cnt / cnt.sum(); A = 1e-3; W = (A / (A + p)).astype(np.float32)          # SIF weights from our corpus
def emb(tids, M):
    if not tids: return np.zeros(M.shape[1], np.float32)
    v = (W[tids, None] * M[tids]).sum(0); n = np.linalg.norm(v); return v / n if n else v
mean = T.mean(0); U, S, Vt = np.linalg.svd(T - mean, full_matrices=False); P = Vt[:256].T              # PCA 256 of the token table
TP = (T - mean) @ P
t1 = time.perf_counter(); D_f = np.stack([emb(d, T) for d in doc_tok]); t_emb = time.perf_counter() - t1
D_p = np.stack([emb(d, TP) for d in doc_tok])
sig = D_p.std(0); D_t = np.where(np.abs(D_p) < 0.5 * sig, 0, np.sign(D_p)).astype(np.int8)
docs_per_s = len(doc_tok) / (t_tok + t_emb)

def vec_list(D, qv, n=50, tern=False):
    if tern: qv = np.where(np.abs(qv) < 0.5 * sig, 0, np.sign(qv)).astype(np.int8); s = (D.astype(np.int32) @ qv.astype(np.int32)).astype(np.float64)
    else: s = D @ (qv / (np.linalg.norm(qv) or 1))
    top = np.argsort(-s)[:n]; return [(f"vault:{doc_ids[i]}", {"id": doc_ids[i]}) for i in top]

arms = {"nomic_only": 0, "hybrid_fts_nomic": 0, "core_only": 0, "hybrid_fts_core": 0, "core256_only": 0, "core_tern_only": 0, "hybrid_fts_core_tern": 0, "fts_only": 0}
lat = []
for k, q in cand:
    target = ids[k]
    fts = st.fts_vault(q, 50)
    nv = st.vec_vault(srv.emb.query(q), 50)
    tq = time.perf_counter(); qt = ids_of(q); qf = emb(qt, T); qp = emb(qt, TP); cf = vec_list(D_f, qf); lat.append(time.perf_counter() - tq)
    ct = vec_list(D_t, qp, tern=True); cp = vec_list(D_p, qp)
    hit = lambda L: target in [h[1]["id"] for h in L[:5]]
    top5 = lambda lists: [p["id"] for p in shardd.rrf(lists, 5)]
    arms["fts_only"] += hit(fts); arms["nomic_only"] += hit(nv); arms["core_only"] += hit(cf); arms["core256_only"] += hit(cp); arms["core_tern_only"] += hit(ct)
    arms["hybrid_fts_nomic"] += target in top5([("vault_fts", fts), ("vault_vec", nv)])
    arms["hybrid_fts_core"] += target in top5([("vault_fts", fts), ("vault_vec", cf)])
    arms["hybrid_fts_core_tern"] += target in top5([("vault_fts", fts), ("vault_vec", ct)])
n = len(cand); r = {a: round(v / n, 3) for a, v in arms.items()}
nomic_docs_per_s = 1.16                                   # control-center/reflex/embed-bench-result.json (nomic f16 on omnid, this laptop)
r.update({"n": n, "docs": len(doc_tok), "core_docs_per_s": round(docs_per_s, 1), "speedup_vs_nomic": round(docs_per_s / nomic_docs_per_s),
          "query_p50_ms_core": round(1000 * float(np.median(lat)), 2), "table_mb_f32": round(T.nbytes / 2**20, 1), "table_mb_pca256_trit2bit": round(len(V) * 256 * 2 / 8 / 2**20, 2),
          "PASS": r["hybrid_fts_core"] >= 0.775 and docs_per_s / nomic_docs_per_s >= 100, "bar": "hybrid_fts_core >= 0.775 AND >= 100x nomic docs/s"})
print(json.dumps(r, indent=1)); json.dump(r, open(HERE / "result.json", "w"), indent=1)
np.save(EC / "e1/token_table_f32.npy", T); np.save(EC / "e1/sif_weights.npy", W); np.save(EC / "e1/pca256.npy", np.concatenate([mean[None], P.T]))
