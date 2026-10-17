"""EMBED-CORE-0 E5 probe (after E4 FAILED as registered): learn the token table on OUR corpus — Model2Vec's "tokenlearn" idea re-made here
(clean room, from the method description): B (vocab × 768) starts at the distilled table T; minimise Σ_d (1 − cos(x_d · B, nomic_d)) + λ‖B − T‖²
over the 6,194 vault documents (x_d = SIF-weighted bag of WordPiece tokens; nomic_d = the vault's existing nomic vectors — no new nomic runs).
The 200 HUB-3 queries are NEVER used in fitting. Same gate, same FTS, same RRF as e4/eval.py. Batch tokenizer (Rust, parallel) for the speed bar."""
import json, random, re, sys, time
from pathlib import Path
import numpy as np, scipy.sparse as sp, torch
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent; EC = HERE.parent
sys.path.insert(0, "<your-data>/projects/omnid"); import shardd
torch.set_num_threads(6)
T = np.load(EC / "e1/token_table_f32.npy"); W = np.load(EC / "e1/sif_weights.npy"); NV = T.shape[0]
tok = Tokenizer.from_file("<your-data>/bitnet/models/nomic-embed-text-v1.5/tokenizer.json")
SPECIAL = np.array(sorted({tok.token_to_id(x) for x in ("[CLS]", "[SEP]", "[PAD]")}))

IX = Path("<your-data>/vault-indexer/index"); data = json.load(open(IX / "docstore.json"))["docstore/data"]
ids = json.load(open(IX / "nomic_local" / "ids.json"))
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
t0 = time.perf_counter(); X = bags(texts); t_bag = time.perf_counter() - t0

# HOLD-OUT (argv[3] = "holdout"): the 200 target docs + 20 % of the rest never enter the fit; their vectors come from unseen text
hold = np.zeros(len(doc_ids), bool)
if len(sys.argv) > 3 and sys.argv[3] == "holdout":
    pos = {d: j for j, d in enumerate(doc_ids)}; tg = {pos[ids[k]] for k, _ in cand if ids[k] in pos}
    rr = np.random.default_rng(1); hold[list(tg)] = True; hold |= rr.random(len(doc_ids)) < 0.2
fit = ~hold; Xf = X[fit]
Xt = torch.sparse_csr_tensor(torch.tensor(Xf.indptr), torch.tensor(Xf.indices), torch.tensor(Xf.data), size=Xf.shape)
B = torch.tensor(T.copy(), requires_grad=True); T0 = torch.tensor(T); Yt = torch.nn.functional.normalize(torch.tensor(Y[fit]), dim=1)
opt = torch.optim.Adam([B], lr=2e-3); lam = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-4; steps = int(sys.argv[2]) if len(sys.argv) > 2 else 150
t1 = time.perf_counter()
for s in range(steps):
    opt.zero_grad(); P = torch.nn.functional.normalize(Xt @ B, dim=1)
    loss = (1 - (P * Yt).sum(1)).mean() + lam * ((B - T0) ** 2).sum() / NV
    loss.backward(); opt.step()
    if s % 25 == 0 or s == steps - 1: print(f"step {s} loss {loss.item():.4f} mean cos {(P * Yt).sum(1).mean().item():.4f}", flush=True)
t_fit = time.perf_counter() - t1
Bn = B.detach().numpy()
t2 = time.perf_counter(); D = X @ Bn; D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-12; t_emb = time.perf_counter() - t2   # the core embeds docs: bag × table
docs_per_s = len(texts) / (t_bag + t_emb)
Yn = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-12)
heldout_cos = float((D[hold] * Yn[hold]).sum(1).mean()) if hold.any() else None; fit_cos = float((D[fit] * Yn[fit]).sum(1).mean())

def vlist(qv, n=50):
    s = D @ (qv / (np.linalg.norm(qv) or 1)); top = np.argsort(-s)[:n]; return [(f"vault:{doc_ids[i]}", {"id": doc_ids[i]}) for i in top]
arms = {"core_learned_only": 0, "hybrid_fts_core_learned": 0, "fts_only": 0}; lat = []
for k, q in cand:
    target = ids[k]; fts = st.fts_vault(q, 50)
    tq = time.perf_counter(); qv = np.asarray(bags([q]) @ Bn).ravel(); cl = vlist(qv); lat.append(time.perf_counter() - tq)
    hit = lambda L: target in [h[1]["id"] for h in L[:5]]
    arms["fts_only"] += hit(fts); arms["core_learned_only"] += hit(cl)
    arms["hybrid_fts_core_learned"] += target in [p["id"] for p in shardd.rrf([("vault_fts", fts), ("vault_vec", cl)], 5)]
n = len(cand); r = {a: round(v / n, 3) for a, v in arms.items()}
r.update({"holdout_docs": int(hold.sum()), "heldout_doc_cos_vs_nomic": heldout_cos and round(heldout_cos, 4), "fit_doc_cos": round(fit_cos, 4), "lambda": lam, "steps": steps, "fit_secs": round(t_fit, 1), "docs": len(texts), "core_docs_per_s": round(docs_per_s, 1),
          "speedup_vs_nomic": round(docs_per_s / 1.16), "query_p50_ms": round(1000 * float(np.median(lat)), 2),
          "PASS_prereg_bar": r["hybrid_fts_core_learned"] >= 0.775 and docs_per_s / 1.16 >= 100})
print(json.dumps(r, indent=1)); json.dump(r, open(HERE / f"result-l{lam}-s{steps}{'-holdout' if hold.any() else ''}.json", "w"), indent=1)
np.save(EC / f"e5/token_table_learned_l{lam}_s{steps}.npy", Bn.astype(np.float16))
