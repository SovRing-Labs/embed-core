#!/usr/bin/env python3
"""TERNARY-IO-0 T1 — a ternary readout for embed-core: the stored trits ARE the model's output (2026-10-14, OPUS-CONDENSE).
Pre-registered (before the run):
  Data/queries/split/FTS/evaluator = T3-VEC-0 / HELDOUT-1013 (t3_measure.py loader, executed verbatim up to its arm loop).
  Features = the e9 checkpoint's own pre-readout features (Wo swapped for identity; rows L2-normalised as forward() returns).
  Arms at D = 320 (64 B) and 640 (128 B) trits, docs stored ternary, queries float (asymmetric), scores in float64:
    float_e9      e9's trained float readout (reference; HELDOUT-1013 measured core arm hybrid 0.75)
    posthoc       TernaryEncoder(768 -> D) on e9's float output (what the shard store does today)
    ridge_tern    ridge F -> Y_nomic @ A_D on the fit docs (HUB-3 targets excluded), docs = tern(F W) at s 0.33
    ste_tern      ridge_tern + straight-through fine-tuning of W: loss = 1 - cos(STE_tern(F_d W), Y_d A) - cos(F_d W, Y_d A)
  Gates: G1 (plan) held-out hybrid >= 0.78 at 64 B; G2 hybrid >= float_e9 - 0.01 at the same width.
Run: bin/tw run --id tio0-t1 -- python3 projects/embed-core/t1/ternary_readout.py"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
T50DIR = Path("<your-data>/projects/condense/t50")
__file__ = str(T50DIR / "t3_measure.py")
src = (T50DIR / "t3_measure.py").read_text()
exec(compile(src[:src.index("res = {")], __file__, "exec"))   # doc_ids, Y (nomic, unit), Q (nomic queries), cand, fts, targets, evaluate
E9 = Path("./e9")
sys.path.insert(0, str(E9)); sys.path.insert(0, "<your-data>/packages/omni-ring")
from train_reservoir import Reservoir, encode, load_all
from omni_ring.shards import TernaryEncoder
OUTF = Path("./t1/t1_result.json")
torch.manual_seed(0); torch.set_num_threads(4)
tok, special, data, ids_all_e9, TP, _nom, _cand = load_all()
net = Reservoir(TP); net.load_state_dict(torch.load(E9 / "e9_reservoir.pt", map_location="cpu", weights_only=True)); net.eval()
Wo_e9 = net.Wo.detach().clone()
with torch.no_grad():
    net.Wo = torch.nn.Parameter(torch.eye(Wo_e9.shape[0]))
def feats(texts, bs=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            out.append(net(encode(tok, special, texts[i:i + bs])).numpy())
    return np.concatenate(out).astype(np.float32)
t0 = time.time()
Fd = feats([data[d]["__data__"]["text"] for d in doc_ids]); Fq = feats([q for _, q in cand])
print(f"features {Fd.shape} {Fq.shape} in {time.time() - t0:.0f}s", flush=True)
tgt = set(targets); fit = np.array([d not in tgt for d in doc_ids])
def norm(x): return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
De9, Qe9 = norm(Fd @ Wo_e9.numpy()), norm(Fq @ Wo_e9.numpy())   # = e9's own output (its forward normalises F first)
res = {"float_e9": evaluate((Qe9.astype(np.float64) @ De9.T.astype(np.float64)), "float_e9"), "arms": {}}
print("float_e9", res["float_e9"], flush=True)
def tern(Z, s=0.33):
    tau = np.quantile(np.abs(Z), s, axis=1, keepdims=True)
    return (np.sign(Z) * (np.abs(Z) > tau)).astype(np.int8)
for nbytes in (64, 128):
    D = nbytes * 5
    enc = TernaryEncoder(768, D, 11 + D, 0.33)
    A = enc.A
    r = {}
    # posthoc: the shard store's path (e9 float output -> projection -> trits; float query through the same projection)
    r["posthoc"] = evaluate((Qe9.astype(np.float64) @ A.astype(np.float64)) @ enc.encode(De9).T.astype(np.float64), f"ph{D}")
    # ridge into the projected nomic space
    T_ = (Y @ A).astype(np.float64)
    X = Fd[fit].astype(np.float64)
    lam = 1e-2 * np.trace(X.T @ X) / X.shape[1]
    W = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ T_[fit])
    r["ridge_tern"] = evaluate((Fq.astype(np.float64) @ W) @ tern(Fd @ W).T.astype(np.float64), f"rt{D}")
    # straight-through fine-tuning (docs ternary in the forward pass, identity gradient)
    Wt = torch.tensor(W, dtype=torch.float32, requires_grad=True)
    Xf, Tf = torch.from_numpy(Fd[fit]), torch.from_numpy(T_[fit].astype(np.float32))
    opt = torch.optim.Adam([Wt], lr=1e-4)
    for ep in range(30):
        perm = torch.randperm(len(Xf))
        for i in range(0, len(Xf), 256):
            b = perm[i:i + 256]
            Z = Xf[b] @ Wt
            tau = torch.quantile(Z.abs(), 0.33, dim=1, keepdim=True)
            Zt = torch.sign(Z) * (Z.abs() > tau)
            Zs = Z + (Zt - Z).detach()                       # straight-through: forward = trits, backward = identity
            loss = (1 - torch.nn.functional.cosine_similarity(Zs, Tf[b])).mean() + (1 - torch.nn.functional.cosine_similarity(Z, Tf[b])).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    Wn = Wt.detach().numpy().astype(np.float64)
    r["ste_tern"] = evaluate((Fq.astype(np.float64) @ Wn) @ tern(Fd @ Wn.astype(np.float32)).T.astype(np.float64), f"st{D}")
    for k, v in r.items():
        v["G1_ge_0.78"] = v["heldout_hybrid_recall@5"] >= 0.78
        v["G2_ge_float_e9_minus_0.01"] = v["heldout_hybrid_recall@5"] >= res["float_e9"]["heldout_hybrid_recall@5"] - 0.01
    res["arms"][f"{nbytes}B"] = r
    print(nbytes, json.dumps(r), flush=True)
    json.dump(res, open(OUTF, "w"), indent=1)
print("done", time.time() - t0)
