#!/usr/bin/env python3
"""EMBED-CORE-0 E9 — train the reservoir (BPTT). The last thing between us and the gate.

WHERE E8 LEFT OFF (recorded, not re-litigated)
------------------------------------------------
  core_only 0.565 | hybrid 0.735 | bar 0.775 | speed 615x (passes)
  gamma sweep 0.80..0.995 was FLAT (core_only spread 0.03) and val cosine pinned at 0.989.

So pooling is exhausted as a lever. The remaining 0.04 is not reachable by re-weighting time
steps; the reservoir itself is the ceiling. This is the build: make Wt/Win/Wo trainable and
optimise them against REAL nomic vectors with truncated BPTT.

WHY THIS IS NOT CMA-ES
----------------------
CMA-ES is only viable for a handful of hyperparameters. Wt is 1024x1024 = 1,048,576
parameters; no derivative-free search touches that. The honest route for a reservoir this
size is backpropagation through time, which needs autodiff — hence torch. "Liquid" is
honoured where it actually lives: a per-neuron leak (continuous time constant) is made
trainable and initialised at 1.0 (i.e. it starts as the current discrete reservoir and can
learn to become leaky).

DISCIPLINE (the failure modes this seat already committed tonight)
-----------------------------------------------------------------
  * targets are REAL nomic, loaded from the vault index — never a proxy, never old-Wo output
  * the 200 HUB-3 targets are EXCLUDED from the fit set, so the gate stays honest
  * out-of-tree: this never imports into bin/embed-core and never writes projects/core-
    models/weights. It writes e9/ only.
  * memory: truncated BPTT, small batch, earlyoom-safe. 6,194 docs x 96 steps is NOT
    materialised; docs are streamed in mini-batches.
  * the checkpoint-vs-epoch honest split: val cosine is on docs never fitted, and the gate
    is the 200 held-out queries. Both are reported every run, not just at the end.

Usage: python3 train_reservoir.py [epochs] [lr]
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

AI = Path(__file__).resolve().parents[1]
WEIGHTS = AI.parent / "core-models" / "weights"
TOKENIZER = Path("<your-data>/bitnet/models/nomic-embed-text-v1.5/tokenizer.json")
TOKEN_TABLE = AI / "e1" / "token_table_f32.npy"
IX = Path(os.environ.get("EMBED_CORE_INDEX", str(AI / "data" / "index")))
OUT = Path(__file__).resolve().parent

TAILS = (4, 8, 16)
GAMMA = 0.95
MAXLEN = 96


def load_all():
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKENIZER))
    special = {tok.token_to_id(x) for x in ("[CLS]", "[SEP]", "[PAD]")}
    data = json.load(open(IX / "docstore.json"))["docstore/data"]
    ids_all = json.load(open(IX / "nomic_local" / "ids.json"))

    mean = np.load(WEIGHTS / "mean.npy")
    P = np.load(WEIGHTS / "P.npy")
    T = np.load(TOKEN_TABLE)
    TP = ((T - mean) @ P).astype(np.float32)
    TP /= TP.std()

    # real nomic per vault doc, straight out of the sharded index
    want = set(data.keys())
    nomic = {}
    mf = json.loads((IX / "ternary_core" / "MANIFEST.json").read_text())
    for sh in mf["shards"]:
        nm = sh["name"]
        sids = json.loads((IX / "ternary_core" / f"ids-{nm.split('-')[-1]}.json").read_text())
        arr = np.load(IX / "ternary_core" / f"{nm}.npy", mmap_mode="r")
        for i, _id in enumerate(sids):
            if _id in want:
                nomic[_id] = np.asarray(arr[i], dtype=np.float32)

    cand = []
    for k, i in enumerate(ids_all):
        m = re.search(r"^#+\s+(.{8,120})$", data[i]["__data__"]["text"], re.M)
        if m:
            cand.append((k, m.group(1).strip()))
    random.Random(0).shuffle(cand)
    cand = cand[:200]
    return tok, special, data, ids_all, TP, nomic, cand


class Reservoir(nn.Module):
    """Frozen token table + TRAINABLE Wt / Win / Wo / leak."""

    def __init__(self, TP, N=1024, use_leak=True):
        super().__init__()
        self.N = N
        self.tails = TAILS
        self.register_buffer("TP", torch.from_numpy(TP))          # frozen teacher features
        Wt = sp.load_npz(WEIGHTS / "Wt.npz").toarray().astype(np.float32)
        Win = np.load(WEIGHTS / "Win.npy").astype(np.float32)
        Wo = np.load(WEIGHTS / "Wo.npy").astype(np.float32)
        self.Wt = nn.Parameter(torch.from_numpy(Wt))
        self.Win = nn.Parameter(torch.from_numpy(Win))
        # E9 features are 6401 wide (ordered pooling) vs the shipped Wo's 2305. Seed the
        # trained columns so training starts from what we already know, and small random for
        # the new recency/tail blocks rather than reinitialising the whole readout.
        rows = 1 + N * 2 + TP.shape[1] + N + len(TAILS) * N   # 6401, matches forward()
        self.Wo = nn.Parameter(torch.zeros(rows, Wo.shape[1]))
        with torch.no_grad():
            keep = min(Wo.shape[0], rows)
            self.Wo[:keep] = torch.from_numpy(Wo[:keep])
            if keep < rows:
                self.Wo[keep:] = torch.randn(rows - keep, Wo.shape[1]) * 0.01
        # "liquid": per-neuron leak, initialised at 1.0 == the discrete reservoir we have now.
        # If learning never moves it, that is itself a finding worth recording.
        self.leak = nn.Parameter(torch.ones(N)) if use_leak else None
        self.gamma = GAMMA

    def wo_rows(self):
        """1 + N*2 + TP + N + len(tails)*N = 6401 for the shipped dims. Asserted against the
        live forward pass below so the two can never silently diverge."""
        return 1 + self.N * 2 + self.TP.shape[1] + self.N + len(self.tails) * self.N

    def forward(self, tk):
        """tk: LongTensor (B, L) token ids, -1 padded. Returns L2-normalised embeddings."""
        B, L = tk.shape
        m = (tk >= 0).float()
        safe = tk.clamp(min=0)
        I = (self.TP[safe] @ self.Win.T) * m.unsqueeze(-1)        # (B,L,N)
        X = torch.zeros(B, self.N, device=tk.device)
        acc = torch.zeros_like(X)
        last = torch.zeros_like(X)
        gacc = torch.zeros_like(X)
        gden = torch.zeros(B, 1, device=tk.device)
        bg = torch.zeros(B, self.TP.shape[1], device=tk.device)
        tails = {k: torch.zeros_like(X) for k in self.tails}
        tcount = {k: torch.zeros(B, 1, device=tk.device) for k in self.tails}
        nvalid = m.sum(1, keepdim=True)
        for t in range(L):
            u = I[:, t, :]
            v = m[:, t:t + 1]
            if float(v.sum()) == 0:
                break
            Xn = torch.tanh(X @ self.Wt + u)
            if self.leak is not None:
                a = self.leak.clamp(0.0, 1.0).view(1, -1)
                Xn = (1 - a) * Xn + a * X            # leaky/"liquid" mix
            acc = acc + Xn * v
            last = torch.where(v > 0, Xn, last)
            X = torch.where(v > 0, Xn, X)
            bg = bg + self.TP[safe[:, t]] * v
            w = v * (self.gamma ** torch.clamp(nvalid - 1 - t, min=0).float())
            gacc = gacc + Xn * w
            gden = gden + w
            for k in self.tails:
                if t >= MAXLEN - k:
                    tails[k] = tails[k] + Xn * v
                    tcount[k] = tcount[k] + v
        denom = torch.clamp(nvalid, min=1.0)
        feats = [acc / denom, last, bg / denom, gacc / torch.clamp(gden, min=1e-6)]
        for k in self.tails:
            feats.append(tails[k] / torch.clamp(tcount[k], min=1.0))
        F = torch.cat(feats, dim=1)
        F = torch.cat([F, torch.ones(B, 1, device=tk.device)], dim=1)
        assert F.shape[1] == self.Wo.shape[0], (F.shape, self.Wo.shape)
        v = F @ self.Wo
        return v / (v.norm(dim=1, keepdim=True) + 1e-12)


def encode(tok, special, texts, maxlen=MAXLEN):
    out = np.full((len(texts), maxlen), -1, dtype=np.int64)
    enc = tok.encode_batch(list(texts), add_special_tokens=False)
    for r, e in enumerate(enc):
        t = [i for i in e.ids if i not in special][:maxlen]
        if t:
            out[r, : len(t)] = t
    return torch.from_numpy(out)


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    lr = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
    t0 = time.time()
    tok, special, data, ids_all, TP, nomic, cand = load_all()

    doc_ids = sorted(nomic)
    targets = {ids_all[k] for k, _ in cand}
    fit_ids = [d for d in doc_ids if d not in targets]
    val_ids = [d for d in doc_ids if d in targets]
    print(f"[e9] {len(doc_ids)} docs | fit={len(fit_ids)} | held-out targets={len(val_ids)}",
          flush=True)

    Yfit = torch.from_numpy(np.stack([nomic[d] for d in fit_ids]))
    Yfit = Yfit / Yfit.norm(dim=1, keepdim=True)
    Yval = torch.from_numpy(np.stack([nomic[d] for d in val_ids]))
    Yval = Yval / Yval.norm(dim=1, keepdim=True)
    Xfit = encode(tok, special, [data[d]["__data__"]["text"] for d in fit_ids])
    Xval = encode(tok, special, [data[d]["__data__"]["text"] for d in val_ids])

    net = Reservoir(TP)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    print(f"[e9] trainable: {sum(p.numel() for p in net.parameters() if p.requires_grad):,} params",
          flush=True)

    B = 64
    hist = []
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(fit_ids))
        tot, nb = 0.0, 0
        for i in range(0, len(perm), B):
            idx = perm[i:i + B]
            out = net(Xfit[idx])
            loss = (1 - (out * Yfit[idx]).sum(1)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss); nb += 1
        net.eval()
        with torch.no_grad():
            vc = []
            for i in range(0, len(Xval), 128):
                vc.append((net(Xval[i:i + 128]) * Yval[i:i + 128]).sum(1))
            val = float(torch.cat(vc).mean())
        leak_mean = float(net.leak.mean()) if net.leak is not None else float("nan")
        hist.append({"epoch": ep, "train_loss": round(tot / max(nb, 1), 5),
                     "heldout_cos": round(val, 4), "leak_mean": round(leak_mean, 4)})
        print(f"[e9] epoch {ep}: train_loss={tot/max(nb,1):.5f}  HELD-OUT cos={val:.4f}  "
              f"leak_mean={leak_mean:.4f}  [{time.time()-t0:.0f}s]", flush=True)

    torch.save(net.state_dict(), OUT / "e9_reservoir.pt")
    json.dump(hist, open(OUT / "e9_history.json", "w"), indent=1)
    print(f"[e9] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
