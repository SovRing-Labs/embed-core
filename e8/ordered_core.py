#!/usr/bin/env python3
"""EMBED-CORE-0 E8 — order-sensitive pooling for the Liquid Ternary Core.

WHY (diagnosed from bin/embed-core, not guessed)
------------------------------------------------
The reservoir is ALREADY recurrent (`Wt`, rho=0.95). Order information exists. The bug is
in how the sequence is pooled into the readout:

    feats = concat([ mean_x (1024), last (1024), bg (256) ])     = 2304

  * `mean_x` is a mean over reservoir states. A mean is permutation-invariant: shuffling
    the tokens leaves it unchanged. ORDER-DEAD.
  * `bg` is a mean of token PCA projections. Also a bag. ORDER-DEAD.
  * `last` is the single most recent state. Order-aware, but one time step is a weak signal.

So 1280 of 2304 feature dims (56%) cannot see word order at all, and the remaining 1024 see
only the final step. That, not parameter count, is why E6 held-out cosine capped near 0.92
and why core recall@5 is 0.435 while FTS alone is 0.735.

WHAT THIS CHANGES (pooling only)
---------------------------------
  * `mean_x` -> recency-weighted mean: sum_t Xn_t * gamma^(T-t). Order enters through the
    weight, so a token's POSITION matters, not just its presence. One hyperparameter.
  * multi-scale tail means over the last {4, 8, 16} tokens, concatenated.
  * `last` and `bg` unchanged, so the existing readout still has its old features.

Nothing about the reservoir changes. `Wt`, `Win`, `P`, and the token table are loaded
frozen and reused. `Wo` is refit by closed-form ridge — no backprop, no training loop.

Cost of the forward pass: one extra pass is needed for the tail means, so ~1.15x. The
recency weight is applied inside the existing loop, so it is essentially free.

HONEST EXPECTATION (stated before running, so it cannot be moved afterwards)
--------------------------------------------------------------------------
If held-out cosine still caps near 0.92 and core recall@5 stays near 0.435, then the
RESERVOIR is the limit, not the pooling, and the honest answer is to stop rather than keep
adding blocks. That is a real possible outcome and it is recorded as one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from tokenizers import Tokenizer

AI = Path(__file__).resolve().parents[1]
WEIGHTS = AI.parent / "core-models" / "weights"
TOKENIZER = Path("<your-data>/bitnet/models/nomic-embed-text-v1.5/tokenizer.json")
TOKEN_TABLE = AI / "e1" / "token_table_f32.npy"

TAILS = (4, 8, 16)


class OrderedCore:
    """Same frozen reservoir as LiquidTernaryEmbedder, with order-sensitive pooling."""

    def __init__(self, gamma: float = 0.97):
        self.tok = Tokenizer.from_file(str(TOKENIZER))
        self.special = {self.tok.token_to_id(x) for x in ("[CLS]", "[SEP]", "[PAD]")}
        self.mean = np.load(WEIGHTS / "mean.npy")   # (768,) — see TP below
        self.P = np.load(WEIGHTS / "P.npy")
        self.Win = np.load(WEIGHTS / "Win.npy")
        self.Wt = sp.load_npz(WEIGHTS / "Wt.npz")
        self.mu = np.load(WEIGHTS / "mu.npy")
        self.sd = np.load(WEIGHTS / "sd.npy")
        self.meta = json.load(open(WEIGHTS / "meta.json"))
        self.N = self.meta["N"]
        self.L = 96
        self.gamma = gamma

        T = np.load(TOKEN_TABLE)
        TP = ((T - self.mean) @ self.P).astype(np.float32)   # (30522,768)@(768,256)->(30522,256)
        TP /= TP.std()
        self.TP = TP
        I = (TP @ self.Win.T.astype(np.float32)).astype(np.float32)
        I /= I.std()
        self.I = I

    def features(self, texts: list[str]) -> np.ndarray:
        """Return the 2304-dim baseline features (same as live embed-core)."""
        return self._pool(texts, ordered=False)

    def ordered_features(self, texts: list[str]) -> np.ndarray:
        """Recency-weighted mean + multi-scale tail means, concatenated after the baseline."""
        return self._pool(texts, ordered=True)

    def _pool(self, texts: list[str], ordered: bool) -> np.ndarray:
        enc = self.tok.encode_batch(texts, add_special_tokens=False)
        b = len(texts)
        tk = np.full((b, self.L), -1, dtype=np.int32)
        for r, e in enumerate(enc):
            t = [i for i in e.ids if i not in self.special][: self.L]
            tk[r, : len(t)] = t
        m = tk >= 0
        nvalid = m.sum(axis=1, keepdims=True)      # actual token count per row
        X = np.zeros((b, self.N), dtype=np.float32)
        acc = np.zeros_like(X)
        last = np.zeros_like(X)
        tail = {k: np.zeros_like(X) for k in TAILS}
        cnt = {k: np.zeros((b, 1), dtype=np.float32) for k in TAILS}
        gacc = np.zeros_like(X) if ordered else None
        gden = np.zeros((b, 1), dtype=np.float32) if ordered else None
        bg = np.zeros((b, TP_SHAPE), dtype=np.float32)

        for t in range(self.L):
            valid = m[:, t:t + 1]
            if not valid.any():
                break
            u = np.where(valid, self.I[np.clip(tk[:, t], 0, None)], 0)
            Xn = np.tanh((X @ self.Wt) + u)
            acc += np.where(valid, Xn, 0)
            last = np.where(valid, Xn, last)
            X = np.where(valid, Xn, X)
            bg += np.where(valid, self.TP[np.clip(tk[:, t], 0, None)], 0)
            if ordered:
                pos = valid.astype(np.float32)
                # LENGTH-INVARIANT recency. Using gamma**(L-1-t) with L fixed at 96 meant a
                # 5-token heading got weights gamma^95..gamma^91 — all ~equal, so its recency
                # feature collapsed to a plain mean — while a 96-token doc spanned gamma^95..
                # gamma^0. Queries and docs therefore lived in DIFFERENT feature spaces and
                # core-only recall came out 0.000 despite 0.989 cosine on held-out docs.
                # Weight must be relative to the end of the ACTUAL sequence, then divided by
                # its own sum so it is scale-free.
                w = pos * (self.gamma ** np.maximum(nvalid - 1 - t, 0))
                gacc += Xn * w
                gden += w
                for k in TAILS:
                    if t >= self.L - k:
                        tail[k] += Xn * pos
                        cnt[k] += pos

        denom = np.maximum(m.sum(axis=1, keepdims=True), 1).astype(np.float32)
        mean_x = acc / denom
        bg /= denom
        feats = [mean_x, last, bg]
        if ordered:
            rec = gacc / np.maximum(gden, 1e-6)
            feats.append(rec)
            for k in TAILS:
                feats.append(tail[k] / np.maximum(cnt[k], 1.0))
        F = np.concatenate(feats, axis=1)
        if not ordered:
            F = (F - self.mu[:, : F.shape[1]]) / self.sd[:, : F.shape[1]]
        return F

    def readout(self, F: np.ndarray, Wo: np.ndarray) -> np.ndarray:
        """F must ALREADY carry its bias column.

        The bias is appended once, in fit_eval, before the ridge solve — so that Wo is fit
        against the same matrix it is later applied to. Adding it here as well produced a
        6401 vs 6402 width mismatch and, before that, would have silently fit a Wo one
        column narrower than the features at inference.
        """
        v = F @ Wo
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)


TP_SHAPE = 256
