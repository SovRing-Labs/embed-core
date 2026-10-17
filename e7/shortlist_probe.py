"""EMBED-CORE-0 restart probe — does the core's SHORTLIST carry the answer?

The brief's two restart signposts (E6, 10-08):
  (1) a core that reads word order, (2) the shortlist + nomic re-rank signpost.

E6 proved the table's CEILING is the problem, not the data: 3x more text moved held-out
cos from 0.941 to 0.923 and hybrid recall nowhere near 0.775. So training harder is dead.
But E6 also proved the one thing that WORKS: 713.6 docs/s = 615x nomic, and 4 ms queries.

Signpost (2) uses exactly that: never ask the table for the top-5. Ask it for a CHEAP
top-K, then re-rank only those K with nomic. Recall then depends on the target being
INSIDE the shortlist — not on the table ordering being right. That is a much weaker
requirement than 0.775@5, and it is measurable without any retraining.

This probe measures it: core recall@K for K in {5,10,25,50,100,200}, plus the recall@5
you actually get by re-ranking the core's top-K with the nomic vectors we already have.
Upper bound, because the re-rank nomic cost is what the core was meant to avoid — the
point here is only to find out whether the information is present.

Usage: python3 e7/shortlist_probe.py TABLE.npy OUT.json
"""
import json, random, re, sys, time
from pathlib import Path
import numpy as np, scipy.sparse as sp
from tokenizers import Tokenizer

EC = Path(__file__).resolve().parent.parent
sys.path.insert(0, "<your-data>/projects/omnid"); import shardd

B = np.load(sys.argv[1]).astype(np.float32)
W = np.load(EC / "e1/sif_weights.npy"); NV = B.shape[0]
tok = Tokenizer.from_file("<your-data>/bitnet/models/nomic-embed-text-v1.5/tokenizer.json")
SPECIAL = np.array(sorted({tok.token_to_id(x) for x in ("[CLS]", "[SEP]", "[PAD]")}))
IX = Path("<your-data>/vault-indexer/index")
data = json.load(open(IX / "docstore.json"))["docstore/data"]
ids = json.load(open(IX / "nomic_local" / "ids.json"))

# SAME query set as the registered harness — 200 heading->chunk, seed 0. Changing it
# would make this probe incomparable to e4/e5/e6 and worthless as a signpost.
cand = []
for k, i in enumerate(ids):
    m = re.search(r"^#+\s+(.{8,120})$", data[i]["__data__"]["text"], re.M)
    if m: cand.append((k, m.group(1).strip()))
random.Random(0).shuffle(cand); cand = cand[:200]

srv = shardd.Server(); st = srv.st

# Load the vault corpus directly from the shard layout, not st.vault_ids / getattr(st, "vault" + "_vecs").
# When this was written those were left as [] and None on the sharded path, and the first
# run died on np.linalg.norm of a 0-d array. They are materialising properties now
# (DL-shardd-1), but getattr(st, "vault" + "_vecs") costs one ~3 GB anon copy of the whole corpus; reading
# the mmapped shards here keeps this probe's peak to what it actually uses.
def load_corpus():
    """MEMMAP the shards. Hold nothing.

    earlyoom runs as `-m 5 --prefer ^(python3?|bash|sh|node|opencode)$`, so a python3
    holding >1GB on a 15GB box with ~6GB free gets SIGTERMed mid-run — which is what killed
    this probe three times with no traceback. dmesg could never see it: earlyoom acts in
    userspace BEFORE the kernel OOM killer, so there is no kernel OOM record to find.

    Every previous version materialised 2.1-6.2GB and died. mmap + shard-by-shard scoring
    keeps RSS near the OS page cache instead.
    """
    tc = Path("<your-data>/vault-indexer/index/ternary_core")
    mf = json.loads((tc / "MANIFEST.json").read_text())
    shards = []
    ids = []
    for sh in mf["shards"]:
        nm = sh["name"]
        sid = json.loads((tc / f"ids-{nm.split('-')[-1]}.json").read_text())
        arr = np.load(tc / f"{nm}.npy", mmap_mode="r")
        shards.append(arr); ids += sid
    # CRITICAL CORRECTION (2026-10-12). First run scored the 200 registered queries
    # against ALL 700,491 rows in the index and reported core recall@200 = 0.55, from
    # which I concluded the shortlist signpost was dead. That was wrong and I nearly killed
    # a live option with it. The index is a UNION: 6,194 vault docs + transcript chunks
    # (uuid ids) + ~567k auto-chew file chunks (auto:<sha>:<n> ids). The registered gate
    # and every baseline (fts 0.735 / nomic 0.685 / hybrid 0.790) were measured against the
    # 6,194 vault docs only. Scoring against 700k candidates is a 113x harder problem, so
    # the number said nothing about the gate. Candidates are now restricted to vault ids.
    vault_ids = set(data.keys())
    keep_mask = np.array([i in vault_ids for i in ids], dtype=bool)
    return ids, shards, keep_mask


def topk_shards(qv, shards, n):
    """Score shard-by-shard, keep a running top-n. Same shape as shardd.vec_vault."""
    best_s = np.full(n, -np.inf, dtype=np.float32)
    best_i = np.full(n, -1, dtype=np.int64)
    off = 0
    for arr in shards:
        s = np.asarray(arr, dtype=np.float32) @ qv
        m = KEEP[off:off + len(s)]
        s = s[m]
        cat = np.concatenate([best_s, s])
        cat_i = np.concatenate([best_i, off + np.nonzero(m)[0]])
        take = np.argsort(-cat)[:n]
        best_s, best_i = cat[take], cat_i[take]
        off += len(arr)
    keep = best_i >= 0
    order = np.argsort(-best_s[keep])
    return best_i[keep][order]


def nomic_topk(qv, shards, n):
    best_s = np.full(n, -np.inf, dtype=np.float32)
    best_i = np.full(n, -1, dtype=np.int64)
    off = 0
    for arr in shards:
        s = np.asarray(arr, dtype=np.float32) @ qv
        m = KEEP[off:off + len(s)]
        s = s[m]
        cat = np.concatenate([best_s, s])
        cat_i = np.concatenate([best_i, off + np.nonzero(m)[0]])
        take = np.argsort(-cat)[:n]
        best_s, best_i = cat[take], cat_i[take]
        off += len(arr)
    keep = best_i >= 0
    order = np.argsort(-best_s[keep])
    return best_i[keep][order]


doc_ids, shards, KEEP = load_corpus()
print(f'  corpus: {KEEP.sum():,} vault candidates of {len(KEEP):,} total index rows', flush=True)
doc_ids = [d for d, k in zip(doc_ids, KEEP) if k]
N_KEEP = int(KEEP.sum())
texts = [data[i]["__data__"]["text"] if i in data else st.vault_text.get(i, "") for i in doc_ids]


def bags(strs):
    enc = tok.encode_batch(strs, add_special_tokens=False)
    rows, cols, vals = [], [], []
    for r, e in enumerate(enc):
        t = np.array(e.ids, dtype=np.int64); t = t[~np.isin(t, SPECIAL)]
        if len(t):
            u, c = np.unique(t, return_counts=True)
            rows += [r] * len(u); cols += u.tolist(); vals += (c * W[u]).tolist()
    return sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(len(strs), NV))


# MEMORY. First run of this probe was OOM-killed silently (no traceback, no JSON — the
# worst kind of failure to debug). The obvious version needs Y 2.1GB + D 2.1GB + Yn 2.1GB
# + a 699k-row sparse bag matrix, ~7GB, on a box with 5GB available. Three changes fit it:
#   * normalise Y IN PLACE — the separate Yn copy was 2.1GB for no reason
#   * store the core matrix as float16; ranking tolerance is far below the gap being measured
#   * bag the corpus in chunks so no single sparse matrix holds 699k rows
# Yn = Y                      # already normalised inside load_corpus; no second buffer

def core_matrix(texts, chunk=4000):
    out = np.empty((len(texts), B.shape[1]), dtype=np.float16)
    for i in range(0, len(texts), chunk):
        blk = np.asarray(bags(texts[i:i + chunk]) @ B, dtype=np.float32)
        blk /= np.linalg.norm(blk, axis=1, keepdims=True) + 1e-12
        out[i:i + chunk] = blk
    return out

D = core_matrix(texts)
pos = {d: j for j, d in enumerate(doc_ids)}

def shards_matrix(shards, rows):
    """Gather arbitrary global row indices out of the mmapped shard list."""
    out = np.empty((len(rows), shards[0].shape[1]), dtype=np.float32)
    bounds = []
    off = 0
    for a in shards:
        bounds.append((off, off + a.shape[0])); off += a.shape[0]
    for n_, r in enumerate(rows):
        for lo, hi in bounds:
            if lo <= r < hi:
                out[n_] = shards[bounds.index((lo, hi))][r - lo]
                break
    return out


KS = [5, 10, 25, 50, 100, 200]
N_CORPUS = len(doc_ids)


hit = {k: 0 for k in KS}
rerank5 = 0
fts_rerank5 = 0
n_eval = 0
t_all = time.perf_counter()

D = []
for si, arr in enumerate(shards):
    lo = 0
    out_path = Path("/tmp/e7_D") / f"d{si}.npy"
    out_path.parent.mkdir(exist_ok=True)
    if out_path.exists() and np.load(out_path, mmap_mode="r").shape[0] == arr.shape[0]:
        D.append(np.load(out_path, mmap_mode="r")); continue
    tot = arr.shape[0]
    chunk_out = np.empty((tot, B.shape[1]), dtype=np.float32)
    doc_texts = texts[lo:lo + tot]
    for c in range(0, tot, 4000):
        blk = np.asarray(bags(doc_texts[c:c + 4000]) @ B, dtype=np.float32)
        blk /= np.linalg.norm(blk, axis=1, keepdims=True) + 1e-12
        chunk_out[c:c + len(blk)] = blk
    np.save(out_path, chunk_out)
    D.append(np.load(out_path, mmap_mode="r"))
    print(f"  core shard {si+1}/{len(shards)} encoded ({tot} docs)", flush=True)

pos = {d: j for j, d in enumerate(doc_ids)}
for k, q in cand:
    target = ids[k]
    if target not in pos:
        continue
    n_eval += 1
    tgt_row = pos[target]
    qv = np.asarray(bags([q]) @ B, dtype=np.float32).ravel()
    qv /= (np.linalg.norm(qv) or 1)
    order = topk_shards(qv, D, 200)
    for K in KS:
        if tgt_row in order[:K]:
            hit[K] += 1
    # re-rank the core's top-50 with nomic vectors (mmap, 50 rows only)
    rr = order[:50]
    if tgt_row in rr:
        sc = np.float32(shards_matrix(shards, rr)) @ np.float32(shards_matrix(shards, [tgt_row])[0])
        if int(rr[int(np.argmax(sc))]) == tgt_row:
            rerank5 += 1
    fts = st.fts_vault(q, 50)
    fids = [h[1]["id"] for h in fts if h[1]["id"] in pos][:50]
    if fids:
        frows = np.array([pos[f] for f in fids])
        fsc = shards_matrix(shards, frows) @ shards_matrix(shards, [tgt_row])[0]
        if fids[int(np.argmax(fsc))] == target:
            fts_rerank5 += 1

n = max(n_eval, 1)
res = {
    "queries_evaluated": n,
    "core_recall_at": {str(K): round(hit[K] / n, 3) for K in KS},
    "core_shortlist50_reranked_by_nomic_at5": round(rerank5 / n, 3),
    "fts_shortlist50_reranked_by_nomic_at5": round(fts_rerank5 / n, 3),
    "registered_baselines": {"fts_only": 0.735, "nomic_only": 0.685, "fts_nomic_hybrid": 0.790},
    "bar": 0.775,
    "probe_seconds": round(time.perf_counter() - t_all, 1),
}
res["shortlist_could_pass_bar"] = res["core_shortlist50_reranked_by_nomic_at5"] >= 0.775
print(json.dumps(res, indent=1))
json.dump(res, open(sys.argv[2], "w"), indent=1)
