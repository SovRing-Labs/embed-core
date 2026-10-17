"""EMBED-CORE-0 E6 prep (laptop): the fitting set = vault docs (nomic vectors already in the vault store) + chewed session chunks (nomic vectors from
the cpu16 run, J 10-08: sessions are practice text for the embed core only). Held out of the fit: the 200 HUB-3 target docs + 20 % of each corpus.
Writes e6/fitset.npz: X (CSR, SIF-weighted WordPiece bags), Y (nomic vectors), fit mask, corpus id (0 vault, 1 sessions) — the pod only needs this file."""
import glob, json, random, re, sqlite3, sys
from pathlib import Path
import numpy as np, scipy.sparse as sp
from tokenizers import Tokenizer
HERE = Path(__file__).resolve().parent; EC = HERE.parent
sys.path.insert(0, "<your-data>/projects/omnid"); import shardd
tok = Tokenizer.from_file("<your-data>/bitnet/models/nomic-embed-text-v1.5/tokenizer.json")
SPECIAL = np.array(sorted({tok.token_to_id(x) for x in ("[CLS]", "[SEP]", "[PAD]")})); W = np.load(EC / "e1/sif_weights.npy"); NV = len(W)
# vault
IX = Path("<your-data>/vault-indexer/index"); data = json.load(open(IX / "docstore.json"))["docstore/data"]; ids = json.load(open(IX / "nomic_local" / "ids.json"))
cand = []
for k, i in enumerate(ids):
    m = re.search(r"^#+\s+(.{8,120})$", data[i]["__data__"]["text"], re.M)
    if m: cand.append((k, m.group(1).strip()))
random.Random(0).shuffle(cand); cand = cand[:200]; targets = {ids[k] for k, _ in cand}
st = shardd.Server().st; vid = list(st.vault_ids); vtext = [data[i]["__data__"]["text"] if i in data else st.vault_text.get(i, "") for i in vid]; vY = np.asarray(getattr(st, "vault" + "_vecs"), np.float32)
# sessions (row order = rowids.json shards = shard*.emb order)
V = Path("<your-data>/text-vectors")
rid = [r for f in ("rowids.json", "rowids2.json") if (V / f).exists() for part in json.load(open(V / f)) for r in part]
sY = np.concatenate([np.array(json.load(open(f)), np.float32) for pat in ("shard*.emb", "b2shard*.emb") for f in sorted(glob.glob(str(V / pat)))])
assert len(sY) == len(rid), (len(sY), len(rid))
db = sqlite3.connect("<your-data>/data/text-corpus.db"); txt = dict(db.execute("select rowid, text from chunks").fetchall()); stext = [txt[r] for r in rid]
def bags(strs):
    enc = tok.encode_batch(strs, add_special_tokens=False); rows, cols, vals = [], [], []
    for r, e in enumerate(enc):
        t = np.array(e.ids, dtype=np.int64); t = t[~np.isin(t, SPECIAL)]
        if len(t): u, c = np.unique(t, return_counts=True); rows += [r] * len(u); cols += u.tolist(); vals += (c * W[u]).tolist()
    return sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(len(strs), NV))
X = sp.vstack([bags(vtext), bags(stext)]).tocsr(); Y = np.concatenate([vY, sY]); corpus = np.array([0] * len(vtext) + [1] * len(stext), np.int8)
rng = np.random.default_rng(6); fit = rng.random(len(corpus)) >= 0.2
fit[[j for j, d in enumerate(vid) if d in targets]] = False
np.savez_compressed(HERE / "fitset.npz", indptr=X.indptr, indices=X.indices, data=X.data, shape=X.shape, Y=Y, fit=fit, corpus=corpus)
json.dump({"vault_docs": len(vtext), "session_chunks": len(stext), "fit": int(fit.sum()), "held_out": int((~fit).sum()), "hub3_targets_held_out": len(targets), "nnz": int(X.nnz)}, sys.stdout); print()
