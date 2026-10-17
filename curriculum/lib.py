"""EMBED-CURRIC-0 shared helpers: paths, level table, split rule, jsonl io, embedder loader, clean filters."""
import hashlib
import json
import re
from importlib.machinery import SourceFileLoader
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
LEVELS_DIR = HERE / "levels"
PREK_DIR = HERE / "prek"
SEED = 1013                 # fixed seed for every sample + the P2 split seed (heldout-1013)
HELDOUT_MOD = 5             # 1 group in 5 is held out of every level, by hash of the GROUP (never by row)
SIGMA = 0.017               # P2 spread of s1 hybrid recall over 5 seeds (heldout-s1.json: 0.785 +- 0.017)
LEVEL_IDS = ["L0", "L1", "L2", "L3", "L4"]

# level -> (beat-label procedure tag carried by every pair row, what the pair means)
PROCEDURE = {
    "L0": "nomic-neighbour",     # text -> its nomic nearest neighbour (what s1 already learned, kept for review)
    "L1": "title>body",          # docs/systems heading -> section body
    "L2": "concept>neighbour",   # atlas node -> linked node
    "L3": "task>artifact",       # board task -> its delivered artifact
    "L4": "question>answer",     # transcript question -> answer
}

# Text we never train on (clean-room FLEET_RULES "drop" + brief: no downloads, no private vault, no week-plan files).
PRIVATE_RX = re.compile(r"research/|Downloads|Obsidian|week-plan|/\x5fPads/|^\x5fPads/|/\x5fReviews/|^\x5fReviews/|agent-memory|"
                        r"\.env|\.key|credential|secret", re.I)
SECRET_RX = re.compile(r"sk-[A-Za-z0-9]{12,}|[A-Za-z0-9+]{32,}|password|passwd|api[_-]?key|bearer\s|"
                       r"[\w.+-]+@[\w-]+\.[\w.]+|\b\d{1,3}(?:\.\d{1,3}){3}\b|BEGIN [A-Z ]*PRIVATE", re.I)


def group_heldout(group):
    """Deterministic: the whole group (system / task / session / concept) is held out or trained, never split."""
    h = hashlib.sha256(f"{SEED}:{group}".encode()).digest()
    return int.from_bytes(h[:4], "big") % HELDOUT_MOD == 0


def norm(t):
    return re.sub(r"\s+", " ", t).strip()


def tid(*parts):
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def usable(text, lo=20):
    return len(text) >= lo and not SECRET_RX.search(text)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row(level, group, q, pos, negs, src, lic):
    """One pair row: query, positive, hard negatives, the level/procedure beat-label tag, provenance."""
    return {"id": tid(level, group, q, pos), "level": level, "label": f"{level}/{PROCEDURE[level]}", "group": group,
            "q": q, "pos": pos, "neg": negs, "src": src, "lic": lic, "split": "heldout" if group_heldout(group) else "train"}


class S1:
    """The live s1 core (bin/embed-core LiquidTernaryEmbedder): embed_texts(list[str]) -> unit-norm float32 [n, 768]."""
    name = "s1"

    def __init__(self):
        mod = SourceFileLoader("embed_core", str(REPO / "bin/embed-core")).load_module()
        self.e = mod.LiquidTernaryEmbedder.get()

    def embed_texts(self, texts, bs=256):
        import numpy as np
        out = [self.e.embed_texts(texts[i:i + bs]) for i in range(0, len(texts), bs)]
        return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


def load_embedder(spec):
    """'s1' or 'path/to/mod.py:Class' (Class() must have embed_texts)."""
    if spec == "s1":
        return S1()
    path, _, cls = spec.partition(":")
    mod = SourceFileLoader("emb_ext", path).load_module()
    return getattr(mod, cls)()
