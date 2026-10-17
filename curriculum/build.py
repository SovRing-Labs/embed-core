#!/usr/bin/env python3
"""EMBED-CURRIC-0 builder: levels L0-L4, the pre-K probe sets and schedule.json. Build only; no training, no fits.

  build.py levels [--l0 600]      write levels/L{0..4}.jsonl (+ levels/MANIFEST.json); L0 needs omnid (nomic) for ~0.17 s/text
  build.py prek                   write prek/probes.jsonl (7 abilities)
  build.py schedule               write schedule.json (helix: pass k = level k + review(<k) + preview(k+1)); needs levels/
  build.py expand --pas K         print the row ids/rows of pass K (deterministic, fixed seeds)
  build.py all                    levels + prek + schedule

Every sample is drawn with random.Random(SEED=1013). Held-out is decided per GROUP (system / concept / task / session) by hash,
so a held-out group never appears in a train row at any level; train rows sharing any exact text with a held-out row are dropped on top (finalize()).
Sources + licences: README.md. Text containing secrets/e-mails/IPs, or paths under research/, private planning, private reviews, Obsidian,
week-plan is never emitted (lib.PRIVATE_RX / SECRET_RX).
"""
import argparse
import ast
import json
import random
import re
import sqlite3
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lib  # noqa: E402
from lib import REPO, LEVELS_DIR, PREK_DIR, SEED, LEVEL_IDS, row, usable, norm  # noqa: E402

CC = REPO / "control-center"
LIC_FLEET = "fleet-owned (J's repo; seat-authored text, kept for J's licence decision — see README)"
LIC_SESS = "J 2026-10-08: sessions are practice text for the embed core only (not for export)"
CAPS = {"L1": 3000, "L2": 1500, "L3": 1500, "L4": 1500}


def rng(tag):
    return random.Random(f"{SEED}:{tag}")


# ---------------------------------------------------------------- docs/systems sections (L0 pool, L1, pre-K)
def doc_sections():
    """[(system, file, heading, body)] from docs/systems/*/{MANUAL,DEVLOG}.md, frontmatter stripped."""
    out = []
    for p in sorted((REPO / "docs/systems").glob("*/*.md")):
        if p.name not in ("MANUAL.md", "DEVLOG.md"):
            continue
        text = p.read_text(errors="ignore")
        text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)
        parts = re.split(r"^(#{1,4} .+)$", text, flags=re.M)
        for i in range(1, len(parts) - 1, 2):
            head = parts[i].lstrip("# ").strip()
            body = norm(parts[i + 1])
            if len(head) >= 6 and len(body) >= 200 and usable(head) and usable(body[:1200]):
                out.append((p.parent.name, p.name, head, body[:1200]))
    return out


def build_L1(secs):
    by_sys = {}
    for s in secs:
        by_sys.setdefault(s[0], []).append(s)
    r, rows = rng("L1"), []
    for system, ss in sorted(by_sys.items()):
        for sysn, fn, head, body in ss:
            others = [x for x in ss if x[2] != head]
            r.shuffle(others)
            negs = [o[3] for o in others[:3]]
            if not negs:
                continue
            rows.append(row("L1", system, f"{system}: {head}", body, negs, f"docs/systems/{system}/{fn}", LIC_FLEET))
    return rows


def build_L0(secs, n):
    """Paragraph pool -> nomic doc vectors (omnid) -> each text's nomic nearest neighbour is the positive; a rank 8-20 neighbour is the
    hard negative. Pools are per split (group = system) so the neighbour never crosses train/held-out."""
    import numpy as np
    sys.path.insert(0, str(REPO / "projects/omnid"))
    from omnid_client import Omnid
    r = rng("L0")
    paras = []
    for system, fn, head, body in secs:
        for chunk in re.findall(r".{200,500}?(?:[.!?](?=\s)|$)", body):
            if usable(chunk):
                paras.append((system, f"docs/systems/{system}/{fn}", chunk.strip()))
    r.shuffle(paras)
    seen, pool = set(), []
    for p in paras:
        if p[2] not in seen:
            seen.add(p[2])
            pool.append(p)
    pool = pool[:n]
    cache = LEVELS_DIR / "L0.nomic.npy"
    LEVELS_DIR.mkdir(parents=True, exist_ok=True)
    texts = [p[2] for p in pool]
    key = lib.tid(*texts[:50], str(len(texts)))
    meta = LEVELS_DIR / "L0.nomic.key"
    if cache.exists() and meta.exists() and meta.read_text() == key:
        V = np.load(cache)
    else:
        c, vs = Omnid(timeout=60), []
        for i in range(0, len(texts), 32):
            vs += c.embed(texts[i:i + 32], kind="doc")
        V = np.asarray(vs, np.float32)
        V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        np.save(cache, V)
        meta.write_text(key)
    rows = []
    for held in (False, True):
        idx = [i for i, p in enumerate(pool) if lib.group_heldout(p[0]) == held]
        if len(idx) < 12:
            continue
        S = V[idx] @ V[idx].T
        np.fill_diagonal(S, -1)
        for a, i in enumerate(idx):
            order = np.argsort(-S[a])
            j = order[0]
            if S[a, j] < 0.55 or pool[idx[j]][0] == pool[i][0] and pool[idx[j]][2][:80] == pool[i][2][:80]:
                continue
            negs = [pool[idx[k]][2] for k in order[8:20]]
            r.shuffle(negs)
            rows.append(row("L0", pool[i][0], pool[i][2], pool[idx[j]][2], negs[:3], pool[i][1],
                            LIC_FLEET + "; neighbour labels from nomic-embed-text-v1.5 (Apache-2.0)"))
    return rows


# ---------------------------------------------------------------- L2 atlas
def build_L2():
    db = sqlite3.connect(f"file:{CC}/atlas.db?mode=ro", uri=True)
    nodes = {(k, i): (t or "", b or "") for k, i, t, b, pub in db.execute("select kind,id,title,body,public from nodes") if pub}
    def doc(key):
        t, b = nodes[key]
        return norm(f"{t}. {b}")[:600]
    links = {}
    for sk, s, rel, dk, d in db.execute("select src_kind,src,rel,dst_kind,dst from links"):
        if (sk, s) in nodes and (dk, d) in nodes:
            links.setdefault((sk, s), set()).add((dk, d))
    r, rows = rng("L2"), []
    by_kind = {}
    for k in nodes:
        by_kind.setdefault(k[0], []).append(k)
    for src, dsts in sorted(links.items()):
        for dst in sorted(dsts):
            held = lib.group_heldout(f"{dst[0]}:{dst[1]}")      # L2 group = the linked node: its text is held out whole
            pool = [x for x in by_kind[dst[0]] if x not in dsts and x != src and lib.group_heldout(f"{x[0]}:{x[1]}") == held]
            r.shuffle(pool)
            q, pos = doc(src), doc(dst)
            if usable(q) and usable(pos) and pool:
                rows.append(row("L2", f"{dst[0]}:{dst[1]}", q, pos, [doc(x) for x in pool[:3]], "control-center/atlas.db", LIC_FLEET))
    return rows


# ---------------------------------------------------------------- L3 board task -> artifact
ART_EXT = {".md", ".py", ".sh", ".rs", ".toml", ".json", ".txt", ".html"}


def build_L3():
    db = sqlite3.connect(f"file:{CC}/tasks.db?mode=ro", uri=True)
    q = """select t.id,t.title,coalesce(t.description,''),d.path,coalesce(d.project,'') from tasks t join deliverables d on d.task_id=t.id
           where t.status='done' and d.status='verified' order by t.id, d.id"""
    tasks, seen = [], set()
    for tid_, title, desc, path, proj in db.execute(q):
        if tid_ in seen or lib.PRIVATE_RX.search(path) or path.startswith(("spawn-briefs", "/")):
            continue
        p = REPO / path
        if p.suffix not in ART_EXT or not p.is_file() or p.stat().st_size > 400_000:
            continue
        body = norm(re.sub(r"\A---\n.*?\n---\n", "", p.read_text(errors="ignore"), flags=re.S))[:1500]
        qt = norm(f"{title}. {desc}")[:500]
        if len(body) >= 150 and usable(body) and usable(qt, 15):
            seen.add(tid_)
            grp = path.split("/")[2] if path.startswith("docs/systems/") else tid_   # docs/systems artifacts share L0/L1's split
            tasks.append((tid_, qt, body, path, p.suffix, grp))
    r, rows = rng("L3"), []
    for t in tasks:
        same = [x for x in tasks if x[0] != t[0] and x[4] == t[4] and lib.group_heldout(x[5]) == lib.group_heldout(t[5])]
        r.shuffle(same)
        rows.append(row("L3", t[5], t[1], t[2], [x[2] for x in same[:3]], f"tasks.db#{t[0]} -> {t[3]}", LIC_FLEET))
    return rows


# ---------------------------------------------------------------- L4 transcript question -> answer
USER_RX = re.compile(r"USER: (.*?)(?=\nASSISTANT:|\nTOOL |\nRESULT:|\nUSER:|\Z)", re.S)
ASST_RX = re.compile(r"ASSISTANT: (.*?)(?=\nUSER:|\nTOOL |\nRESULT:|\nASSISTANT:|\Z)", re.S)


def qa(text):
    u, a = USER_RX.search(text), ASST_RX.search(text)
    if not (u and a):
        return None
    q, ans = norm(u.group(1)), norm(a.group(1))
    if q.startswith("You are a peer agent") or not (25 <= len(q) <= 600) or len(ans) < 100:
        return None
    return q, ans[:800]


def build_L4(n):
    db = sqlite3.connect(f"file:{REPO}/data/text-corpus.db?mode=ro", uri=True)
    ids = [x for (x,) in db.execute("select rowid from chunks where project_id like '%user-project%' and project_id not like '%Obsidian%'")]
    r = rng("L4")
    r.shuffle(ids)
    rows, seenq = [], set()
    for rid in ids:
        if len(rows) >= n:
            break
        topics, text = db.execute("select topics,text from chunks where rowid=?", (rid,)).fetchone()
        p = qa(text)
        if not p or p[0] in seenq or not usable(p[0]) or not usable(p[1]):
            continue
        negs = []
        for o in (-2, -1, 1, 2, 3):
            got = db.execute("select topics,text from chunks where rowid=?", (rid + o,)).fetchone()
            if got and got[0] == topics:
                a = ASST_RX.search(got[1])
                if a and len(norm(a.group(1))) >= 100 and usable(norm(a.group(1))[:800]):
                    negs.append(norm(a.group(1))[:800])
        if negs:
            seenq.add(p[0])
            rows.append(row("L4", topics, p[0], p[1], negs[:3], f"data/text-corpus.db#{rid}", LIC_SESS))
    return rows


# ---------------------------------------------------------------- finalize: global exact-text leak removal
def finalize(levels):
    held_txt = set()
    for rows in levels.values():
        for x in rows:
            if x["split"] == "heldout":
                held_txt.update(norm(t) for t in [x["q"], x["pos"], *x["neg"]])
    dropped = {}
    for lv, rows in levels.items():
        keep = [x for x in rows if x["split"] == "heldout" or not ({norm(t) for t in [x["q"], x["pos"], *x["neg"]]} & held_txt)]
        dropped[lv] = len(rows) - len(keep)
        levels[lv] = keep
    return dropped


def cmd_levels(a):
    secs = doc_sections()
    levels = {"L1": build_L1(secs), "L2": build_L2(), "L3": build_L3(), "L4": build_L4(CAPS["L4"])}
    for lv, rows in levels.items():                       # cap by deterministic shuffle, keep both splits
        cap = CAPS[lv]
        if len(rows) > cap:
            r = rng(f"cap{lv}")
            held = [x for x in rows if x["split"] == "heldout"]
            tr = [x for x in rows if x["split"] == "train"]
            r.shuffle(held), r.shuffle(tr)
            nh = round(cap * len(held) / len(rows))
            levels[lv] = sorted(held[:nh] + tr[:cap - nh], key=lambda x: x["id"])
    levels["L0"] = build_L0(secs, a.l0)
    dropped = finalize(levels)
    man = {"seed": SEED, "heldout_mod": lib.HELDOUT_MOD, "leak_dropped_train_rows": dropped, "levels": {}}
    for lv in LEVEL_IDS:
        rows = sorted(levels[lv], key=lambda x: x["id"])
        lib.write_jsonl(LEVELS_DIR / f"{lv}.jsonl", rows)
        man["levels"][lv] = {"rows": len(rows), "train": sum(x["split"] == "train" for x in rows),
                             "heldout": sum(x["split"] == "heldout" for x in rows),
                             "groups": len({x["group"] for x in rows}), "label": f"{lv}/{lib.PROCEDURE[lv]}",
                             "sha256": lib.sha_file(LEVELS_DIR / f"{lv}.jsonl")}
    (LEVELS_DIR / "MANIFEST.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(man["levels"], indent=1), "leak-dropped TRAIN rows:", dropped)


# ---------------------------------------------------------------- pre-K: know its abilities
PARAPHRASE = [  # (query, paraphrase, same-domain topic-shift negative) — hand-written for this set
    ("How do I stop a long-running job safely?", "What is the safe way to terminate a job that runs for a long time?", "How do I start a job inside a capped unit?"),
    ("The build failed because a dependency is missing.", "A missing dependency caused the build to fail.", "The build succeeded and all tests passed."),
    ("Where are the test results saved?", "In which folder do the results of the tests get stored?", "Where is the test harness source kept?"),
    ("The server restarts every night at two.", "Every night at 2 a.m. the server is restarted.", "The server was installed on Monday."),
    ("Check the pod before reporting a run as live.", "Before you say a run is going, look at the pod.", "Check the laptop before closing the lid."),
    ("The embedding is computed from the token table.", "Token table lookups produce the embedding.", "The token table is stored as a numpy file."),
    ("Held-out data must never be trained on.", "Never train on the data that is held out.", "Training data must be shuffled with a fixed seed."),
    ("The queue is empty so the worker idles.", "With nothing in the queue, the worker does nothing.", "The queue grows when many tasks are staged."),
    ("Delete the cache before rebuilding.", "Remove the cache first, then rebuild.", "Rebuild the cache whenever the format changes."),
    ("Each ring turns one notch every tick.", "On every tick, every ring advances by a single step.", "Each ring has its own modulus."),
    ("The gate fails when an earlier level drops.", "If an earlier level gets worse, the gate does not pass.", "The gate passes when the target level rises."),
    ("Our laptop has limited memory.", "The memory on the laptop is limited.", "Our pod has sixteen cores."),
    ("Write the plan before you act.", "Plan in writing first, then act.", "Verify the result after you act."),
    ("The model reads the answer from a shard.", "A shard is where the model looks up the answer.", "Shards are written when a session ends."),
    ("Search the docs with a keyword.", "Use a keyword to look through the documentation.", "Write the docs after the feature ships."),
    ("The disk is almost full.", "There is very little free space left on the disk.", "The disk was formatted last year."),
    ("A failed run is a signpost, not a verdict.", "When a run fails it points the way instead of settling the question.", "A passed run is recorded in the log."),
    ("Run heavy batches on the pod first.", "Big batch jobs should go to the pod before anything else.", "Quick steps run on the laptop."),
    ("The curriculum spirals over earlier layers.", "Every pass of the curriculum goes back over the layers before it.", "The curriculum has five levels."),
    ("Start the service under a name.", "Give the service a name when you start it.", "Stop the service by its recorded process id."),
    ("The hash identifies the exact file contents.", "A hash tells you precisely what the file contains.", "The file was modified yesterday."),
    ("Rotate the log when it gets big.", "Once the log is large, rotate it.", "Read the log from the bottom up."),
    ("The vector is normalised to unit length.", "We scale the vector so its length is one.", "The vector has 768 dimensions."),
    ("Report the number, not just the verdict.", "Give the measured value together with pass or fail.", "Report the verdict in the first sentence."),
]
MULTI = [  # (english, translation, other english sentence)
    ("The test passed on the first run.", "La prueba pasó en la primera ejecución.", "The build failed on the second run."),
    ("Where is the configuration file?", "Où se trouve le fichier de configuration ?", "Who wrote the configuration file?"),
    ("The server is not responding.", "Der Server antwortet nicht.", "The server is responding slowly."),
    ("Save the results in a new folder.", "Guarda los resultados en una carpeta nueva.", "Delete the results from the old folder."),
    ("The model learns from its own answers.", "Le modèle apprend de ses propres réponses.", "The model forgets its earlier answers."),
    ("Check the memory before you start.", "Prüfe den Speicher, bevor du anfängst.", "Check the disk before you finish."),
    ("The queue has three tasks waiting.", "La cola tiene tres tareas esperando.", "The queue has no tasks waiting."),
    ("Close the file when you are done.", "Ferme le fichier quand tu as terminé.", "Open the file when you are ready."),
    ("The network connection is slow today.", "Die Netzwerkverbindung ist heute langsam.", "The network connection is fast today."),
    ("A small model can run on any computer.", "Un modelo pequeño puede ejecutarse en cualquier ordenador.", "A large model needs a data centre."),
    ("The result is saved at the end.", "Le résultat est enregistré à la fin.", "The result is shown at the start."),
    ("Every row has a label.", "Jede Zeile hat eine Bezeichnung.", "Every column has a width."),
]


def sentences(secs):
    out = []
    for system, fn, head, body in secs:
        for s in re.split(r"(?<=[.!?])\s+", body):
            if 60 <= len(s) <= 220 and usable(s) and s[0].isupper() and "|" not in s and "`" not in s:
                out.append((system, s))
    return out


def ascii_fold(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def python_functions():
    out = []
    for p in sorted((REPO / "bin").glob("*"))[:400]:
        if not p.is_file() or p.stat().st_size > 200_000:
            continue
        try:
            src = p.read_text(errors="ignore")
            if not src.startswith("#!/usr/bin/env python"):
                continue
            tree = ast.parse(src)
        except (SyntaxError, ValueError, UnicodeDecodeError):
            continue
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef):
                d = ast.get_docstring(fn)
                if not d or len(d.split("\n")[0]) < 30 or len(d.split("\n")[0]) > 160:
                    continue
                code = ast.get_source_segment(src, fn) or ""
                code = code.replace(ast.get_source_segment(src, fn.body[0]) or "\0", "", 1)
                code = "\n".join(l for l in code.split("\n") if l.strip())[:600]
                if len(code) >= 120 and usable(code) and usable(d):
                    out.append((p.name, d.split("\n")[0].strip(), code))
    return out


def task_ids():
    db = sqlite3.connect(f"file:{CC}/tasks.db?mode=ro", uri=True)
    return [(i, norm(t)[:100]) for i, t in db.execute("select id,title from tasks where length(id) between 6 and 24 and id glob '*[0-9]*'")
            if usable(t, 10)]


def near_id(i, ids, r):
    """An id one edit away if one exists on the board, else mutate the last digit."""
    c = [x for x, _ in ids if x != i and len(x) == len(i) and sum(a != b for a, b in zip(x, i)) == 1]
    if c:
        return r.choice(c)
    return i[:-1] + str((int(i[-1]) + 1) % 10) if i[-1].isdigit() else i + "2"


def cmd_prek(a):
    secs = doc_sections()
    sents = sentences(secs)
    r = rng("prek")
    r.shuffle(sents)
    per = a.per
    P = []

    def add(ability, q, pos, neg, src):
        P.append({"id": lib.tid(ability, q, pos), "level": "preK", "ability": ability, "q": q, "pos": pos, "neg": [neg], "src": src})
    for k, (sy, s) in enumerate(sents[:per]):                                      # exact duplicate
        add("exact-dup", s, s, sents[per + k][1], f"docs/systems/{sy}")
    for k, (sy, s) in enumerate(sents[per:2 * per]):                               # near duplicate: case/space/punct + one word cut
        w = s.split()
        cut = " ".join(w[:len(w) // 2] + w[len(w) // 2 + 1:])
        edit = [s.lower().replace(" ", "  "), s.rstrip(".") + " !", cut][k % 3]
        same = [x for y, x in sents if y == sy and x != s]
        add("near-dup", s, edit, r.choice(same) if same else sents[2 * per + k][1], f"docs/systems/{sy}")
    for q, p, n in PARAPHRASE[:per]:                                               # paraphrase vs topic-shift
        add("paraphrase", q, p, n, "hand-written")
    L = [s for s in secs]
    r.shuffle(L)
    for k, (sy, fn, head, body) in enumerate(L[:per]):                             # short <-> long
        o = L[(k + 7) % len(L)]
        if k % 2:
            add("short-long", head, body, o[3], f"docs/systems/{sy}")
        else:
            add("short-long", body[:700], head, o[2], f"docs/systems/{sy}")
    # prose <-> code
    fns = python_functions()
    r.shuffle(fns)
    for k, (fn, d, code) in enumerate(fns[:per]):
        add("code-prose", d, code, fns[(k + 11) % len(fns)][2], f"bin/{fn}")
    for k, (en, tr, o) in enumerate(MULTI):                                        # multilingual + unicode variants
        add("multilingual", en, tr, o, "hand-written")
    for k, (sy, s) in enumerate(sents[2 * per:3 * per]):
        v = [unicodedata.normalize("NFD", s + " café naïve"), ascii_fold(s + " café naïve"), s.replace(" ", " ") ][k % 3]
        add("multilingual", s + " café naïve", v, sents[3 * per + k][1] + " café naïve", f"docs/systems/{sy}")
    ids = task_ids()
    r.shuffle(ids)
    for k, (i, title) in enumerate(ids[:per]):                                     # numbers / ids
        add("numbers-ids", f"what is {i}", f"{i}: {title}", f"{near_id(i, ids, r)}: {title}", "tasks.db")
    lib.write_jsonl(PREK_DIR / "probes.jsonl", P)
    cnt = {}
    for x in P:
        cnt[x["ability"]] = cnt.get(x["ability"], 0) + 1
    print("prek probes:", len(P), cnt)


# ---------------------------------------------------------------- helix schedule
REVIEW_FRAC, PREVIEW_FRAC, MIN_REVIEW = 0.15, 0.10, 20


def pass_plan(man, k):
    lv = LEVEL_IDS[k]
    nxt = LEVEL_IDS[k + 1] if k + 1 < len(LEVEL_IDS) else None
    L = man["levels"]
    parts = [{"role": "focus", "level": lv, "n": L[lv]["train"], "seed": f"{SEED}:{k}:focus:{lv}"}]
    for e in LEVEL_IDS[:k]:
        parts.append({"role": "review", "level": e, "n": min(L[e]["train"], max(MIN_REVIEW, round(REVIEW_FRAC * L[e]["train"]))),
                      "seed": f"{SEED}:{k}:review:{e}"})
    if nxt:
        parts.append({"role": "preview", "level": nxt, "n": min(L[nxt]["train"], max(MIN_REVIEW, round(PREVIEW_FRAC * L[nxt]["train"]))),
                      "seed": f"{SEED}:{k}:preview:{nxt}"})
    return {"pass": k, "level": lv, "gate": {"target": lv, "earlier": LEVEL_IDS[:k], "sigma": lib.SIGMA}, "parts": parts,
            "next": nxt or "L5 (not built: slot reserved; the circle closes when L5 exists)"}


def cmd_schedule(a):
    man = json.load(open(LEVELS_DIR / "MANIFEST.json"))
    sch = {"seed": SEED, "design": "helix (circular): pass k = all train rows of level k + review sample of every level < k + preview of level k+1",
           "review_frac": REVIEW_FRAC, "preview_frac": PREVIEW_FRAC, "min_sample": MIN_REVIEW,
           "level_sha256": {lv: man["levels"][lv]["sha256"] for lv in LEVEL_IDS},
           "passes": [pass_plan(man, k) for k in range(len(LEVEL_IDS))]}
    (lib.HERE / "schedule.json").write_text(json.dumps(sch, indent=1) + "\n")
    for p in sch["passes"]:
        print(p["pass"], p["level"], [(x["role"], x["level"], x["n"]) for x in p["parts"]])


def cmd_expand(a):
    sch = json.load(open(lib.HERE / "schedule.json"))
    p = sch["passes"][a.pas]
    out = []
    for part in p["parts"]:
        tr = sorted([x for x in lib.read_jsonl(LEVELS_DIR / f"{part['level']}.jsonl") if x["split"] == "train"], key=lambda x: x["id"])
        out += [dict(x, role=part["role"]) for x in random.Random(part["seed"]).sample(tr, part["n"])]
    for x in out:
        print(json.dumps(x, ensure_ascii=False) if a.rows else f"{x['role']}\t{x['level']}\t{x['id']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("levels"); s.add_argument("--l0", type=int, default=600); s.set_defaults(f=cmd_levels)
    s = sp.add_parser("prek"); s.add_argument("--per", type=int, default=30); s.set_defaults(f=cmd_prek)
    sp.add_parser("schedule").set_defaults(f=cmd_schedule)
    s = sp.add_parser("expand"); s.add_argument("--pas", type=int, required=True); s.add_argument("--rows", action="store_true"); s.set_defaults(f=cmd_expand)
    s = sp.add_parser("all"); s.add_argument("--l0", type=int, default=600); s.add_argument("--per", type=int, default=30)
    s.set_defaults(f=lambda a: (cmd_levels(a), cmd_prek(a), cmd_schedule(a)))
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
