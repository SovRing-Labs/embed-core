# EMBED-CORE-0 — pre-registered gate (written 2026-10-08 BEFORE any core result; brief E0)
Harness: the HUB-3 set (`projects/omnid/hub3_eval.py` selection: 200 vault heading → chunk queries, random seed 0, same ids), same FTS5 store, same RRF.
Arms: nomic-only (reference 0.685) · core-only · hybrid FTS + nomic (reference 0.790) · hybrid FTS + core.
PASS = hybrid FTS + core recall@5 ≥ 0.775 AND core embedding throughput ≥ 100× nomic's (docs/s on the vault chunks, same machine).
Reported regardless: core-only recall@5, p50 query latency, table size (MB), ternary vs float variant. A FAIL keeps nomic in the hybrid; the numbers say why.
